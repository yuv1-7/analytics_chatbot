import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from agent.state import AgentState
from agent.tools import ALL_TOOLS, execute_sql_query
from core.schema_context import SCHEMA_CONTEXT
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("gemini_api_key"),
    temperature=0.7
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


class ParsedIntent(BaseModel):
    use_case: Optional[str] = Field(
        description="One of: NRx_forecasting, HCP_engagement, feature_importance_analysis, model_drift_detection, messaging_optimization"
    )
    models_requested: Optional[List[str]] = Field(
        default=None,
        description="List of model names or types mentioned (e.g., ['Random Forest', 'XGBoost', 'ensemble'])"
    )
    comparison_type: Optional[str] = Field(
        default=None,
        description="Type of comparison: performance, predictions, feature_importance, drift, ensemble_vs_base"
    )
    time_range: Optional[Dict[str, str]] = Field(
        default=None,
        description="Time period mentioned (e.g., {'period': 'last_month', 'start': '2024-09', 'end': '2024-10'})"
    )
    metrics_requested: Optional[List[str]] = Field(
        default=None,
        description="Metrics mentioned (e.g., ['RMSE', 'AUC', 'accuracy'])"
    )
    entities_requested: Optional[List[str]] = Field(
        default=None,
        description="Specific entities mentioned (e.g., HCP IDs, territory codes)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if query is ambiguous or missing critical information"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user for clarification"
    )
    requires_visualization: bool = Field(
        default=False,
        description="True if query implies need for charts/graphs"
    )
    references_previous_context: bool = Field(
        default=False,
        description="True if query refers to previous results (e.g., 'these', 'them', 'those models')"
    )
    resolved_models: Optional[List[str]] = Field(
        default=None,
        description="Model names after resolving references from conversation context"
    )


class SQLQuerySpec(BaseModel):
    sql_query: str = Field(
        description="The SQL SELECT query to execute"
    )
    query_purpose: str = Field(
        description="What this query retrieves (for documentation)"
    )
    expected_columns: List[str] = Field(
        description="Expected column names in result"
    )


def extract_model_names_from_text(text: str) -> List[str]:
    model_patterns = [
        r'model[_\s]id[:\s]+([a-f0-9\-]{36})',
        r'model[_\s]name[:\s]+([^\n,]+)',
        r'\b(RF_\w+|XGB_\w+|LGB_\w+|ENS_\w+)\b',
    ]
    
    models = []
    for pattern in model_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        models.extend([m.strip() for m in matches])
    
    return list(set(models))


def query_understanding_agent(state: AgentState) -> dict:
    query = state['user_query']
    messages = state.get('messages', [])
    conversation_context = state.get('conversation_context', {})
    mentioned_models = state.get('mentioned_models', [])
    current_topic = state.get('current_topic')
    clarification_attempts = state.get('clarification_attempts', 0)
    last_query_summary = state.get('last_query_summary')
    
    context_parts = []
    
    if mentioned_models:
        context_parts.append(f"Previously mentioned models: {', '.join(mentioned_models[:5])}")
    
    if current_topic:
        context_parts.append(f"Current topic: {current_topic}")
    
    if last_query_summary:
        context_parts.append(f"Previous query: {last_query_summary}")
    
    recent_content = []
    for msg in messages[-3:]:
        if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
            if any(keyword in msg.content.lower() for keyword in ['model', 'rmse', 'auc', 'ensemble', 'nrx', 'hcp']):
                recent_content.append(msg.content[:200])
    
    if recent_content:
        context_parts.append(f"Recent discussion: {' | '.join(recent_content)}")
    
    context_summary = "\n".join(context_parts) if context_parts else "No prior context"
    
    system_msg = f"""You are a query parser for pharma commercial analytics. Parse user queries to extract:

USE CASES:
- NRx_forecasting: Predicting new prescriptions
- HCP_engagement: HCP response to marketing
- feature_importance_analysis: Understanding key drivers
- model_drift_detection: Detecting model performance changes over time
- messaging_optimization: Next-best-action for HCP targeting

MODELS (fuzzy matching - user may say "random forest", "RF", "forest", etc.):
- Base: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, Neural Network, Decision Tree
- Ensembles: stacking, boosting, bagging, meta-learner

COMPARISON TYPES:
- performance: Compare model metrics
- predictions: Compare actual predictions
- feature_importance: Compare which features matter most
- drift: Detect changes over time
- ensemble_vs_base: Why ensemble performs better/worse

METRICS: RMSE, MAE, R2, AUC-ROC, Accuracy, Precision, Recall, F1, TRx, NRx

CONVERSATION CONTEXT (USE THIS TO AVOID ASKING FOR INFO ALREADY GIVEN):
{context_summary}

CRITICAL CLARIFICATION RULES:
1. Check conversation context FIRST before asking for clarification
2. Current clarification attempts: {clarification_attempts}
3. If attempts >= 3, DO NOT ask for clarification. Make reasonable assumptions.
4. If user mentions "these models", "them", "those" - resolve from mentioned_models list
5. If user asks "compare performance" without specifying models:
   - If mentioned_models exists: Use those models
   - If use_case exists: Assume comparison across all models for that use case
   - Otherwise: Ask for clarification (only if attempts == 0)
6. If only vague query like "show me models" - assume they want to see all active models
7. For follow-up queries (when context exists), be MORE lenient - assume continuation of topic

REFERENCE RESOLUTION PRIORITY:
1. If "these/those/them/they" mentioned → Use mentioned_models from context
2. If "the ensemble" mentioned → Look for ensemble in context, or assume ensemble_vs_base comparison
3. If "that model" mentioned → Use last mentioned model
4. Only set needs_clarification=True if attempts==0 AND no context available AND query is truly ambiguous

Extract all relevant information. BE GENEROUS with assumptions when context exists.

Examples:
- "Compare Random Forest vs XGBoost for NRx forecasting" → use_case=NRx_forecasting, models=['Random Forest', 'XGBoost']
- "Compare these" (with context models=['RF', 'XGB']) → resolved_models=['RF', 'XGB'], comparison_type=performance
- "show performance" (with context use_case=NRx_forecasting) → use_case=NRx_forecasting, comparison_type=performance
- "why is ensemble worse" (no context, attempts=0) → needs_clarification=True
- "why is ensemble worse" (no context, attempts=1) → comparison_type=ensemble_vs_base, make assumptions"""
    
    structured_llm = llm.with_structured_output(ParsedIntent)
    
    context_messages = [SystemMessage(content=system_msg)]
    
    if messages:
        context_messages.extend(messages[-6:])
    
    context_messages.append(HumanMessage(content=query))
    
    result = structured_llm.invoke(context_messages)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('query_understanding')
    
    final_models = result.models_requested or []
    new_mentioned_models = []
    
    if result.references_previous_context:
        if result.resolved_models:
            final_models = result.resolved_models
        elif mentioned_models:
            final_models = mentioned_models
            result.resolved_models = mentioned_models
            result.needs_clarification = False
        elif clarification_attempts >= 3:
            final_models = []
            result.needs_clarification = False
            result.comparison_type = result.comparison_type or 'performance'
        else:
            result.needs_clarification = True
            result.clarification_question = "Which models would you like to compare? Please specify model names (e.g., Random Forest, XGBoost)."
    
    if clarification_attempts >= 3:
        result.needs_clarification = False
        
        if not result.use_case and current_topic:
            result.use_case = current_topic
        
        if not final_models and mentioned_models:
            final_models = mentioned_models
    
    if final_models:
        new_mentioned_models.extend(final_models)
    
    last_ai_messages = [msg for msg in messages[-6:] if isinstance(msg, AIMessage)]
    for msg in last_ai_messages:
        if hasattr(msg, 'content') and msg.content:
            extracted = extract_model_names_from_text(msg.content)
            new_mentioned_models.extend(extracted)
    
    summary_parts = []
    if result.use_case:
        summary_parts.append(f"use_case: {result.use_case}")
    if final_models:
        summary_parts.append(f"models: {', '.join(final_models[:3])}")
    if result.comparison_type:
        summary_parts.append(f"comparison: {result.comparison_type}")
    
    return {
        "messages": [HumanMessage(content=query)],
        "parsed_intent": result.dict(),
        "use_case": result.use_case,
        "models_requested": final_models,
        "comparison_type": result.comparison_type,
        "time_range": result.time_range,
        "metrics_requested": result.metrics_requested,
        "entities_requested": result.entities_requested,
        "needs_clarification": result.needs_clarification,
        "clarification_question": result.clarification_question,
        "requires_visualization": result.requires_visualization,
        "execution_path": execution_path,
        "mentioned_models": new_mentioned_models if new_mentioned_models else None,
        "current_topic": result.use_case if result.use_case else state.get('current_topic'),
        "last_query_summary": " | ".join(summary_parts) if summary_parts else None,
        "clarification_attempts": clarification_attempts
    }


def context_retrieval_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('context_retrieval')
    
    try:
        from core.vector_retriever import get_vector_retriever
        
        retriever = get_vector_retriever()
        
        user_query = state.get('user_query', '')
        use_case = state.get('use_case')
        comparison_type = state.get('comparison_type')
        models_requested = state.get('models_requested', [])
        
        relevant_docs = retriever.search_with_context(
            query=user_query,
            use_case=use_case,
            comparison_type=comparison_type,
            n_results=5
        )
        
        context_docs = []
        for doc in relevant_docs:
            context_docs.append({
                'doc_id': doc['doc_id'],
                'category': doc['metadata'].get('category', 'unknown'),
                'title': doc['metadata'].get('title', 'Untitled'),
                'content': doc['content'],
                'relevance_score': 1.0 - doc['distance'],
                'keywords': doc['metadata'].get('keywords', []),
                'source': 'vector_db'
            })
        
        print(f"Retrieved {len(context_docs)} relevant documents from vector DB")
        for doc in context_docs[:3]:
            print(f"  - {doc['title']} (relevance: {doc['relevance_score']:.3f})")
        
        return {
            "context_documents": context_docs,
            "execution_path": execution_path
        }
    
    except Exception as e:
        print(f"Vector DB retrieval failed: {e}")
        print("Falling back to empty context")
        
        context_docs = [{
            'type': 'error',
            'content': f'Vector DB retrieval failed: {str(e)}. Please run setup_vector_db.py to initialize the database.',
            'source': 'fallback'
        }]
        
        return {
            "context_documents": context_docs,
            "execution_path": execution_path
        }


def sql_generation_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('sql_generation')
    
    use_case = state.get('use_case')
    models_requested = state.get('models_requested', [])
    comparison_type = state.get('comparison_type')
    metrics_requested = state.get('metrics_requested', [])
    time_range = state.get('time_range')
    context_docs = state.get('context_documents', [])
    
    prompt = f"""
You are a SQL expert specializing in pharma commercial analytics databases. Your task is to generate a **valid PostgreSQL SELECT query** that accurately answers the user's question.

USER QUERY: {state['user_query']}

PARSED INTENT:
- Use Case: {use_case}
- Models Requested: {models_requested}
- Comparison Type: {comparison_type}
- Metrics: {metrics_requested}
- Time Range: {time_range}

{SCHEMA_CONTEXT}

{context_docs}

INSTRUCTIONS:
1. Generate a **valid PostgreSQL SELECT query** only.
2. Use appropriate **JOINs** to connect related tables.
3. Always **filter models with `is_active = true`**.
4. Use **`data_split = 'test'`** for performance metrics unless the user specifies otherwise.
5. Use the **`latest_model_executions` view** to retrieve the most recent model execution.
6. Handle **NULL values** appropriately.
7. Use **ILIKE** for all case-insensitive string comparisons.
8. Apply **meaningful ordering** for results.
9. Limit results to a **reasonable size** (e.g., `LIMIT 100`).

IMPORTANT – Fuzzy Matching for Model Names:
- Use partial matches, e.g.:
  - `ILIKE '%random forest%'` instead of `= 'Random Forest'`
  - `ILIKE '%xgboost%'` or `ILIKE '%xgb%'`
  - `ILIKE '%lightgbm%'` or `ILIKE '%lgb%'`

Example:
```sql
WHERE (m.model_name ILIKE '%random forest%' OR m.algorithm ILIKE '%random forest%')
  AND m.use_case ILIKE '%nrx%'
"""

    try:
        structured_llm = llm.with_structured_output(SQLQuerySpec)
        result = structured_llm.invoke(prompt)
        
        return {
            "generated_sql": result.sql_query,
            "sql_purpose": result.query_purpose,
            "expected_columns": result.expected_columns,
            "execution_path": execution_path,
            "messages": [AIMessage(content=f"Generated SQL query: {result.query_purpose}")]
        }
    
    except Exception as e:
        print(f"SQL generation failed: {e}")
        return {
            "generated_sql": None,
            "sql_purpose": "Failed to generate SQL",
            "expected_columns": [],
            "execution_path": execution_path,
            "messages": [AIMessage(content=f"Failed to generate SQL: {str(e)}")]
        }


def data_retrieval_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('data_retrieval')
    
    generated_sql = state.get('generated_sql')
    
    if not generated_sql:
        return {
            "execution_path": execution_path,
            "messages": [AIMessage(content="No SQL query was generated")]
        }
    
    response = execute_sql_query.invoke(generated_sql)
    tool_message=ToolMessage(content=json.dumps(response), tool_call_id=f"sql_exec_{id(generated_sql)}")
    
    extracted_models = extract_model_names_from_text(str(response.get('content','')))
    
    return {
        "messages": [tool_message],
        "execution_path": execution_path,
        "mentioned_models": extracted_models if extracted_models else None
    }


class AnalysisOutput(BaseModel):
    computed_metrics: Dict[str, Any] = Field(description="Key calculated metrics")
    patterns: List[str] = Field(description="Identified patterns and trends")
    anomalies: List[str] = Field(description="Unusual observations or outliers")
    statistical_summary: Dict[str, Any] = Field(description="Statistical summaries")


def analysis_computation_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('analysis_computation')
    
    messages = state.get('messages', [])
    tool_results = []
    
    from langchain_core.messages import ToolMessage
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                tool_results.append(result)
            except:
                continue
    
    if not tool_results or not any(r.get('success') for r in tool_results):
        return {
            "analysis_results": {
                'raw_data': tool_results,
                'summary': 'No data available for analysis',
                'computed_metrics': {},
                'insights': []
            },
            "execution_path": execution_path
        }
    
    successful_data = []
    for result in tool_results:
        if result.get('success') and result.get('data'):
            successful_data.extend(result['data'])
    
    analysis_prompt = f"""Analyze the following retrieved data and provide structured insights.

User Query: {state['user_query']}
Comparison Type: {state.get('comparison_type', 'general')}
Use Case: {state.get('use_case', 'unknown')}

Retrieved Data:
{json.dumps(successful_data[:50], indent=2, default=str)}

Provide analysis in the following structure:
1. **Key Metrics**: Calculate or extract important quantitative metrics
2. **Patterns**: Identify trends, correlations, or patterns in the data
3. **Anomalies**: Note any unusual values or outliers
4. **Statistical Summary**: Basic statistics (mean, min, max, std for numerical columns)
5. **Comparison Insights**: If comparing models/versions, highlight differences

Output as JSON with keys: computed_metrics (dict), patterns (list), anomalies (list), statistical_summary (dict)

Focus on actionable insights relevant to pharma commercial analytics."""

    try:
        structured_llm = llm.with_structured_output(AnalysisOutput)
        analysis = structured_llm.invoke(analysis_prompt)
        
        analysis_results = {
            'raw_data': tool_results,
            'computed_metrics': analysis.computed_metrics,
            'patterns': analysis.patterns,
            'anomalies': analysis.anomalies,
            'statistical_summary': analysis.statistical_summary,
            'data_row_count': len(successful_data)
        }
        
    except Exception as e:
        print(f"LLM analysis failed, falling back to basic analysis: {e}")
        
        analysis_results = {
            'raw_data': tool_results,
            'computed_metrics': _compute_basic_metrics(successful_data),
            'patterns': [],
            'anomalies': [],
            'statistical_summary': {},
            'data_row_count': len(successful_data),
            'analysis_error': str(e)
        }
    
    return {
        "analysis_results": analysis_results,
        "execution_path": execution_path
    }


def _compute_basic_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    
    metrics = {}
    
    metric_columns = ['rmse', 'mae', 'r2_score', 'r2', 'auc_roc', 'accuracy', 
                     'precision', 'recall', 'f1_score', 'drift_score',
                     'ensemble_advantage', 'metric_value', 'importance_score']
    
    for col in metric_columns:
        values = [row.get(col) for row in data if row.get(col) is not None]
        if values and all(isinstance(v, (int, float)) for v in values):
            metrics[col] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    model_names = list(set(row.get('model_name') for row in data if row.get('model_name')))
    if model_names:
        metrics['models_analyzed'] = model_names
    
    return metrics

class ChartSpec(BaseModel):
    chart_type: str = Field(description="bar, line, scatter, box, histogram")
    title: str
    x_axis: str
    y_axis: str
    color: Optional[str] = None
    facet: Optional[str] = None
    orientation: Optional[str] = Field(default="v", description="v or h")
    barmode: Optional[str] = Field(default=None, description="group, stack, or null")
    sort_by: Optional[str] = None
    sort_ascending: bool = True
    filter: Optional[Dict[str, Any]] = None
    explanation: str


class VizSpecOutput(BaseModel):
    strategy: str = Field(description="single_chart, multiple_charts, or no_visualization")
    reason: str
    charts: List[ChartSpec]
    warnings: List[str] = Field(default_factory=list)


VIZ_RULES = """
CRITICAL VISUALIZATION RULES:

1. NEVER stack metrics with different scales (RMSE, R2, MAE, AUC, etc.)
   - These metrics have different units and interpretations
   - Stacking them creates nonsensical visualizations

2. For model performance comparison with multiple metrics:
   - PREFERRED: Create separate chart for each metric
   - Each chart compares all models on ONE metric
   - Use barmode='group' if combining in one chart (rarely recommended)

3. For single metric comparison:
   - Simple bar chart: models on x-axis, metric on y-axis
   - Sort by metric value for clarity

4. For time series:
   - Use line charts
   - Color by entity (model, HCP, region)

5. For rankings/importance:
   - Horizontal bars sorted by value
   - Most important at top

6. For distributions:
   - Histograms for continuous
   - Box plots for comparison across groups

7. Data size limits:
   - >100 points: Suggest filtering or aggregation
   - High cardinality (>20 unique values): Aggregate or limit

8. Always be explicit:
   - Specify barmode for bar charts
   - Specify orientation for clarity
   - Provide clear titles and explanations
"""


def visualization_specification_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_spec')
    
    analysis_results = state.get('analysis_results', {})
    raw_data = analysis_results.get('raw_data', [])
    
    df = None
    for result in raw_data:
        if result.get('success') and result.get('data'):
            df = pd.DataFrame(result['data'])
            break
    
    if df is None or df.empty:
        print("No data for visualization")
        return {
            "execution_path": execution_path,
            "visualization_specs": [],
            "rendered_charts": []
        }
    
    if len(df) > 100:
        print(f"Limiting data from {len(df)} to 100 rows")
        df = df.head(100)
    
    user_query = state['user_query']
    
    data_summary = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'cardinality': {col: int(df[col].nunique()) for col in df.columns},
        'sample_data': df.head(5).to_dict('records')
    }
    
    prompt = f"""You are a visualization expert for pharma analytics.

USER QUERY: {user_query}

DATA STRUCTURE:
{json.dumps(data_summary, indent=2, default=str)}

{VIZ_RULES}

EXAMPLES:

Example 1 - Multi-metric model comparison:
Input: columns=['model_name', 'metric_name', 'metric_value']
       metric_name has values: ['rmse', 'mae', 'r2', 'mape']
Output:
{{
  "strategy": "multiple_charts",
  "reason": "Multiple metrics with different scales require separate charts for clarity",
  "charts": [
    {{
      "chart_type": "bar",
      "title": "RMSE Comparison Across Models",
      "x_axis": "model_name",
      "y_axis": "metric_value",
      "filter": {{"metric_name": "rmse"}},
      "barmode": null,
      "explanation": "Lower RMSE indicates better model accuracy"
    }},
    {{
      "chart_type": "bar",
      "title": "R² Score Comparison Across Models",
      "x_axis": "model_name",
      "y_axis": "metric_value",
      "filter": {{"metric_name": "r2"}},
      "barmode": null,
      "explanation": "Higher R² indicates better model fit"
    }}
  ]
}}

Example 2 - Single metric comparison:
Input: columns=['model_name', 'rmse']
Output:
{{
  "strategy": "single_chart",
  "reason": "Single metric allows direct comparison in one chart",
  "charts": [
    {{
      "chart_type": "bar",
      "title": "Model Performance - RMSE",
      "x_axis": "model_name",
      "y_axis": "rmse",
      "sort_by": "rmse",
      "sort_ascending": true,
      "explanation": "Models sorted by RMSE (lower is better)"
    }}
  ]
}}

Example 3 - Time series:
Input: columns=['date', 'model_name', 'metric_value']
Output:
{{
  "strategy": "single_chart",
  "reason": "Temporal data shows trends over time",
  "charts": [
    {{
      "chart_type": "line",
      "title": "Model Performance Over Time",
      "x_axis": "date",
      "y_axis": "metric_value",
      "color": "model_name",
      "explanation": "Performance trends by model"
    }}
  ]
}}

Now analyze the actual data and generate visualization specifications.
Return valid JSON only.
"""
    
    try:
        structured_llm = llm.with_structured_output(VizSpecOutput)
        viz_spec = structured_llm.invoke(prompt)
        
        validation_issues = validate_viz_spec(viz_spec, df)
        
        if validation_issues:
            print(f"Validation issues: {validation_issues}")
            viz_spec.warnings.extend(validation_issues)
        
        rendered_charts = []
        for chart_spec in viz_spec.charts:
            try:
                fig = render_chart(df, chart_spec)
                if fig:
                    rendered_charts.append({
                        'title': chart_spec.title,
                        'figure': fig,
                        'type': chart_spec.chart_type,
                        'explanation': chart_spec.explanation
                    })
            except Exception as e:
                print(f"Chart rendering failed: {e}")
                viz_spec.warnings.append(f"Failed to render {chart_spec.title}: {str(e)}")
        
        return {
            "visualization_specs": [spec.dict() for spec in viz_spec.charts],
            "rendered_charts": rendered_charts,
            "viz_strategy": viz_spec.strategy,
            "viz_reasoning": viz_spec.reason,
            "viz_warnings": viz_spec.warnings,
            "execution_path": execution_path,
            "requires_visualization": len(rendered_charts) > 0
        }
    
    except Exception as e:
        print(f"Visualization spec generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "visualization_specs": [],
            "rendered_charts": [],
            "viz_warnings": [f"Visualization generation failed: {str(e)}"],
            "execution_path": execution_path
        }


def validate_viz_spec(spec: VizSpecOutput, df: pd.DataFrame) -> List[str]:
    issues = []
    
    for chart in spec.charts:
        required_cols = [chart.x_axis, chart.y_axis]
        if chart.color:
            required_cols.append(chart.color)
        if chart.facet:
            required_cols.append(chart.facet)
        
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Column '{col}' not found in data")
        
        if chart.barmode == 'stack' and is_metrics_comparison(df, chart):
            issues.append(f"BLOCKED: Cannot stack metrics in '{chart.title}' - switching to grouped")
            chart.barmode = 'group'
        
        if chart.filter:
            filter_col = list(chart.filter.keys())[0]
            if filter_col not in df.columns:
                issues.append(f"Filter column '{filter_col}' not found")
    
    return issues


def is_metrics_comparison(df: pd.DataFrame, chart: ChartSpec) -> bool:
    metric_indicators = ['metric', 'rmse', 'mae', 'r2', 'auc', 'accuracy', 
                        'precision', 'recall', 'f1', 'mape', 'mse']
    
    if chart.color:
        color_values = df[chart.color].unique() if chart.color in df.columns else []
        for val in color_values:
            if any(indicator in str(val).lower() for indicator in metric_indicators):
                return True
    
    y_col_name = chart.y_axis.lower()
    if any(indicator in y_col_name for indicator in metric_indicators):
        return True
    
    return False


def render_chart(df: pd.DataFrame, spec: ChartSpec) -> Optional[go.Figure]:
    try:
        plot_df = df.copy()
        
        if spec.filter:
            for col, val in spec.filter.items():
                plot_df = plot_df[plot_df[col] == val]
        
        if len(plot_df) == 0:
            print(f"No data after filtering for {spec.title}")
            return None
        
        if spec.sort_by and spec.sort_by in plot_df.columns:
            plot_df = plot_df.sort_values(spec.sort_by, ascending=spec.sort_ascending)
        
        kwargs = {
            'data_frame': plot_df,
            'x': spec.x_axis,
            'y': spec.y_axis,
            'title': spec.title
        }
        
        if spec.color and spec.color in plot_df.columns:
            kwargs['color'] = spec.color
        
        if spec.facet and spec.facet in plot_df.columns:
            kwargs['facet_col'] = spec.facet
        
        if spec.chart_type == 'bar':
            if spec.orientation:
                kwargs['orientation'] = spec.orientation
            fig = px.bar(**kwargs)
            if spec.barmode:
                fig.update_layout(barmode=spec.barmode)
        
        elif spec.chart_type == 'line':
            kwargs['markers'] = True
            fig = px.line(**kwargs)
        
        elif spec.chart_type == 'scatter':
            fig = px.scatter(**kwargs)
        
        elif spec.chart_type == 'box':
            fig = px.box(**kwargs)
        
        elif spec.chart_type == 'histogram':
            fig = px.histogram(**kwargs)
        
        else:
            print(f"Unsupported chart type: {spec.chart_type}")
            return None
        
        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=12)
        )
        
        return fig
    
    except Exception as e:
        print(f"Error rendering chart '{spec.title}': {e}")
        import traceback
        traceback.print_exc()
        return None


def visualization_rendering_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_rendering')
    
    rendered_charts = state.get('rendered_charts', [])
    
    return {
        "rendered_charts": rendered_charts,
        "execution_path": execution_path
    }


def insight_generation_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('insight_generation')
    
    analysis_results = state.get('analysis_results', {})
    context_docs = state.get('context_documents', [])
    comparison_type = state.get('comparison_type')
    rendered_charts = state.get('rendered_charts', [])
    viz_strategy = state.get('viz_strategy')
    viz_reasoning = state.get('viz_reasoning')
    viz_warnings = state.get('viz_warnings', [])
    
    viz_context = ""
    if rendered_charts:
        viz_descriptions = []
        for i, chart in enumerate(rendered_charts, 1):
            viz_descriptions.append(
                f"**Chart {i}: {chart['title']}**\n"
                f"Type: {chart['type']}\n"
                f"Shows: {chart.get('explanation', 'Data visualization')}"
            )
        
        viz_context = f"""

📊 **Visualizations Generated ({viz_strategy}):**
{viz_reasoning}

{chr(10).join(viz_descriptions)}

Reference these charts naturally in your explanation (e.g., "As shown in Chart 1...", "The visualization illustrates...").
"""
    
    if viz_warnings:
        viz_context += f"\n\n⚠️ **Visualization Notes:**\n" + "\n".join(f"- {w}" for w in viz_warnings)
    
    prompt = f"""Generate a clear, business-focused explanation of the analysis results.

Query: {state['user_query']}
Comparison Type: {comparison_type}

Analysis Results:
{json.dumps(analysis_results, indent=2, default=str)}

Context:
{json.dumps(context_docs[:3], indent=2, default=str)}
{viz_context}

**Instructions:**
1. Direct answer to the user's question
2. Key quantitative findings (cite specific numbers)
3. Reference the charts naturally (e.g., "Chart 1 shows that...", "As visualized above...")
4. Explain WHY these results occurred (contextual reasoning)
5. Business implications for pharma commercial analytics
6. Actionable recommendations if applicable

Keep language clear and avoid jargon. Structure your response to flow with the visualizations.
"""
    
    response = llm.invoke(prompt)
    extracted_models = extract_model_names_from_text(response.content)
    return {
        "messages": [AIMessage(content=response.content)],
        "final_insights": response.content,
        "rendered_charts": rendered_charts,
        "execution_path": execution_path,
        "mentioned_models": extracted_models if extracted_models else None
    }


def orchestrator_agent(state: AgentState) -> dict:
    needs_clarification = state.get('needs_clarification', False)
    loop_count = state.get('loop_count', 0)
    clarification_attempts = state.get('clarification_attempts', 0)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('orchestrator')

    if clarification_attempts >= 3:
        print(f"DEBUG: Max clarification attempts reached ({clarification_attempts}), proceeding anyway")
        needs_clarification = False
    
    if loop_count > 16:
        return {
            "next_action": "end",
            "execution_path": execution_path,
            "messages": [AIMessage(content="Maximum conversation depth reached.")]
        }
    
    if needs_clarification:
        clarification = state.get('clarification_question', "Could you please provide more details?")
        return {
            "next_action": "ask_clarification",
            "execution_path": execution_path,
            "clarification_attempts": clarification_attempts + 1,
            "messages": [AIMessage(content=clarification)]
        }
    
    use_case = state.get('use_case')
    models_requested = state.get('models_requested')
    
    if not use_case and not models_requested:
        return {
            "next_action": "ask_clarification",
            "execution_path": execution_path,
            "needs_clarification": True,
            "clarification_question": "I need more information. Which use case or models are you interested in?",
            "messages": [AIMessage(content="I need more information. Which use case or models are you interested in?")]
        }
    
    return {
        "next_action": "retrieve_data",
        "execution_path": execution_path,
        "loop_count": loop_count + 1
    }