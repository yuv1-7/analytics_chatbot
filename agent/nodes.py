import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from agent.state import AgentState
from agent.tools import ALL_TOOLS
from core.schema_context import SCHEMA_CONTEXT, METRIC_GUIDE
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


class VisualizationSpec(BaseModel):
    visualizations: List[Dict[str, Any]] = Field(
        description="List of visualization specifications"
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
    
    context_summary = ""
    if mentioned_models:
        context_summary += f"\nPreviously mentioned models: {', '.join(mentioned_models)}"
    if current_topic:
        context_summary += f"\nCurrent topic: {current_topic}"
    
    system_msg = f"""You are a query parser for pharma commercial analytics. Parse user queries to extract:

USE CASES:
- NRx_forecasting: Predicting new prescriptions
- HCP_engagement: HCP response to marketing
- feature_importance_analysis: Understanding key drivers
- model_drift_detection: Detecting model performance changes over time
- messaging_optimization: Next-best-action for HCP targeting

MODELS:
- Base models: Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, Neural Network, Decision Tree
- Ensembles: stacking, boosting, bagging, meta-learner

COMPARISON TYPES:
- performance: Compare model metrics (RMSE, AUC, accuracy, etc.)
- predictions: Compare actual predictions
- feature_importance: Compare which features matter most
- drift: Detect changes over time
- ensemble_vs_base: Why ensemble performs better/worse than base models

METRICS:
- Regression: RMSE, MAE, R2, MAPE
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Pharma-specific: TRx (total prescriptions), NRx (new prescriptions)

VISUALIZATION TRIGGERS:
Set requires_visualization=True if query contains:
- "show", "plot", "chart", "graph", "visualize", "display"
- Comparison of metrics/models
- Trends over time
- Distribution analysis

CONVERSATION CONTEXT:{context_summary}

REFERENCE RESOLUTION:
If the query contains references like "these", "them", "those models", "it", "they":
1. Set references_previous_context=True
2. Look at conversation context to determine what they refer to
3. Set resolved_models with the actual model names from context
4. If context is insufficient, set needs_clarification=True

Extract all relevant information. If query is ambiguous or missing critical info, set needs_clarification=True.
Save info as you go. Don't ask for info already provided by the user.

Examples:
- "Compare Random Forest vs XGBoost for NRx forecasting last month" → use_case=NRx_forecasting, models_requested=['Random Forest', 'XGBoost'], time_range={{'period': 'last_month'}}, requires_visualization=True
- "Compare performance between these" → references_previous_context=True, resolved_models=[from context], comparison_type=performance
- "Why did the ensemble perform worse?" → comparison_type=ensemble_vs_base, needs_clarification=True if no models in context"""
    
    structured_llm = llm.with_structured_output(ParsedIntent)
    
    context_messages = [SystemMessage(content=system_msg)]
    if messages:
        context_messages.extend(messages[-10:])
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
        else:
            result.needs_clarification = True
            result.clarification_question = "I don't have context about which models you're referring to. Could you please specify the model names?"
    
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
        summary_parts.append(f"models: {', '.join(final_models)}")
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
        "last_query_summary": " | ".join(summary_parts) if summary_parts else None
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
    context_docs=state.get('context_documents',[])
    
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

{METRIC_GUIDE}

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
    
    context = f"""Execute the following SQL query that was generated to answer the user's question:

User Query: {state['user_query']}
Query Purpose: {state.get('sql_purpose', 'Data retrieval')}

Use the execute_sql_query tool to run this query. """
    
    messages = [
        SystemMessage(content=context),
        HumanMessage(content=generated_sql)
    ]
    
    response = llm_with_tools.invoke(messages)
    
    extracted_models = extract_model_names_from_text(str(response.content))
    
    return {
        "messages": [response],
        "execution_path": execution_path,
        "mentioned_models": extracted_models if extracted_models else None
    }

class AnalysisOutput(BaseModel):
            computed_metrics: Dict[str, Any] = Field(description="Key calculated metrics")
            patterns: List[str] = Field(description="Identified patterns and trends")
            anomalies: List[str] = Field(description="Unusual observations or outliers")
            statistical_summary: Dict[str, Any] = Field(description="Statistical summaries")

def analysis_computation_agent(state: AgentState) -> dict:
    """
    Performs intelligent analysis on retrieved data using LLM.
    Computes metrics, identifies patterns, and detects anomalies.
    """
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
    
    # If no data retrieved, return empty analysis
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
{json.dumps(successful_data[:50], indent=2, default=str)}  # Limit to 50 rows to avoid token limits

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
        
        # Fallback: Basic pandas-style analysis
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
    """
    Fallback basic metric computation when LLM fails.
    Extracts common pharma metrics from data.
    """
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
def visualization_specification_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_spec')
    
    if not state.get('requires_visualization'):
        return {"execution_path": execution_path}
    
    analysis_results = state.get('analysis_results', {})
    comparison_type = state.get('comparison_type')
    
    prompt = f"""
Based on the query and analysis results, generate visualization specifications.

Query: {state['user_query']}
Comparison Type: {comparison_type}
Analysis Results Available: {list(analysis_results.keys())}

Output a JSON with visualization specs. Each spec should include:
- type: bar_chart, line_chart, scatter_plot, heatmap, box_plot
- title: Descriptive title
- data_key: Which key from analysis_results to use
- x, y: Column names for axes
- additional_params: Any extra configuration

Example output:
{{
    "visualizations": [
        {{
            "type": "bar_chart",
            "title": "Ensemble vs Base Model Performance",
            "data_key": "computed_metrics",
            "x": "metric_name",
            "y": "improvement_percentage",
            "additional_params": {{"color": "metric_name"}}
        }}
    ]
}}
"""
    
    structured_llm = llm.with_structured_output(VisualizationSpec)
    
    try:
        result = structured_llm.invoke(prompt)
        # Convert Pydantic models to dictionaries
        if hasattr(result, 'visualizations'):
            viz_specs = [
                v.dict() if hasattr(v, 'dict') else v.model_dump() if hasattr(v, 'model_dump') else v
                for v in result.visualizations
            ]
        else:
            viz_specs = _get_default_viz_specs(comparison_type)
    except Exception as e:
        print(f"Visualization spec generation failed: {e}")
        viz_specs = _get_default_viz_specs(comparison_type)
    
    return {
        "visualization_specs": viz_specs,
        "execution_path": execution_path
    }


def _get_default_viz_specs(comparison_type: str) -> List[Dict[str, Any]]:
    defaults = {
        'ensemble_vs_base': [
            {
                "type": "bar_chart",
                "title": "Model Performance Comparison",
                "data_key": "computed_metrics",
                "x": "metric_name",
                "y": "improvement_percentage",
                "additional_params": {}
            }
        ],
        'performance': [
            {
                "type": "bar_chart",
                "title": "Performance Metrics",
                "data_key": "raw_data",
                "x": "model_name",
                "y": "metric_value",
                "additional_params": {}
            }
        ],
        'drift': [
            {
                "type": "line_chart",
                "title": "Drift Score Over Time",
                "data_key": "raw_data",
                "x": "timestamp",
                "y": "drift_score",
                "additional_params": {}
            }
        ]
    }
    
    return defaults.get(comparison_type, [])


def visualization_rendering_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_rendering')
    
    viz_specs = state.get('visualization_specs', [])
    analysis_results = state.get('analysis_results', {})
    
    if not viz_specs:
        print("No visualization specs found")
        return {"execution_path": execution_path}
    
    rendered_charts = []
    
    for spec in viz_specs:
        try:
            chart = _render_chart(spec, analysis_results)
            if chart:
                rendered_charts.append({
                    'title': spec.get('title', 'Chart'),
                    'figure': chart,
                    'type': spec.get('type', 'unknown')
                })
                print(f"Successfully rendered chart: {spec.get('title')}")
            else:
                print(f"Failed to render chart: {spec.get('title')}")
        except Exception as e:
            print(f"Failed to render chart '{spec.get('title')}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Total charts rendered: {len(rendered_charts)}")
    
    return {
        "rendered_charts": rendered_charts,
        "execution_path": execution_path
    }


def _render_chart(spec: Dict[str, Any], analysis_results: Dict[str, Any]) -> Any:
    import pandas as pd
    import plotly.express as px
    
    data_key = spec.get('data_key', 'raw_data')
    data = analysis_results.get(data_key)
    
    print(f"Rendering chart: {spec.get('title')}")
    print(f"Data key: {data_key}")
    print(f"Data type: {type(data)}")
    
    if not data:
        print(f"No data found for key: {data_key}")
        return None
    
    # Convert data to DataFrame
    df = None
    
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        if data_key == 'computed_metrics':
            # Handle computed metrics format
            rows = []
            for metric_name, metric_values in data.items():
                row = {'metric_name': metric_name}
                row.update(metric_values)
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            # Try to convert dict to DataFrame
            try:
                df = pd.DataFrame([data])
            except:
                df = pd.DataFrame(data)
    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            if 'data' in data[0]:
                df = pd.DataFrame(data[0]['data'])
            else:
                df = pd.DataFrame(data)
        else:
            try:
                df = pd.DataFrame(data)
            except:
                print(f"Could not convert list to DataFrame")
                return None
    else:
        print(f"Unsupported data type: {type(data)}")
        return None
    
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return None
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    chart_type = spec.get('type')
    title = spec.get('title', 'Chart')
    x_col = spec.get('x')
    y_col = spec.get('y')
    
    # Verify columns exist
    if x_col and x_col not in df.columns:
        print(f"Warning: x column '{x_col}' not found in DataFrame. Available: {df.columns.tolist()}")
        # Try to find a suitable column
        x_col = df.columns[0] if len(df.columns) > 0 else None
    
    if y_col and y_col not in df.columns:
        print(f"Warning: y column '{y_col}' not found in DataFrame. Available: {df.columns.tolist()}")
        # Try to find a suitable column
        y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    try:
        fig = None
        
        if chart_type == 'bar_chart':
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'line_chart':
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'scatter_plot':
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'heatmap':
            if df.shape[0] > 1 and df.select_dtypes(include=[np.number]).shape[1] > 1:
                fig = px.imshow(df.select_dtypes(include=[np.number]).corr(), title=title)
            else:
                print("Not enough numeric data for heatmap")
                return None
        elif chart_type == 'box_plot':
            fig = px.box(df, y=y_col, title=title)
        else:
            print(f"Unknown chart type: {chart_type}")
            return None
        
        if fig:
            # Update layout for better display
            fig.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
        
        return fig
        
    except Exception as e:
        print(f"Chart rendering error: {e}")
        import traceback
        traceback.print_exc()
        return None


def insight_generation_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('insight_generation')
    
    analysis_results = state.get('analysis_results', {})
    context_docs = state.get('context_documents', [])
    comparison_type = state.get('comparison_type')
    generated_sql = state.get('generated_sql')
    sql_purpose = state.get('sql_purpose')
    
    prompt = f"""Generate a clear, business-focused explanation of the analysis results.

Query: {state['user_query']}
Comparison Type: {comparison_type}
SQL Query Purpose: {sql_purpose}

Analysis Results:
{json.dumps(analysis_results, indent=2, default=str)}

Context:
{json.dumps(context_docs, indent=2)}

Provide:
1. Direct answer to the user's question
2. Key quantitative findings (numbers, percentages)
3. Contextual reasoning (WHY these results occurred)
4. Business implications
5. Recommendations (if applicable)

Keep language simple and avoid technical jargon. Focus on actionable insights."""
    
    response = llm.invoke(prompt)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "final_insights": response.content,
        "execution_path": execution_path
    }


def orchestrator_agent(state: AgentState) -> dict:
    needs_clarification = state.get('needs_clarification', False)
    loop_count = state.get('loop_count', 0)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('orchestrator')
    
    if loop_count > 16:
        return {
            "next_action": "end",
            "execution_path": execution_path,
            "messages": [AIMessage(content="Maximum conversation depth reached. Please start a new query.")]
        }
    
    if needs_clarification:
        clarification = state.get('clarification_question', "Could you please provide more details about your query?")
        return {
            "next_action": "ask_clarification",
            "execution_path": execution_path,
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