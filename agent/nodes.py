import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from agent.state import AgentState
from agent.tools import ALL_TOOLS
from core.schema_context import SCHEMA_CONTEXT, METRIC_GUIDE
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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


def query_understanding_agent(state: AgentState) -> dict:
    """Parse user query to extract structured intent"""
    query = state['user_query']
    messages = state.get('messages', [])
    
    system_msg = """You are a query parser for pharma commercial analytics. Parse user queries to extract:

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

Extract all relevant information. If query is ambiguous or missing critical info, set needs_clarification=True and provide a clarification_question.
Save info as you go. Don't ask for info already provided by the user.

Examples:
- "Compare Random Forest vs XGBoost for NRx forecasting last month" → use_case=NRx_forecasting, models_requested=['Random Forest', 'XGBoost'], time_range={'period': 'last_month'}, requires_visualization=True
- "Why did the ensemble perform worse?" → comparison_type=ensemble_vs_base, needs_clarification=True (which use case? which ensemble?)
- "Show me drift detection results" → use_case=model_drift_detection, requires_visualization=True"""
    
    structured_llm = llm.with_structured_output(ParsedIntent)
    
    context_messages = [SystemMessage(content=system_msg)]
    if messages:
        context_messages.extend(messages[-5:])
    context_messages.append(HumanMessage(content=query))
    
    result = structured_llm.invoke(context_messages)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('query_understanding')
    
    return {
        "messages": [HumanMessage(content=query)],
        "parsed_intent": result.dict(),
        "use_case": result.use_case,
        "models_requested": result.models_requested,
        "comparison_type": result.comparison_type,
        "time_range": result.time_range,
        "metrics_requested": result.metrics_requested,
        "entities_requested": result.entities_requested,
        "needs_clarification": result.needs_clarification,
        "clarification_question": result.clarification_question,
        "requires_visualization": result.requires_visualization,
        "execution_path": execution_path
    }


def context_retrieval_agent(state: AgentState) -> dict:
    """Retrieve relevant context from vector DB (COMMENTED OUT - TO BE IMPLEMENTED)"""
    execution_path = state.get('execution_path', [])
    execution_path.append('context_retrieval')
    
    # Placeholder: Return empty context for now
    context_docs = []
    
    # Optional: Add basic context based on query parameters (without vector DB)
    use_case = state.get('use_case')
    models = state.get('models_requested', [])
    
    if use_case:
        context_docs.append({
            'type': 'use_case_context',
            'content': f'Context for {use_case} (Vector DB retrieval disabled)',
            'source': 'placeholder'
        })
    
    if models:
        context_docs.append({
            'type': 'model_context',
            'content': f'Information about {", ".join(models)} (Vector DB retrieval disabled)',
            'source': 'placeholder'
        })
    
    return {
        "context_documents": context_docs,
        "execution_path": execution_path
    }


def sql_generation_agent(state: AgentState) -> dict:
    """Generate SQL queries based on parsed intent"""
    
    execution_path = state.get('execution_path', [])
    execution_path.append('sql_generation')
    
    use_case = state.get('use_case')
    models_requested = state.get('models_requested', [])
    comparison_type = state.get('comparison_type')
    metrics_requested = state.get('metrics_requested', [])
    time_range = state.get('time_range')
    
    prompt = f"""You are a SQL expert for a pharma commercial analytics database. Generate a SQL query to answer the user's question.

USER QUERY: {state['user_query']}

PARSED INTENT:
- Use Case: {use_case}
- Models Requested: {models_requested}
- Comparison Type: {comparison_type}
- Metrics: {metrics_requested}
- Time Range: {time_range}

{SCHEMA_CONTEXT}

{METRIC_GUIDE}

INSTRUCTIONS:
1. Generate a VALID PostgreSQL SELECT query
2. Use appropriate JOINs to connect tables
3. Filter for is_active = true when querying models
4. Use data_split = 'test' for performance metrics unless specified
5. Use latest_model_executions view for most recent execution
6. Handle NULL values appropriately
7. Use ILIKE for case-insensitive string matching
8. Order results meaningfully
9. Limit results to reasonable numbers (e.g., LIMIT 100)

COMMON PATTERNS:

For ENSEMBLE VS BASE COMPARISON:
```sql
-- Use the pre-built view
SELECT * FROM ensemble_vs_base_performance 
WHERE model_name ILIKE '%ensemble_name%';

-- Or build custom comparison
SELECT 
    m.model_name,
    m.model_type,
    pm.metric_name,
    pm.metric_value
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN performance_metrics pm ON lme.execution_id = pm.execution_id
WHERE m.model_name IN ('ensemble_name', 'base_model_1', 'base_model_2')
  AND pm.data_split = 'test'
  AND pm.metric_name IN ('rmse', 'r2_score')
ORDER BY m.model_type, pm.metric_name;
```

For FEATURE IMPORTANCE:
```sql
SELECT 
    fi.feature_name,
    fi.importance_score,
    fi.importance_type,
    fi.rank
FROM models m
JOIN latest_model_executions lme ON m.model_id = lme.model_id
JOIN feature_importance fi ON lme.execution_id = fi.execution_id
WHERE m.model_name ILIKE '%model_name%'
ORDER BY fi.rank
LIMIT 20;
```

For DRIFT DETECTION:
```sql
SELECT 
    m.model_name,
    me.drift_detected,
    me.drift_score,
    me.execution_timestamp
FROM models m
JOIN model_executions me ON m.model_id = me.model_id
WHERE m.model_name ILIKE '%model_name%'
  AND me.execution_status = 'success'
ORDER BY me.execution_timestamp DESC
LIMIT 10;
```

For MODEL SEARCH:
```sql
SELECT 
    model_id,
    model_name,
    model_type,
    algorithm,
    use_case,
    version,
    description
FROM models
WHERE is_active = true
  AND (model_name ILIKE '%search_term%' 
       OR algorithm ILIKE '%search_term%'
       OR description ILIKE '%search_term%')
ORDER BY created_at DESC
LIMIT 50;
```

For VERSION COMPARISON:
```sql
SELECT 
    vc.*,
    m.model_name
FROM version_comparisons vc
JOIN models m ON vc.model_id = m.model_id
WHERE m.model_name ILIKE '%model_name%'
  AND vc.old_version = 'v1.0'
  AND vc.new_version = 'v2.0';
```

Generate the SQL query now. Make sure it's a valid SELECT statement."""

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
    """Agent that calls the SQL execution tool with the generated query"""
    
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
    
    return {
        "messages": [response],
        "execution_path": execution_path
    }


def analysis_computation_agent(state: AgentState) -> dict:
    """Perform calculations and analysis on retrieved data"""
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
    
    analysis_results = {
        'raw_data': tool_results,
        'computed_metrics': {},
        'trends': [],
        'anomalies': []
    }
    
    comparison_type = state.get('comparison_type')
    
    if comparison_type == 'ensemble_vs_base' and tool_results:
        for result in tool_results:
            if result.get('success') and result.get('data'):
                data = result['data']
                for row in data:
                    if 'ensemble_advantage' in row:
                        metric_name = row.get('model_name', 'ensemble')
                        analysis_results['computed_metrics'][metric_name] = {
                            'ensemble_advantage': row.get('ensemble_advantage'),
                            'ensemble_rmse': row.get('ensemble_rmse'),
                            'avg_base_rmse': row.get('avg_base_rmse'),
                            'ensemble_r2': row.get('ensemble_r2'),
                            'avg_base_r2': row.get('avg_base_r2')
                        }
    
    return {
        "analysis_results": analysis_results,
        "execution_path": execution_path
    }


def visualization_specification_agent(state: AgentState) -> dict:
    """Determine which visualizations to generate"""
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
        viz_specs = result.visualizations
    except Exception as e:
        print(f"Visualization spec generation failed: {e}")
        # Fallback: Generate default visualization based on comparison type
        viz_specs = _get_default_viz_specs(comparison_type)
    
    return {
        "visualization_specs": viz_specs,
        "execution_path": execution_path
    }


def _get_default_viz_specs(comparison_type: str) -> List[Dict[str, Any]]:
    """Fallback visualization specs based on comparison type"""
    defaults = {
        'ensemble_vs_base': [
            {
                "type": "bar_chart",
                "title": "Model Performance Comparison",
                "data_key": "computed_metrics",
                "x": "metric_name",
                "y": "improvement_percentage"
            }
        ],
        'performance': [
            {
                "type": "bar_chart",
                "title": "Performance Metrics",
                "data_key": "raw_data",
                "x": "model_name",
                "y": "metric_value"
            }
        ],
        'drift': [
            {
                "type": "line_chart",
                "title": "Drift Score Over Time",
                "data_key": "raw_data",
                "x": "timestamp",
                "y": "drift_score"
            }
        ]
    }
    
    return defaults.get(comparison_type, [])


def visualization_rendering_agent(state: AgentState) -> dict:
    """Generate actual chart objects from specifications"""
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_rendering')
    
    viz_specs = state.get('visualization_specs', [])
    analysis_results = state.get('analysis_results', {})
    
    if not viz_specs:
        return {"execution_path": execution_path}
    
    rendered_charts = []
    
    for spec in viz_specs:
        try:
            chart = _render_chart(spec, analysis_results)
            if chart:
                rendered_charts.append({
                    'title': spec['title'],
                    'figure': chart,
                    'type': spec['type']
                })
        except Exception as e:
            print(f"Failed to render chart: {e}")
            continue
    
    return {
        "rendered_charts": rendered_charts,
        "execution_path": execution_path
    }


def _render_chart(spec: Dict[str, Any], analysis_results: Dict[str, Any]) -> Any:
    """Helper function to render individual charts"""
    data_key = spec.get('data_key', 'raw_data')
    data = analysis_results.get(data_key)
    
    if not data:
        return None
    
    if isinstance(data, dict):
        if data_key == 'computed_metrics':
            df = pd.DataFrame([
                {'metric': k, **v} for k, v in data.items()
            ])
        else:
            df = pd.DataFrame([data])
    elif isinstance(data, list):
        if data and isinstance(data[0], dict) and 'data' in data[0]:
            df = pd.DataFrame(data[0]['data'])
        else:
            df = pd.DataFrame(data)
    else:
        return None
    
    if df.empty:
        return None
    
    chart_type = spec['type']
    title = spec['title']
    try:
        if chart_type == 'bar_chart':
            fig = px.bar(df, x=spec.get('x'), y=spec.get('y'), title=title)
        elif chart_type == 'line_chart':
            fig = px.line(df, x=spec.get('x'), y=spec.get('y'), title=title)
        elif chart_type == 'scatter_plot':
            fig = px.scatter(df, x=spec.get('x'), y=spec.get('y'), title=title)
        elif chart_type == 'heatmap':
            fig = px.imshow(df.corr() if df.shape[0] > 1 else df, title=title)
        elif chart_type == 'box_plot':
            fig = px.box(df, y=spec.get('y'), title=title)
        else:
            return None
        
        return fig
    except Exception as e:
        print(f"Chart rendering error: {e}")
        return None


def insight_generation_agent(state: AgentState) -> dict:
    """Generate human-readable narrative insights"""
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
    """Control flow orchestrator"""
    needs_clarification = state.get('needs_clarification', False)
    loop_count = state.get('loop_count', 0)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('orchestrator')
    
    if loop_count > 5:
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