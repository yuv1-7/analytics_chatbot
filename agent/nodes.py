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
3. If attempts >= 1, DO NOT ask for clarification. Make reasonable assumptions.
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
    
    # Include recent conversation for better context
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

def analyze_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    structure = {
        'row_count': len(df),
        'columns': list(df.columns),
        'column_types': {},
        'cardinality': {},
        'has_temporal': False,
        'temporal_columns': [],
        'numeric_columns': [],
        'categorical_columns': []
    }
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            structure['column_types'][col] = 'numeric'
            structure['numeric_columns'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            structure['column_types'][col] = 'temporal'
            structure['temporal_columns'].append(col)
            structure['has_temporal'] = True
        else:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > 0.8 * len(df):
                    structure['column_types'][col] = 'temporal'
                    structure['temporal_columns'].append(col)
                    structure['has_temporal'] = True
                    continue
            except:
                pass
            
            structure['column_types'][col] = 'categorical'
            structure['categorical_columns'].append(col)
        
        structure['cardinality'][col] = df[col].nunique()
    
    return structure


def should_visualize(analysis_results: Dict[str, Any], query_context: str) -> bool:
    raw_data = analysis_results.get('raw_data', [])
    
    if not raw_data:
        return False
    
    for result in raw_data:
        if result.get('success') and result.get('data'):
            data = result['data']
            if len(data) > 1:
                return True
    
    computed_metrics = analysis_results.get('computed_metrics', {})
    if computed_metrics and len(computed_metrics) > 1:
        return True
    
    viz_keywords = ['compare', 'trend', 'performance', 'over time', 'distribution', 'ranking', 'top', 'vs']
    query_lower = query_context.lower()
    return any(keyword in query_lower for keyword in viz_keywords)


def select_chart_type(structure: Dict[str, Any], query_context: str) -> Dict[str, Any]:
    row_count = structure['row_count']
    temporal_cols = structure['temporal_columns']
    numeric_cols = structure['numeric_columns']
    categorical_cols = structure['categorical_columns']
    cardinality = structure['cardinality']
    
    if row_count > 100:
        high_card_cats = [col for col in categorical_cols if cardinality.get(col, 0) > 50]
        if high_card_cats:
            return {
                'chart_type': None,
                'warning': f'Too many unique values in {high_card_cats[0]} ({cardinality[high_card_cats[0]]} values)',
                'suggestion': 'Consider filtering or aggregating data'
            }
    
    if temporal_cols and numeric_cols:
        return {
            'chart_type': 'line_chart',
            'primary_role': 'temporal',
            'suitable_for': 'trends over time'
        }
    
    low_card_cats = [col for col in categorical_cols if cardinality.get(col, 0) <= 20]
    if low_card_cats and numeric_cols:
        return {
            'chart_type': 'bar_chart',
            'primary_role': 'comparison',
            'suitable_for': 'comparing categories'
        }
    
    if 'rank' in [c.lower() for c in structure['columns']] or \
       any('importance' in c.lower() for c in structure['columns']):
        return {
            'chart_type': 'bar_chart',
            'primary_role': 'ranking',
            'suitable_for': 'showing rankings'
        }
    
    if len(numeric_cols) >= 2:
        return {
            'chart_type': 'scatter_plot',
            'primary_role': 'correlation',
            'suitable_for': 'relationships between variables'
        }
    
    if len(numeric_cols) == 1 and row_count <= 5:
        return {
            'chart_type': 'metric_card',
            'primary_role': 'summary',
            'suitable_for': 'key metrics'
        }
    
    if categorical_cols and numeric_cols:
        return {
            'chart_type': 'bar_chart',
            'primary_role': 'general',
            'suitable_for': 'general comparison'
        }
    
    return {
        'chart_type': None,
        'warning': 'No suitable visualization pattern detected',
        'suggestion': 'Data may be better viewed as a table'
    }


class VizColumnMapping(BaseModel):
    x: str = Field(description="Column name for x-axis")
    y: str = Field(description="Column name for y-axis")
    color: Optional[str] = Field(default=None, description="Column name for color grouping")
    facet: Optional[str] = Field(default=None, description="Column name for faceting")


def map_columns_to_chart(
    structure: Dict[str, Any],
    chart_selection: Dict[str, Any],
    query_context: str
) -> Optional[Dict[str, Any]]:
    chart_type = chart_selection.get('chart_type')
    if not chart_type or chart_type == 'metric_card':
        return None
    
    available_columns = structure['columns']
    column_types = structure['column_types']
    
    column_info = []
    for col in available_columns:
        col_type = column_types[col]
        cardinality = structure['cardinality'][col]
        column_info.append(f"- {col} ({col_type}, {cardinality} unique values)")
    
    prompt = f"""Map semantic roles to EXACT column names for a {chart_type}.

AVAILABLE COLUMNS:
{chr(10).join(column_info)}

USER QUERY CONTEXT: {query_context}

RULES:
1. Use ONLY column names from the available columns list above
2. For {chart_type}:
   - x: {'temporal column' if chart_type == 'line_chart' else 'categorical column for grouping'}
   - y: numeric column to visualize
   - color: optional categorical column for grouping (only if relevant)
   - facet: optional categorical column for subplots (only if relevant)
3. Choose columns that best answer the user's query
4. Leave color/facet as null if not needed

Output valid JSON mapping only."""
    
    try:
        structured_llm = llm.with_structured_output(VizColumnMapping)
        mapping = structured_llm.invoke(prompt)
        
        for field in ['x', 'y', 'color', 'facet']:
            value = getattr(mapping, field)
            if value and value not in available_columns:
                return None
        
        return {
            'x': mapping.x,
            'y': mapping.y,
            'color': mapping.color,
            'facet': mapping.facet
        }
    
    except Exception as e:
        return None


def visualization_specification_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_spec')
    
    analysis_results = state.get('analysis_results', {})
    
    if not should_visualize(analysis_results, state.get('user_query', '')):
        return {"execution_path": execution_path, "visualization_specs": []}
    
    raw_data = analysis_results.get('raw_data', [])
    
    df = None
    for result in raw_data:
        if result.get('success') and result.get('data'):
            df = pd.DataFrame(result['data'])
            break
    
    if df is None or df.empty:
        return {"execution_path": execution_path, "visualization_specs": []}
    
    if len(df) > 100:
        df = df.head(100)
    
    structure = analyze_data_structure(df)
    
    chart_selection = select_chart_type(structure, state['user_query'])
    
    if not chart_selection.get('chart_type'):
        return {"execution_path": execution_path, "visualization_specs": []}
    
    column_mapping = map_columns_to_chart(structure, chart_selection, state['user_query'])
    
    if not column_mapping:
        x_col = (structure['temporal_columns'] or structure['categorical_columns'] or structure['columns'])[0]
        y_col = structure['numeric_columns'][0] if structure['numeric_columns'] else structure['columns'][1]
        column_mapping = {'x': x_col, 'y': y_col, 'color': None, 'facet': None}
    
    viz_spec = {
        'type': chart_selection['chart_type'],
        'title': f"{state['user_query'][:60]}...",
        'x': column_mapping['x'],
        'y': column_mapping['y'],
        'color': column_mapping.get('color'),
        'facet': column_mapping.get('facet'),
        'data': df.to_dict('records')
    }
    
    return {
        "visualization_specs": [viz_spec],
        "execution_path": execution_path,
        "requires_visualization": True
    }


def visualization_rendering_agent(state: AgentState) -> dict:
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_rendering')
    
    viz_specs = state.get('visualization_specs', [])
    
    if not viz_specs:
        return {"execution_path": execution_path}
    
    rendered_charts = []
    
    for spec in viz_specs:
        try:
            df = pd.DataFrame(spec['data'])
            
            if df.empty:
                continue
            
            chart_type = spec['type']
            x_col = spec['x']
            y_col = spec['y']
            color_col = spec.get('color')
            
            if x_col not in df.columns or y_col not in df.columns:
                continue
            
            fig = None
            
            if chart_type == 'bar_chart':
                if color_col and color_col in df.columns:
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=spec['title'])
                else:
                    fig = px.bar(df, x=x_col, y=y_col, title=spec['title'])
            
            elif chart_type == 'line_chart':
                if color_col and color_col in df.columns:
                    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=spec['title'])
                else:
                    fig = px.line(df, x=x_col, y=y_col, title=spec['title'])
            
            elif chart_type == 'scatter_plot':
                if color_col and color_col in df.columns:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=spec['title'])
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=spec['title'])
            
            else:
                continue
            
            if fig:
                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                rendered_charts.append({
                    'title': spec['title'],
                    'figure': fig,
                    'type': chart_type
                })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
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
    generated_sql = state.get('generated_sql')
    sql_purpose = state.get('sql_purpose')
    rendered_charts = state.get('rendered_charts', [])
    
    viz_context = ""
    if rendered_charts:
        viz_descriptions = []
        for chart in rendered_charts:
            viz_descriptions.append(f"- {chart['type'].replace('_', ' ').title()}: {chart['title']}")
        viz_context = f"\n\nVisualizations Generated:\n" + "\n".join(viz_descriptions)
    
    prompt = f"""Generate a clear, business-focused explanation of the analysis results.

Query: {state['user_query']}
Comparison Type: {comparison_type}
SQL Query Purpose: {sql_purpose}

Analysis Results:
{json.dumps(analysis_results, indent=2, default=str)}

Context:
{json.dumps(context_docs, indent=2)}
{viz_context}

Provide:
1. Direct answer to the user's question
2. Key quantitative findings (numbers, percentages)
3. Contextual reasoning (WHY these results occurred)
4. Business implications
5. Recommendations (if applicable)
6. Reference to visualizations if generated (e.g., "As shown in the bar chart above...")

Keep language simple and avoid technical jargon. Focus on actionable insights."""
    
    response = llm.invoke(prompt)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "final_insights": response.content,
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

def orchestrator_agent(state: AgentState) -> dict:
    needs_clarification = state.get('needs_clarification', False)
    loop_count = state.get('loop_count', 0)
    clarification_attempts = state.get('clarification_attempts', 0)  # Add this
    
    execution_path = state.get('execution_path', [])
    execution_path.append('orchestrator')
    
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
            "clarification_attempts": clarification_attempts + 1,  # Add this
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