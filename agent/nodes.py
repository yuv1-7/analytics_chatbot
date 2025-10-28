import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from agent.state import AgentState
from agent.tools import ALL_TOOLS
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("gemini_api_key"),
    temperature=0.7
)

# Bind tools to LLM for data retrieval
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
    
    # ============================================================================
    # VECTOR DB RETRIEVAL - COMMENTED OUT
    # ============================================================================
    # TODO: Implement ChromaDB retrieval
    # Example implementation:
    #
    # from langchain.vectorstores import Chroma
    # from langchain.embeddings import GoogleGenerativeAIEmbeddings
    #
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001",
    #     google_api_key=os.getenv("gemini_api_key")
    # )
    #
    # vectorstore = Chroma(
    #     persist_directory="./chroma_db",
    #     embedding_function=embeddings,
    #     collection_name="pharma_models_context"
    # )
    #
    # use_case = state.get('use_case')
    # models = state.get('models_requested', [])
    # comparison_type = state.get('comparison_type')
    #
    # # Build search query
    # search_parts = []
    # if use_case:
    #     search_parts.append(f"use case: {use_case}")
    # if models:
    #     search_parts.append(f"models: {', '.join(models)}")
    # if comparison_type:
    #     search_parts.append(f"comparison: {comparison_type}")
    #
    # search_query = " ".join(search_parts)
    #
    # # Retrieve relevant documents
    # docs = vectorstore.similarity_search(search_query, k=5)
    #
    # context_docs = []
    # for doc in docs:
    #     context_docs.append({
    #         'type': doc.metadata.get('type', 'general'),
    #         'content': doc.page_content,
    #         'metadata': doc.metadata
    #     })
    # ============================================================================
    
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


def data_retrieval_agent(state: AgentState) -> dict:
    """Agent that decides which tools to call and retrieves data"""
    
    execution_path = state.get('execution_path', [])
    execution_path.append('data_retrieval')
    
    # Build context for tool selection
    context = f"""Based on the parsed query intent, determine which tools to call to retrieve the necessary data.

Parsed Intent:
- Use Case: {state.get('use_case')}
- Models Requested: {state.get('models_requested')}
- Comparison Type: {state.get('comparison_type')}
- Metrics: {state.get('metrics_requested')}
- Time Range: {state.get('time_range')}

Available tools:
1. get_ensemble_vs_base_performance - Compare ensemble vs base models
2. get_model_performance_summary - Get performance metrics for models
3. get_drift_detection_summary - Check for model drift
4. compare_model_versions - Compare two versions of a model
5. get_feature_importance_analysis - Get feature importance rankings
6. get_prediction_analysis - Get prediction results
7. search_models - Search for models by name/description

Call the appropriate tool(s) to retrieve the data needed to answer the user's question."""

    messages = [
        SystemMessage(content=context),
        HumanMessage(content=state['user_query'])
    ]
    
    # Get tool calls from LLM
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "execution_path": execution_path,
        "next_action": "analyze" if state.get('requires_visualization') else "generate_insights"
    }


def analysis_computation_agent(state: AgentState) -> dict:
    """Perform calculations and analysis on retrieved data"""
    execution_path = state.get('execution_path', [])
    execution_path.append('analysis_computation')
    
    # Extract data from tool messages
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
    
    # Perform analysis based on comparison type
    analysis_results = {
        'raw_data': tool_results,
        'computed_metrics': {},
        'trends': [],
        'anomalies': []
    }
    
    comparison_type = state.get('comparison_type')
    
    # Example: Compute percentage improvements for ensemble comparisons
    if comparison_type == 'ensemble_vs_base':
        for result in tool_results:
            if 'comparison' in result:
                for metric, values in result['comparison'].items():
                    if isinstance(values, dict) and 'improvement_vs_average' in values:
                        analysis_results['computed_metrics'][metric] = {
                            'improvement_percentage': values['improvement_vs_average'],
                            'ensemble_value': values.get('ensemble_value'),
                            'base_average': values.get('base_average')
                        }
    
    return {
        "analysis_results": analysis_results,
        "execution_path": execution_path
    }


def visualization_specification_agent(state: AgentState) -> dict:
    """Determine which visualizations to generate"""
    execution_path = state.get('execution_path', [])
    execution_path.append('visualization_spec')
    
    # Only run if visualization is required
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
    
    # Convert data to DataFrame if needed
    if isinstance(data, dict):
        # Handle computed metrics
        if data_key == 'computed_metrics':
            df = pd.DataFrame([
                {'metric': k, **v} for k, v in data.items()
            ])
        else:
            df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return None
    
    chart_type = spec['type']
    title = spec['title']
    
    # Generate chart based on type
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


def insight_generation_agent(state: AgentState) -> dict:
    """Generate human-readable narrative insights"""
    execution_path = state.get('execution_path', [])
    execution_path.append('insight_generation')
    
    analysis_results = state.get('analysis_results', {})
    context_docs = state.get('context_documents', [])
    comparison_type = state.get('comparison_type')
    
    prompt = f"""Generate a clear, business-focused explanation of the analysis results.

Query: {state['user_query']}
Comparison Type: {comparison_type}

Analysis Results:
{json.dumps(analysis_results, indent=2)}

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