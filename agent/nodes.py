import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from agent.state import AgentState
from agent.tools import ALL_TOOLS

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

def query_understanding_agent(state: AgentState) -> dict:
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

Extract all relevant information. If query is ambiguous or missing critical info, set needs_clarification=True and provide a clarification_question.
Save info as you go. Dont ask for info already provided by the user.

Examples:
- "Compare Random Forest vs XGBoost for NRx forecasting last month" → use_case=NRx_forecasting, models_requested=['Random Forest', 'XGBoost'], time_range={'period': 'last_month'}
- "Why did the ensemble perform worse?" → comparison_type=ensemble_vs_base, needs_clarification=True (which use case? which ensemble?)
- "Show me drift detection results" → use_case=model_drift_detection"""
    
    structured_llm = llm.with_structured_output(ParsedIntent)
    
    context_messages = [SystemMessage(content=system_msg)]
    if messages:
        context_messages.extend(messages[-5:])
    context_messages.append(HumanMessage(content=query))
    
    result = structured_llm.invoke(context_messages)
    
    execution_path = state.get('execution_path', [])
    execution_path.append('query_understanding')
    print(result)
    
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
        "next_action": "generate_response"
    }

def orchestrator_agent(state: AgentState) -> dict:
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