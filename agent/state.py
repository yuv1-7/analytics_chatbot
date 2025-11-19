from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from operator import add 

def merge_lists(left: Optional[List], right: Optional[List]) -> Optional[List]:
    if right is None:
        return left
    if left is None:
        return right
    seen = set(left)
    result = list(left)
    for item in right:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def max_value(left: Optional[int], right: Optional[int]) -> Optional[int]:
    """Keep the maximum value for counter fields"""
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)

# In agent/state.py

def safe_add_lists(left: Optional[List], right: Optional[List]) -> List:
    """Safely add two lists, treating None as empty list"""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    simplified_query: Optional[str]
    
    parsed_intent: Optional[Dict[str, Any]]
    use_case: Optional[str]
    models_requested: Optional[List[str]]
    comparison_type: Optional[str]
    time_range: Optional[Dict[str, Any]]
    metrics_requested: Optional[List[str]]
    entities_requested: Optional[List[str]]
    requires_visualization: bool
    
    context_documents: Optional[List[Dict[str, Any]]]
    
    generated_sql: Optional[str]
    sql_purpose: Optional[str]
    expected_columns: Optional[List[str]]
    
    sql_retry_count: Annotated[int, max_value]  
    needs_sql_retry: bool  
    sql_error_feedback: Optional[str]
    
    retrieved_data: Optional[Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]]
    
    analysis_results: Optional[Dict[str, Any]]
    
    visualization_specs: Optional[List[Dict[str, Any]]]
    rendered_charts: Annotated[Optional[List[Dict[str, Any]]], safe_add_lists]  # ← CHANGED
    viz_strategy: Optional[str]
    viz_reasoning: Optional[str]
    viz_warnings: Optional[List[str]]
    
    final_insights: Optional[str]
    
    needs_clarification: bool
    clarification_question: Optional[str]
    loop_count: Annotated[int, max_value] 
    next_action: Optional[str]
    execution_path: Annotated[List[str], safe_add_lists]  # ← CHANGED
    
    conversation_context: Optional[Dict[str, Any]]
    mentioned_models: Annotated[Optional[List[str]], merge_lists]
    mentioned_model_ids: Annotated[Optional[List[str]], merge_lists]
    last_query_summary: Optional[str]
    current_topic: Optional[str]
    clarification_attempts: Annotated[int, max_value] 
    
    personalized_business_context: Optional[str]
    user_id: Optional[str]
    
    session_id: str
    turn_number: Annotated[int, max_value] 
    needs_memory: bool
    needs_database: bool
    conversation_summaries: Optional[List[Dict]]
    summary_generated: bool
