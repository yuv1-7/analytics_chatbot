from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

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

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    
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
    
    retrieved_data: Optional[Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]]
    
    analysis_results: Optional[Dict[str, Any]]
    
    visualization_specs: Optional[List[Dict[str, Any]]]
    rendered_charts: Optional[List[Dict[str, Any]]]
    
    final_insights: Optional[str]
    
    needs_clarification: bool
    clarification_question: Optional[str]
    loop_count: int
    next_action: Optional[str]
    execution_path: List[str]
    
    conversation_context: Optional[Dict[str, Any]]
    mentioned_models: Annotated[Optional[List[str]], merge_lists]
    mentioned_model_ids: Annotated[Optional[List[str]], merge_lists]
    last_query_summary: Optional[str]
    current_topic: Optional[str]