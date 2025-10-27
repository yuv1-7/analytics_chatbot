from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    
    # Query Understanding outputs
    parsed_intent: Optional[Dict[str, Any]]
    use_case: Optional[str]
    models_requested: Optional[List[str]]
    comparison_type: Optional[str]
    time_range: Optional[Dict[str, Any]]
    metrics_requested: Optional[List[str]]
    entities_requested: Optional[List[str]]
    
    # Data retrieval outputs
    retrieved_data: Optional[Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]]
    
    # Control flow
    needs_clarification: bool
    clarification_question: Optional[str]
    loop_count: int
    next_action: Optional[str]
    execution_path: List[str]