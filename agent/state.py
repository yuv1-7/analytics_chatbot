from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Messages
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
    requires_visualization: bool
    
    # Context retrieval outputs
    context_documents: Optional[List[Dict[str, Any]]]
    
    # SQL Generation outputs
    generated_sql: Optional[str]
    sql_purpose: Optional[str]
    expected_columns: Optional[List[str]]
    
    # Data retrieval outputs
    retrieved_data: Optional[Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]]
    
    # Analysis outputs
    analysis_results: Optional[Dict[str, Any]]
    
    # Visualization outputs
    visualization_specs: Optional[List[Dict[str, Any]]]
    rendered_charts: Optional[List[Dict[str, Any]]]
    
    # Insight generation outputs
    final_insights: Optional[str]
    
    # Control flow
    needs_clarification: bool
    clarification_question: Optional[str]
    loop_count: int
    next_action: Optional[str]
    execution_path: List[str]