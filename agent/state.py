from typing import TypedDict, Optional, List, Dict, Any, Literal
from datetime import datetime
from sqlalchemy.orm import Session

class AgentState(TypedDict):
    
    # USER INPUT
    user_query: str
    session_id: str
    timestamp: datetime
    
    # DATABASE SESSION
    db_session: Optional[Session]
    
    # QUERY UNDERSTANDING
    intent: Optional[Literal[
        "SINGLE_MODEL_PERFORMANCE",
        "MODEL_COMPARISON", 
        "ENSEMBLE_ANALYSIS",
        "FEATURE_IMPORTANCE",
        "DRIFT_DETECTION",
        "PREDICTION_ANALYSIS"
    ]]
    entities: Dict[str, Any]
    needs_clarification: bool
    
    # DATA RETRIEVAL (ORM-based)
    orm_queries: List[Dict[str, Any]]
    orm_objects: Dict[str, List[Any]]
    data_retrieval_success: bool
    
    # ANALYSIS
    analysis: Dict[str, Any]
    ensemble_analysis: Optional[Dict[str, Any]]
    
    # INSIGHTS
    insights: str
    recommendations: List[str]
    
    # RESPONSE
    formatted_response: str
    
    # CONTROL
    errors: List[str]
    routing_history: List[str]