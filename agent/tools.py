from langchain_core.tools import tool
from typing import Optional, List
from core.sql_executor import get_sql_executor
import json


@tool
def execute_sql_query(sql_query: str) -> dict:
    """
    Execute a SQL query against the analytics database.
    
    This tool should be used AFTER the sql_generation_agent has created a valid SQL query.
    The SQL query must be a SELECT statement (read-only).
    
    Use this tool when you need to:
    - Retrieve model performance metrics
    - Compare ensemble vs base models
    - Get feature importance rankings
    - Check for model drift
    - Analyze predictions
    - Compare model versions
    - Search for models
    
    Args:
        sql_query: A valid SQL SELECT query string
    
    Returns:
        Dictionary with query results including:
        - success (bool): Whether query executed successfully
        - data (list): List of row dictionaries
        - row_count (int): Number of rows returned
        - columns (list): Column names
        - error (str): Error message if failed
    
    Example:
        sql_query = "SELECT model_name, algorithm FROM models WHERE is_active = true LIMIT 10"
    """
    executor = get_sql_executor()
    result = executor.execute_query(sql_query, max_rows=1000)
    
    # Convert result to JSON-serializable format
    if result['success']:
        # Convert any non-serializable types
        data = result['data']
        for row in data:
            for key, value in row.items():
                if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                    row[key] = str(value)
        
        return {
            'success': True,
            'data': data,
            'row_count': result['row_count'],
            'columns': result['columns'],
            'truncated': result.get('truncated', False),
            'message': f"Query executed successfully. Retrieved {result['row_count']} rows."
        }
    else:
        return {
            'success': False,
            'error': result['error'],
            'data': [],
            'row_count': 0,
            'columns': []
        }


@tool
def fetch_full_insight(turn_id: int) -> dict:
    """
    Retrieve the complete insight from a past conversation turn.
    
    Use this tool when the user asks about a specific past turn or needs
    detailed information from earlier in the conversation.
    
    When to use:
    - User asks "why" or "explain in detail" about a past turn
    - User references a specific turn number (e.g., "turn 3")
    - User says "tell me more" about past analysis
    
    When NOT to use:
    - User asks new questions (use database query instead)
    - Information needed is from current turn
    
    Args:
        turn_id: The turn number to retrieve (e.g., 3 for "turn 3")
    
    Returns:
        Dictionary with:
        - success (bool): Whether retrieval succeeded
        - turn (int): Turn number
        - user_query (str): Original query from that turn
        - full_insight (str): Complete insight text
        - timestamp (str): When the insight was generated
        - total_chunks (int): Number of chunks stored
        - error (str): Error message if failed
    
    Example:
        User: "Why did XGBoost perform better in turn 3?"
        → Call fetch_full_insight(3) to get complete details
    """
    try:
        from core.memory_manager import get_memory_manager
        
        # Get session_id from current state
        import streamlit as st
        session_id = st.session_state.get('session_id', 'default_session')
        
        memory_manager = get_memory_manager()
        result = memory_manager.get_full_turn(
            session_id=session_id,
            turn_number=turn_id
        )
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to fetch turn {turn_id}: {str(e)}"
        }


ALL_TOOLS = [
    execute_sql_query,
    fetch_full_insight
]