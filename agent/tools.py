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

ALL_TOOLS = [
    execute_sql_query
]