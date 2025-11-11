"""
Safe SQL execution module with validation and result formatting.
"""
import re
from typing import Dict, Any, List, Optional
from core.database import get_db_connection
import psycopg2.extras


class SQLExecutionError(Exception):
    """Custom exception for SQL execution errors"""
    pass

class SQLValidator:
    """Validates SQL queries for safety"""
    
    FORBIDDEN_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
        'TRUNCATE', 'GRANT', 'REVOKE', 'EXECUTE', 'EXEC',
        'MERGE', 'REPLACE', 'CALL'
    ]
    
    @classmethod
    def is_read_only(cls, sql: str) -> bool:
        """Check if SQL is read-only (SELECT/WITH only)"""
        # Remove comments
        sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        
        sql_upper = sql_clean.upper()
        for keyword in cls.FORBIDDEN_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return False
        
        if not (re.search(r'\bSELECT\b', sql_upper) or re.search(r'\bWITH\b', sql_upper)):
            return False
        
        return True
    
    @classmethod
    def validate_query(cls, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety.
        Returns (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty query"
        
        if not cls.is_read_only(sql):
            return False, "Query contains forbidden operations. Only SELECT queries are allowed."
        
        if sql.count(';') > 1:
            return False, "Multiple statements not allowed"
         
        return True, None


class SQLExecutor:
    """Executes SQL queries safely and formats results"""
    
    def __init__(self, query_timeout: int = 30):
        """
        Initialize SQL executor.
        
        Args:
            query_timeout: Query timeout in seconds (default 30)
        """
        self.query_timeout = query_timeout
        self.validator = SQLValidator()
    
    def execute_query(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute a SQL query and return formatted results.
        
        Args:
            sql: SQL query string
            params: Optional query parameters
            max_rows: Maximum rows to return (default 1000)
        
        Returns:
            Dict with 'success', 'data', 'row_count', 'columns', 'error'
        """
        is_valid, error_msg = self.validator.validate_query(sql)
        if not is_valid:
            return {
                'success': False,
                'error': f"Query validation failed: {error_msg}",
                'data': None,
                'row_count': 0,
                'columns': []
            }
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = {self.query_timeout * 1000}")
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if params:
                        cur.execute(sql, params)
                    else:
                        cur.execute(sql)
                    
                    rows = cur.fetchmany(max_rows)
                    
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    
                    data = [dict(row) for row in rows]
                    
                    total_rows = cur.rowcount if cur.rowcount != -1 else len(data)
                    
                    return {
                        'success': True,
                        'data': data,
                        'row_count': len(data),
                        'total_rows': total_rows,
                        'columns': columns,
                        'error': None,
                        'truncated': len(data) >= max_rows
                    }
        
        except psycopg2.Error as e:
            return {
                'success': False,
                'error': f"Database error: {str(e)}",
                'data': None,
                'row_count': 0,
                'columns': [],
                'sql_error_code': e.pgcode if hasattr(e, 'pgcode') else None
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {str(e)}",
                'data': None,
                'row_count': 0,
                'columns': []
            }
    
    def execute_multiple_queries(
        self, 
        queries: List[str],
        max_rows: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple queries sequentially.
        
        Args:
            queries: List of SQL query strings
            max_rows: Maximum rows per query
        
        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.execute_query(query, max_rows=max_rows)
            results.append(result)
            
            if not result['success']:
                break
        
        return results
    
    def format_result_for_llm(self, result: Dict[str, Any]) -> str:
        """
        Format query result for LLM consumption.
        
        Args:
            result: Result dictionary from execute_query
        
        Returns:
            Formatted string representation
        """
        if not result['success']:
            return f"Query failed: {result['error']}"
        
        if result['row_count'] == 0:
            return "Query executed successfully but returned no rows."
        
        output = []
        output.append(f"Query returned {result['row_count']} rows")
        if result.get('truncated'):
            output.append(f" (truncated from {result.get('total_rows', 'many')} total rows)")
        output.append("\n\n")
        
        data = result['data']
        columns = result['columns']
        
        if len(data) <= 10:
            # For small results, show all rows
            output.append("Results:\n")
            for i, row in enumerate(data, 1):
                output.append(f"\nRow {i}:\n")
                for col in columns:
                    value = row.get(col)
                    output.append(f"  {col}: {value}\n")
        else:
            # For large results, show summary
            output.append(f"First 5 rows:\n")
            for i, row in enumerate(data[:5], 1):
                output.append(f"\nRow {i}:\n")
                for col in columns:
                    value = row.get(col)
                    output.append(f"  {col}: {value}\n")
            
            output.append(f"\n... and {len(data) - 5} more rows\n")
        
        return ''.join(output)


_executor = None


def get_sql_executor() -> SQLExecutor:
    """Get global SQL executor instance"""
    global _executor
    if _executor is None:
        _executor = SQLExecutor(query_timeout=30)
    return _executor