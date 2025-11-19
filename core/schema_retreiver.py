"""
Schema retrieval functions for dynamic schema injection.
Retrieves only relevant schema based on query context.
"""

from typing import List, Dict, Any, Optional
from core.vector_retriever import get_vector_retriever
from core.schema_context_minimal import CORE_SCHEMA_ALWAYS_PRESENT, get_table_dependencies


def retrieve_relevant_schema(
    query: str,
    use_case: Optional[str] = None,
    comparison_type: Optional[str] = None,
    models_requested: Optional[List[str]] = None,
    retry_count: int = 0,
    top_k: int = 3
) -> str:
    """
    Retrieve relevant schema based on query context.
    
    Uses hybrid approach:
    - Always includes core schema
    - Retrieves detailed schema from Pinecone based on query
    - Auto-expands JOIN dependencies
    - Falls back to more retrieval on retry
    
    Args:
        query: User query text
        use_case: Parsed use case (e.g., 'NRx_forecasting')
        comparison_type: Type of comparison (e.g., 'performance', 'drift')
        models_requested: List of model names mentioned
        retry_count: Number of retry attempts (0 = first attempt)
        top_k: Number of schema chunks to retrieve
        
    Returns:
        Formatted schema context string
    """
    
    # Start with core schema (always present)
    schema_parts = [CORE_SCHEMA_ALWAYS_PRESENT]
    
    # Adjust retrieval based on retry attempt
    if retry_count == 0:
        # First attempt: Semantic retrieval (top 3)
        k = top_k
    elif retry_count == 1:
        # First retry: Retrieve more (top 5)
        k = top_k + 2
    else:
        # Second+ retry: Retrieve even more (top 8)
        k = top_k + 5
    
    # Build search query for schema
    search_parts = [query]
    if use_case:
        search_parts.append(use_case)
    if comparison_type:
        search_parts.append(comparison_type)
    if models_requested:
        search_parts.extend(models_requested[:3])  # Add up to 3 model names
    
    search_query = " ".join(search_parts)
    
    # Build filters for Pinecone
    filters = {"category": {"$in": ["schema", "query_pattern", "join_guide", "critical_rules"]}}
    
    # Add use_case filter if available
    if use_case:
        # Note: This requires use_case to be stored in metadata during schema setup
        # For now, we'll rely on semantic search
        pass
    
    # Retrieve from Pinecone
    retriever = get_vector_retriever()
    
    try:
        schema_chunks = retriever.search(
            query=search_query,
            n_results=k,
            filter_dict=filters,
            namespace="schema_knowledge"
        )
        
        if not schema_chunks:
            print(f"[Schema Retrieval] Warning: No schema chunks retrieved for query: {search_query[:100]}")
            schema_parts.append("\n## RELEVANT SCHEMA\n(No additional schema retrieved - using core only)\n")
            return "\n".join(schema_parts)
        
        # Extract table names from retrieved chunks
        retrieved_tables = set()
        for chunk in schema_chunks:
            table_name = chunk['metadata'].get('table_name')
            if table_name:
                retrieved_tables.add(table_name)
        
        # Auto-expand dependencies
        if retrieved_tables:
            expanded_tables = get_table_dependencies(list(retrieved_tables))
            
            # Fetch any missing dependency tables
            missing_tables = set(expanded_tables) - retrieved_tables
            if missing_tables:
                print(f"[Schema Retrieval] Auto-expanding dependencies: {missing_tables}")
                
                for table in missing_tables:
                    # Retrieve specific table document
                    table_chunks = retriever.search(
                        query=f"{table} table schema",
                        n_results=1,
                        filter_dict={"category": {"$eq": "schema"}, "table_name": {"$eq": table}},
                        namespace="schema_knowledge"
                    )
                    if table_chunks:
                        schema_chunks.extend(table_chunks)
        
        # Format retrieved schema
        schema_parts.append("\n## RELEVANT SCHEMA FOR THIS QUERY\n")
        schema_parts.append(f"Retrieved {len(schema_chunks)} schema components:\n")
        
        # Group by category
        by_category = {}
        for chunk in schema_chunks:
            category = chunk['metadata'].get('category', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(chunk)
        
        # Add schemas in order: tables, views, patterns, rules
        category_order = ['schema', 'query_pattern', 'join_guide', 'critical_rules']
        
        for category in category_order:
            if category not in by_category:
                continue
            
            chunks = by_category[category]
            
            if category == 'schema':
                schema_parts.append("\n### Tables & Views\n")
            elif category == 'query_pattern':
                schema_parts.append("\n### Relevant Query Patterns\n")
            elif category == 'join_guide':
                schema_parts.append("\n### JOIN Relationships\n")
            elif category == 'critical_rules':
                schema_parts.append("\n### Critical Rules\n")
            
            for chunk in chunks:
                schema_parts.append(chunk['content'])
                schema_parts.append("\n---\n")
        
        # Add retry context if applicable
        if retry_count > 0:
            schema_parts.append(f"\n## RETRY ATTEMPT {retry_count}\n")
            schema_parts.append(f"Retrieved {k} schema components (expanded from {top_k} to help with retry).\n")
        
        return "\n".join(schema_parts)
    
    except Exception as e:
        print(f"[Schema Retrieval] Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Return just core schema
        schema_parts.append("\n## SCHEMA RETRIEVAL ERROR\n")
        schema_parts.append(f"Error retrieving schema: {str(e)}\n")
        schema_parts.append("Using core schema only.\n")
        
        return "\n".join(schema_parts)


def format_schema_summary(schema_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved schema chunks into readable summary.
    
    Args:
        schema_chunks: List of schema chunk dicts from Pinecone
        
    Returns:
        Formatted string
    """
    if not schema_chunks:
        return "(No schema chunks provided)"
    
    summary_parts = []
    
    for i, chunk in enumerate(schema_chunks, 1):
        doc_id = chunk.get('doc_id', 'unknown')
        category = chunk['metadata'].get('category', 'unknown')
        table_name = chunk['metadata'].get('table_name', '')
        score = chunk.get('score', 0.0)
        
        if table_name:
            summary_parts.append(f"{i}. {table_name} ({category}) - relevance: {score:.2f}")
        else:
            summary_parts.append(f"{i}. {doc_id} ({category}) - relevance: {score:.2f}")
    
    return "\n".join(summary_parts)


def get_table_list_from_chunks(schema_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract list of table names from retrieved schema chunks.
    
    Args:
        schema_chunks: List of schema chunk dicts
        
    Returns:
        List of table names
    """
    tables = []
    for chunk in schema_chunks:
        table_name = chunk['metadata'].get('table_name')
        if table_name and table_name not in tables:
            tables.append(table_name)
    
    return tables