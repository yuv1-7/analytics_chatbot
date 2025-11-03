"""
Pinecone vector database retrieval interface for semantic search.
Used by context_retrieval_agent to fetch relevant documents.
"""

import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class PineconeRetriever:
    """Interface for retrieving documents from Pinecone"""
    
    def __init__(self, index_name="pharma-analytics"):
        """
        Initialize Pinecone retriever
        
        Args:
            index_name: Name of the Pinecone index
        """
        self.index_name = index_name
        
        # Get API key
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY not found in environment. "
                "Please add it to your .env file"
            )
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Connect to index
        try:
            self.index = self.pc.Index(index_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Pinecone index '{index_name}'. "
                f"Did you run setup_pinecone.py first? Error: {e}"
            )
        
        # Load the same encoder used during setup
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant documents
        
        Args:
            query: Search query text
            n_results: Number of results to return
            category_filter: Optional category to filter by
            filter_dict: Optional Pinecone filter dictionary
        
        Returns:
            List of document dictionaries with content and metadata
        """
        # Generate query embedding
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).tolist()[0]
        
        # Build filter
        filter_clause = filter_dict or {}
        if category_filter:
            filter_clause["category"] = {"$eq": category_filter}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            filter=filter_clause if filter_clause else None
        )
        
        # Format results
        documents = []
        for match in results['matches']:
            doc = {
                'doc_id': match['id'],
                'content': match['metadata'].get('content', ''),
                'score': match['score'],  # Pinecone returns similarity score (0-1)
                'distance': 1 - match['score'],  # Convert to distance for consistency
                'metadata': match['metadata']
            }
            documents.append(doc)
        
        return documents
    
    def search_by_category(
        self,
        query: str,
        category: str,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific category
        
        Args:
            query: Search query
            category: Category to search in
            n_results: Number of results
        
        Returns:
            List of matching documents
        """
        return self.search(query, n_results=n_results, category_filter=category)
    
    def search_multi_category(
        self,
        query: str,
        categories: List[str],
        n_results_per_category: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple categories
        
        Args:
            query: Search query
            categories: List of categories to search
            n_results_per_category: Results per category
        
        Returns:
            Dictionary mapping category to results
        """
        results = {}
        for category in categories:
            results[category] = self.search_by_category(
                query, category, n_results=n_results_per_category
            )
        return results
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Document dict or None
        """
        try:
            result = self.index.fetch(ids=[doc_id])
            
            if not result['vectors'] or doc_id not in result['vectors']:
                return None
            
            vector_data = result['vectors'][doc_id]
            
            doc = {
                'doc_id': doc_id,
                'content': vector_data['metadata'].get('content', ''),
                'metadata': vector_data['metadata']
            }
            
            return doc
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all available categories
        Note: This requires querying the index with metadata
        """
        # Pinecone doesn't have a built-in way to get all unique metadata values
        # We'll return the known categories from our setup
        return [
            "use_case",
            "ensemble_method",
            "metric",
            "business_context",
            "features",
            "troubleshooting"
        ]
    
    def search_with_context(
        self,
        query: str,
        use_case: Optional[str] = None,
        comparison_type: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Context-aware search based on parsed query intent
        
        Args:
            query: Original query text
            use_case: Detected use case
            comparison_type: Type of comparison requested
            n_results: Number of results
        
        Returns:
            List of relevant documents
        """
        # Determine which categories to prioritize
        categories_to_search = []
        
        if use_case:
            categories_to_search.append('use_case')
        
        if comparison_type:
            if 'ensemble' in comparison_type.lower():
                categories_to_search.extend(['ensemble_method', 'troubleshooting'])
            if 'performance' in comparison_type.lower():
                categories_to_search.append('metric')
            if 'drift' in comparison_type.lower():
                categories_to_search.extend(['use_case', 'troubleshooting'])
            if 'feature' in comparison_type.lower():
                categories_to_search.extend(['features', 'use_case'])
        
        # Remove duplicates while preserving order
        seen = set()
        categories_to_search = [x for x in categories_to_search if not (x in seen or seen.add(x))]
        
        if not categories_to_search:
            # Default broad search
            return self.search(query, n_results=n_results)
        
        # Search across prioritized categories
        all_results = []
        results_per_category = max(2, n_results // len(categories_to_search))
        
        for category in categories_to_search:
            category_results = self.search_by_category(
                query, category, n_results=results_per_category
            )
            all_results.extend(category_results)
        
        # Sort by score (higher is better in Pinecone) and take top n_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:n_results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        
        return {
            'total_documents': stats['total_vector_count'],
            'dimension': stats['dimension'],
            'index_name': self.index_name,
            'index_fullness': stats.get('index_fullness', 0),
            'namespaces': stats.get('namespaces', {})
        }
    
    def delete_all(self):
        """Delete all vectors from the index (use with caution!)"""
        print("WARNING: This will delete all vectors from the index!")
        response = input("Are you sure? Type 'DELETE' to confirm: ")
        
        if response == 'DELETE':
            self.index.delete(delete_all=True)
            print("All vectors deleted from index")
        else:
            print("Deletion cancelled")


# Singleton instance for reuse
_retriever_instance = None


def get_vector_retriever() -> PineconeRetriever:
    """
    Get or create singleton PineconeRetriever instance
    
    Returns:
        PineconeRetriever instance
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = PineconeRetriever()
    
    return _retriever_instance