"""
Vector database retrieval interface for semantic search.
Used by context_retrieval_agent to fetch relevant documents.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json


class VectorRetriever:
    """Interface for retrieving documents from ChromaDB"""
    
    def __init__(self, persist_directory="./chroma_db", collection_name="pharma_analytics_docs"):
        """
        Initialize vector retriever
        
        Args:
            persist_directory: Path to ChromaDB storage
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load collection '{collection_name}'. "
                f"Did you run setup_vector_db.py first? Error: {e}"
            )
        
        # Load the same encoder used during setup
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant documents
        
        Args:
            query: Search query text
            n_results: Number of results to return
            category_filter: Optional category to filter by
            where: Optional ChromaDB where clause for filtering
        
        Returns:
            List of document dictionaries with content and metadata
        """
        # Generate query embedding
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).tolist()
        
        # Build where clause
        where_clause = where or {}
        if category_filter:
            where_clause["category"] = category_filter
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                'doc_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            
            # Parse JSON fields back to objects
            if 'keywords' in doc['metadata']:
                try:
                    doc['metadata']['keywords'] = json.loads(doc['metadata']['keywords'])
                except:
                    pass
            
            # Parse metadata fields
            parsed_metadata = {}
            for key, value in doc['metadata'].items():
                if key.startswith('meta_'):
                    original_key = key[5:]  # Remove 'meta_' prefix
                    try:
                        parsed_metadata[original_key] = json.loads(value)
                    except:
                        parsed_metadata[original_key] = value
                else:
                    parsed_metadata[key] = value
            
            doc['metadata'] = parsed_metadata
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
            result = self.collection.get(ids=[doc_id])
            
            if not result['ids']:
                return None
            
            doc = {
                'doc_id': result['ids'][0],
                'content': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
            
            # Parse JSON fields
            if 'keywords' in doc['metadata']:
                try:
                    doc['metadata']['keywords'] = json.loads(doc['metadata']['keywords'])
                except:
                    pass
            
            return doc
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories"""
        all_docs = self.collection.get()
        categories = set()
        for metadata in all_docs['metadatas']:
            categories.add(metadata.get('category', 'unknown'))
        return sorted(list(categories))
    
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
        
        # Sort by distance (lower is better) and take top n_results
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        all_docs = self.collection.get()
        
        categories = {}
        for metadata in all_docs['metadatas']:
            cat = metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_documents': len(all_docs['ids']),
            'categories': categories,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory
        }


# Singleton instance for reuse
_retriever_instance = None


def get_vector_retriever() -> VectorRetriever:
    """
    Get or create singleton VectorRetriever instance
    
    Returns:
        VectorRetriever instance
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = VectorRetriever()
    
    return _retriever_instance
