import os
from typing import Dict, List, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()


class ConversationMemoryManager:
    """Manages conversation memory in Pinecone"""
    
    def __init__(self, index_name: str = "pharma-analytics"):
        """
        Initialize memory manager
        
        Args:
            index_name: Pinecone index name
        """
        self.index_name = index_name
        self.namespace = "conversation_memory"
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        # Load encoder (same as domain knowledge)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✓ Memory manager initialized (index: {index_name}, namespace: {self.namespace})")
    
    def store_insight(
        self,
        session_id: str,
        turn_number: int,
        summary: str,
        full_insight: str,
        metadata: Dict
    ) -> bool:
        """
        Store both summary and full insight in Pinecone
        
        Args:
            session_id: Session identifier
            turn_number: Turn number in conversation
            summary: 300-token summary
            full_insight: Full insight text
            metadata: Extracted metadata dict
            
        Returns:
            True if successful
        """
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Generate embeddings
            summary_embedding = self.encoder.encode(summary).tolist()
            full_embedding = self.encoder.encode(full_insight).tolist()
            
            # Prepare metadata (flatten for Pinecone)
            summary_metadata = {
                # Identity
                "session_id": session_id,
                "turn_number": turn_number,
                "chunk_type": "summary",
                "timestamp": timestamp,
                
                # Content
                "summary_text": summary,
                "token_count": len(summary) // 4,  # Approximate
                
                # User query
                "user_query": metadata.get('user_query', ''),
                
                # Entities (convert lists to JSON strings for Pinecone)
                "models": json.dumps(metadata.get('models', [])),
                "use_case": metadata.get('use_case', ''),
                "comparison_type": metadata.get('comparison_type', ''),
                "metrics": json.dumps(metadata.get('metrics', [])),
                
                # Results
                "winner": metadata.get('winner', ''),
                "primary_metric": metadata.get('primary_metric', ''),
                "improvement_pct": float(metadata.get('improvement_pct', 0.0)),
                
                # Content flags
                "has_visualizations": metadata.get('has_visualizations', False),
                "num_visualizations": metadata.get('num_visualizations', 0),
                "has_drift_analysis": metadata.get('has_drift_analysis', False),
                "has_feature_importance": metadata.get('has_feature_importance', False),
                "has_recommendations": metadata.get('has_recommendations', False),
                
                # Linking
                "full_insight_id": f"{session_id}_turn_{turn_number}_full"
            }
            
            full_metadata = {
                # Identity
                "session_id": session_id,
                "turn_number": turn_number,
                "chunk_type": "full_insight",
                "timestamp": timestamp,
                
                # Content (Pinecone has 40KB metadata limit)
                "full_insight_text": full_insight[:40000],  # Truncate if needed
                "token_count": len(full_insight) // 4,
                
                # Key metadata for filtering
                "models": json.dumps(metadata.get('models', [])),
                "use_case": metadata.get('use_case', ''),
                "comparison_type": metadata.get('comparison_type', ''),
                
                # Flags
                "retrieve_rarely": True,
                
                # Linking
                "summary_id": f"{session_id}_turn_{turn_number}_summary"
            }
            
            # Prepare vectors
            vectors = [
                {
                    "id": f"{session_id}_turn_{turn_number}_summary",
                    "values": summary_embedding,
                    "metadata": summary_metadata
                },
                {
                    "id": f"{session_id}_turn_{turn_number}_full",
                    "values": full_embedding,
                    "metadata": full_metadata
                }
            ]
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            
            print(f"✓ Stored turn {turn_number} in memory (summary + full insight)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to store turn {turn_number}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def fetch_full_insight(
        self,
        session_id: str,
        turn_number: int
    ) -> Optional[Dict]:
        """
        Fetch full insight for a specific turn
        
        Args:
            session_id: Session identifier
            turn_number: Turn number
            
        Returns:
            Dict with full insight or None
        """
        try:
            doc_id = f"{session_id}_turn_{turn_number}_full"
            
            result = self.index.fetch(
                ids=[doc_id],
                namespace=self.namespace
            )
            
            if doc_id in result.get('vectors', {}):
                vector_data = result['vectors'][doc_id]
                return {
                    'success': True,
                    'turn': turn_number,
                    'full_insight': vector_data['metadata']['full_insight_text'],
                    'timestamp': vector_data['metadata']['timestamp']
                }
            else:
                return {
                    'success': False,
                    'error': f'Turn {turn_number} not found in session {session_id}'
                }
                
        except Exception as e:
            print(f"Error fetching full insight for turn {turn_number}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_summaries(
        self,
        query: str,
        session_id: str,
        filters: Optional[Dict] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search conversation summaries with hybrid filtering
        
        Args:
            query: Search query
            session_id: Session identifier
            filters: Optional Pinecone filters
            top_k: Number of results
            
        Returns:
            List of summary dicts
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            # Build filter
            filter_dict = filters or {}
            filter_dict["session_id"] = {"$eq": session_id}
            filter_dict["chunk_type"] = {"$eq": "summary"}  # Only summaries
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            # Format results
            summaries = []
            for match in results.get('matches', []):
                metadata = match['metadata']
                summaries.append({
                    'turn': metadata['turn_number'],
                    'summary': metadata['summary_text'],
                    'relevance': match['score'],
                    'models': json.loads(metadata.get('models', '[]')),
                    'use_case': metadata.get('use_case', ''),
                    'has_visualizations': metadata.get('has_visualizations', False),
                    'timestamp': metadata.get('timestamp', ''),
                    'user_query': metadata.get('user_query', '')
                })
            
            return summaries
            
        except Exception as e:
            print(f"Error searching summaries: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete all memory for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            self.index.delete(
                filter={"session_id": {"$eq": session_id}},
                namespace=self.namespace
            )
            
            print(f"✓ Deleted all memory for session {session_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 30) -> Dict:
        """
        Delete sessions older than specified days
        
        Args:
            days_old: Delete sessions older than this many days
            
        Returns:
            Dict with cleanup stats
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat() + "Z"
            
            print(f"Cleaning up sessions older than {days_old} days (before {cutoff_str})")
            
            # Query for old summaries (to get session_ids)
            # Note: Pinecone doesn't have great support for date range queries
            # We'll fetch summaries and check timestamps
            
            # Get all summaries (limited to avoid overwhelming)
            results = self.index.query(
                vector=[0.0] * 384,  # Dummy vector
                filter={"chunk_type": {"$eq": "summary"}},
                top_k=10000,  # Max we can fetch
                namespace=self.namespace,
                include_metadata=True
            )
            
            old_sessions = set()
            for match in results.get('matches', []):
                timestamp = match['metadata'].get('timestamp', '')
                if timestamp < cutoff_str:
                    session_id = match['metadata']['session_id']
                    old_sessions.add(session_id)
            
            # Delete old sessions
            deleted_count = 0
            for session_id in old_sessions:
                if self.delete_session(session_id):
                    deleted_count += 1
            
            print(f"✓ Cleaned up {deleted_count} old sessions")
            
            return {
                'success': True,
                'sessions_deleted': deleted_count,
                'cutoff_date': cutoff_str
            }
            
        except Exception as e:
            print(f"✗ Cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
_memory_manager = None


def get_memory_manager() -> ConversationMemoryManager:
    """Get or create singleton memory manager"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = ConversationMemoryManager()
    
    return _memory_manager