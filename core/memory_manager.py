import os
from typing import Dict, List, Optional, Tuple
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()


class ConversationMemoryManager:
    """Manages conversation memory in Pinecone - Lazy loading with progressive retrieval"""
    
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
        user_query: str,
        insight_text: str
    ) -> bool:
        """
        Store insight as chunks in Pinecone
        
        Args:
            session_id: Session identifier
            turn_number: Turn number in conversation
            user_query: The user's original query
            insight_text: Full insight text to store
            
        Returns:
            True if successful
        """
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Create chunks if insight is very long (>2000 chars)
            # Otherwise store as single chunk
            max_chunk_size = 2000
            chunks = []
            
            if len(insight_text) <= max_chunk_size:
                chunks = [insight_text]
            else:
                # Simple chunking by paragraphs
                paragraphs = insight_text.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= max_chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            
            # Generate embeddings for all chunks
            vectors = []
            for chunk_idx, chunk_text in enumerate(chunks):
                # Generate embedding
                embedding = self.encoder.encode(chunk_text).tolist()
                
                # Prepare metadata (flat structure for Pinecone)
                metadata = {
                    # Identity
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "timestamp": timestamp,
                    
                    # Content
                    "user_query": user_query[:500],  # Truncate if too long
                    "insight_chunk": chunk_text[:40000],  # Pinecone 40KB limit
                    "chunk_length": len(chunk_text)
                }
                
                # Create vector ID
                vector_id = f"{session_id}_turn_{turn_number}_chunk_{chunk_idx}"
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            
            print(f"✓ Stored turn {turn_number} in memory ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to store turn {turn_number}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def lazy_search_conversation_memory(
        self,
        query: str,
        session_id: str,
        current_turn: int,
        max_recent_turns: int = 3
    ) -> Tuple[List[Dict], bool, str]:
        """
        Progressive retrieval strategy:
        1. Search recent chunks (last 3 turns) by relevance
        2. If insufficient, fetch full insights from recent turns
        3. If still insufficient, return flag to ask clarification
        
        Args:
            query: Search query
            session_id: Session identifier
            current_turn: Current turn number
            max_recent_turns: How many recent turns to consider (default 3)
            
        Returns:
            (chunks_found, needs_full_fetch, clarification_reason)
        """
        try:
            print(f"\n[Memory Retrieval] Starting lazy search for turn {current_turn}")
            
            # === STEP 1: Search recent chunks by relevance ===
            print(f"[Step 1] Searching recent {max_recent_turns} turns for relevant chunks...")
            
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            # Calculate turn range (last N turns)
            min_turn = max(1, current_turn - max_recent_turns)
            
            # Build filter for recent turns only
            filter_dict = {
                "session_id": {"$eq": session_id},
                "turn_number": {"$gte": min_turn, "$lt": current_turn}
            }
            
            # Query Pinecone for relevant chunks
            results = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=5,  # Get top 5 most relevant chunks
                namespace=self.namespace,
                include_metadata=True
            )
            
            # Format results
            chunks = []
            for match in results.get('matches', []):
                metadata = match['metadata']
                chunks.append({
                    'turn': metadata['turn_number'],
                    'chunk_index': metadata['chunk_index'],
                    'total_chunks': metadata['total_chunks'],
                    'user_query': metadata['user_query'],
                    'insight_chunk': metadata['insight_chunk'],
                    'relevance': match['score'],
                    'timestamp': metadata.get('timestamp', ''),
                    'is_partial': True  # Flag that this is just a chunk
                })
            
            print(f"[Step 1] Found {len(chunks)} relevant chunks")
            
            # Check if we found any high-relevance chunks
            high_relevance_chunks = [c for c in chunks if c['relevance'] > 0.7]
            
            if high_relevance_chunks:
                print(f"[Step 1] ✓ Found {len(high_relevance_chunks)} high-relevance chunks")
                # Sort by turn and chunk index
                high_relevance_chunks.sort(key=lambda x: (x['turn'], x['chunk_index']))
                return high_relevance_chunks, False, ""
            
            # === STEP 2: No high-relevance chunks, fetch full recent insights ===
            print(f"[Step 2] No high-relevance chunks found. Fetching full insights...")
            
            full_insights = []
            
            # Fetch last 3 turns in reverse order (most recent first)
            for turn_offset in range(1, max_recent_turns + 1):
                target_turn = current_turn - turn_offset
                
                if target_turn < 1:
                    break
                
                print(f"[Step 2] Fetching full insight for turn {target_turn}...")
                full_result = self.get_full_turn(session_id, target_turn)
                
                if full_result['success']:
                    full_insights.append({
                        'turn': target_turn,
                        'user_query': full_result['user_query'],
                        'insight_chunk': full_result['full_insight'],
                        'relevance': 1.0 - (turn_offset * 0.1),  # Decay by recency
                        'timestamp': full_result['timestamp'],
                        'total_chunks': full_result['total_chunks'],
                        'chunk_index': 0,
                        'is_partial': False  # Flag that this is complete
                    })
                    print(f"[Step 2] ✓ Retrieved full insight for turn {target_turn}")
                else:
                    print(f"[Step 2] ✗ Failed to retrieve turn {target_turn}")
            
            if full_insights:
                print(f"[Step 2] ✓ Retrieved {len(full_insights)} full insights")
                return full_insights, True, ""
            
            # === STEP 3: Still nothing found, need clarification ===
            print(f"[Step 3] No memory found in recent turns. Need clarification.")
            
            clarification_msg = (
                f"I couldn't find relevant information in your recent conversation history "
                f"(turns {min_turn}-{current_turn-1}). Could you please:\n"
                f"1. Specify which turn you're referring to (e.g., 'turn 5'), or\n"
                f"2. Provide more details about what you're asking about?"
            )
            
            return [], False, clarification_msg
            
        except Exception as e:
            print(f"Error in lazy search: {e}")
            import traceback
            traceback.print_exc()
            return [], False, f"Memory retrieval error: {str(e)}"
    
    def get_full_turn(
        self,
        session_id: str,
        turn_number: int
    ) -> Dict:
        """
        Retrieve all chunks for a specific turn and reconstruct full insight
        
        Args:
            session_id: Session identifier
            turn_number: Turn number
            
        Returns:
            Dict with full insight or error
        """
        try:
            # Fetch all chunks for this turn
            filter_dict = {
                "session_id": {"$eq": session_id},
                "turn_number": {"$eq": turn_number}
            }
            
            # Query with dummy vector (we want all chunks, not semantic search)
            results = self.index.query(
                vector=[0.0] * 384,
                filter=filter_dict,
                top_k=100,  # Max chunks per turn
                namespace=self.namespace,
                include_metadata=True
            )
            
            if not results.get('matches'):
                return {
                    'success': False,
                    'error': f'Turn {turn_number} not found in session {session_id}'
                }
            
            # Sort chunks by index
            chunks = sorted(
                results['matches'],
                key=lambda x: x['metadata']['chunk_index']
            )
            
            # Reconstruct full insight
            full_insight = "\n\n".join([
                chunk['metadata']['insight_chunk']
                for chunk in chunks
            ])
            
            first_chunk = chunks[0]['metadata']
            
            return {
                'success': True,
                'turn': turn_number,
                'user_query': first_chunk['user_query'],
                'full_insight': full_insight,
                'timestamp': first_chunk['timestamp'],
                'total_chunks': len(chunks)
            }
                
        except Exception as e:
            print(f"Error fetching full turn {turn_number}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_conversation_memory(
        self,
        query: str,
        session_id: str,
        filters: Optional[Dict] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Legacy method for backward compatibility
        Simple semantic search without progressive loading
        """
        try:
            query_embedding = self.encoder.encode(query).tolist()
            
            filter_dict = filters or {}
            filter_dict["session_id"] = {"$eq": session_id}
            
            results = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            chunks = []
            for match in results.get('matches', []):
                metadata = match['metadata']
                chunks.append({
                    'turn': metadata['turn_number'],
                    'chunk_index': metadata['chunk_index'],
                    'total_chunks': metadata['total_chunks'],
                    'user_query': metadata['user_query'],
                    'insight_chunk': metadata['insight_chunk'],
                    'relevance': match['score'],
                    'timestamp': metadata.get('timestamp', ''),
                    'is_partial': True
                })
            
            chunks.sort(key=lambda x: (x['turn'], x['chunk_index']))
            return chunks
            
        except Exception as e:
            print(f"Error searching conversation memory: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all memory for a session"""
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
        """Delete sessions older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat() + "Z"
            
            print(f"Cleaning up sessions older than {days_old} days (before {cutoff_str})")
            
            results = self.index.query(
                vector=[0.0] * 384,
                filter={},
                top_k=10000,
                namespace=self.namespace,
                include_metadata=True
            )
            
            old_sessions = set()
            for match in results.get('matches', []):
                timestamp = match['metadata'].get('timestamp', '')
                if timestamp < cutoff_str:
                    session_id = match['metadata']['session_id']
                    old_sessions.add(session_id)
            
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