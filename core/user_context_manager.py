import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class UserContextManager:
    """Manages user-specific personalized context in Pinecone"""
    
    # Category detection patterns (comprehensive regex)
    CATEGORY_PATTERNS = {
        'competitors': r'\b(competitor|competitive|rival|vs\.|versus|against|market leader|competition|competing)\b',
        'products': r'\b(product|drug|brand|therapeutic|indication|molecule|compound|treatment|medication|therapy)\b',
        'markets': r'\b(geography|region|territory|country|market|segment|area|zone|locale|jurisdiction)\b',
        'kpis': r'\b(metric|kpi|target|goal|objective|forecast|budget|milestone|benchmark|performance indicator)\b',
        'hcp_targeting': r'\b(hcp|physician|doctor|prescriber|specialty|decile|healthcare provider|clinician|practitioner)\b',
        'campaigns': r'\b(campaign|promotion|marketing|launch|message|channel|outreach|initiative|program|activation)\b',
    }
    
    def __init__(self, index_name: str = "pharma-analytics"):
        """Initialize user context manager"""
        self.index_name = index_name
        self.namespace = "user_context"
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "800"))
        self.max_versions = int(os.getenv("MAX_CONTEXT_VERSIONS_PER_CHUNK", "10"))
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        # Load encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✓ User context manager initialized (namespace: {self.namespace})")
    
    def detect_category(self, text: str) -> str:
        """
        Auto-detect category using regex patterns
        
        Args:
            text: Chunk text
            
        Returns:
            Category name or 'general'
        """
        text_lower = text.lower()
        
        # Check each category pattern
        for category, pattern in self.CATEGORY_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return category
        
        return 'general'
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraphs and size limits
        
        Args:
            text: Full context text
            
        Returns:
            List of chunk strings
        """
        if not text or not text.strip():
            return []
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds limit, save current chunk
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def store_user_context(
        self,
        user_id: str,
        context_text: str
    ) -> Dict:
        """
        Store user context (auto-chunks, versions, and archives old)
        
        Args:
            user_id: User identifier
            context_text: Full context text from user
            
        Returns:
            Result dict with stats
        """
        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Archive ALL existing active chunks for this user
            print(f"[Context] Archiving old context for user {user_id}...")
            self._archive_user_context(user_id)
            
            # Chunk the new text
            chunks = self.chunk_text(context_text)
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'No valid chunks created from input'
                }
            
            print(f"[Context] Created {len(chunks)} chunks for user {user_id}")
            
            # Store each chunk
            vectors = []
            for chunk_idx, chunk_text in enumerate(chunks):
                # Detect category
                category = self.detect_category(chunk_text)
                
                # Generate embedding
                embedding = self.encoder.encode(chunk_text).tolist()
                
                # Create vector ID (version 1 for new context)
                vector_id = f"{user_id}_context_{chunk_idx}_v1"
                
                # Metadata (flat for Pinecone)
                metadata = {
                    "user_id": user_id,
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk_text[:40000],  # Pinecone 40KB limit
                    "category": category,
                    "version": 1,
                    "is_archived": False,
                    "created_at": timestamp,
                    "updated_at": timestamp
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                print(f"  - Chunk {chunk_idx}: {category} ({len(chunk_text)} chars)")
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            
            print(f"✓ Stored {len(chunks)} chunks for user {user_id}")
            
            return {
                'success': True,
                'chunks_created': len(chunks),
                'categories': [v['metadata']['category'] for v in vectors],
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"✗ Failed to store context for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _archive_user_context(self, user_id: str):
        """
        Mark all active chunks for user as archived
        Then cleanup old versions exceeding limit
        """
        try:
            # Get all active chunks for user
            results = self.index.query(
                vector=[0.0] * 384,
                filter={
                    "user_id": {"$eq": user_id},
                    "is_archived": {"$eq": False}
                },
                top_k=100,
                namespace=self.namespace,
                include_metadata=True
            )
            
            if not results.get('matches'):
                return
            
            # Update each to archived
            updates = []
            for match in results['matches']:
                metadata = match['metadata'].copy()
                metadata['is_archived'] = True
                metadata['updated_at'] = datetime.utcnow().isoformat() + "Z"
                
                updates.append({
                    "id": match['id'],
                    "values": match['values'],
                    "metadata": metadata
                })
            
            if updates:
                self.index.upsert(vectors=updates, namespace=self.namespace)
                print(f"  ✓ Archived {len(updates)} old chunks")
            
            # Cleanup old versions per chunk
            self._cleanup_old_versions(user_id)
            
        except Exception as e:
            print(f"  ✗ Archive failed: {e}")
    
    def _cleanup_old_versions(self, user_id: str):
        """
        Delete oldest archived versions if exceeding max_versions per chunk
        """
        try:
            # Get all archived chunks for user, grouped by chunk_index
            results = self.index.query(
                vector=[0.0] * 384,
                filter={
                    "user_id": {"$eq": user_id},
                    "is_archived": {"$eq": True}
                },
                top_k=1000,
                namespace=self.namespace,
                include_metadata=True
            )
            
            if not results.get('matches'):
                return
            
            # Group by chunk_index
            chunks_by_index = {}
            for match in results['matches']:
                chunk_idx = match['metadata']['chunk_index']
                if chunk_idx not in chunks_by_index:
                    chunks_by_index[chunk_idx] = []
                chunks_by_index[chunk_idx].append(match)
            
            # For each chunk_index, keep only latest max_versions
            ids_to_delete = []
            for chunk_idx, versions in chunks_by_index.items():
                if len(versions) > self.max_versions:
                    # Sort by version (oldest first)
                    versions_sorted = sorted(versions, key=lambda x: x['metadata']['version'])
                    
                    # Delete oldest
                    to_delete = versions_sorted[:len(versions) - self.max_versions]
                    ids_to_delete.extend([v['id'] for v in to_delete])
            
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete, namespace=self.namespace)
                print(f"  ✓ Cleaned up {len(ids_to_delete)} old versions")
                
        except Exception as e:
            print(f"  ✗ Cleanup failed: {e}")
    
    def retrieve_user_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve top-k semantically relevant context chunks for user
        
        Args:
            user_id: User identifier
            query: Query text for semantic search
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunk dicts with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            # Search with user filter (only active chunks)
            results = self.index.query(
                vector=query_embedding,
                filter={
                    "user_id": {"$eq": user_id},
                    "is_archived": {"$eq": False}
                },
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            chunks = []
            for match in results.get('matches', []):
                metadata = match['metadata']
                chunks.append({
                    'chunk_text': metadata['chunk_text'],
                    'category': metadata['category'],
                    'relevance': match['score'],
                    'chunk_index': metadata['chunk_index'],
                    'version': metadata['version'],
                    'created_at': metadata.get('created_at', ''),
                    'updated_at': metadata.get('updated_at', '')
                })
            
            return chunks
            
        except Exception as e:
            print(f"✗ Retrieve failed for user {user_id}: {e}")
            return []
    
    def get_user_context_summary(self, user_id: str) -> Dict:
        """
        Get summary of user's current context
        
        Returns:
            Dict with chunk count, categories, last updated
        """
        try:
            results = self.index.query(
                vector=[0.0] * 384,
                filter={
                    "user_id": {"$eq": user_id},
                    "is_archived": {"$eq": False}
                },
                top_k=100,
                namespace=self.namespace,
                include_metadata=True
            )
            
            if not results.get('matches'):
                return {
                    'success': True,
                    'has_context': False,
                    'chunk_count': 0
                }
            
            chunks = results['matches']
            categories = {}
            last_updated = None
            
            for chunk in chunks:
                cat = chunk['metadata']['category']
                categories[cat] = categories.get(cat, 0) + 1
                
                updated = chunk['metadata'].get('updated_at')
                if updated and (not last_updated or updated > last_updated):
                    last_updated = updated
            
            return {
                'success': True,
                'has_context': True,
                'chunk_count': len(chunks),
                'categories': categories,
                'last_updated': last_updated,
                'total_chars': sum(len(c['metadata']['chunk_text']) for c in chunks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_version_history(self, user_id: str) -> List[Dict]:
        """
        Get version history for user (all versions, including archived)
        
        Returns:
            List of version dicts sorted by timestamp (newest first)
        """
        try:
            results = self.index.query(
                vector=[0.0] * 384,
                filter={"user_id": {"$eq": user_id}},
                top_k=1000,
                namespace=self.namespace,
                include_metadata=True
            )
            
            versions = []
            for match in results.get('matches', []):
                metadata = match['metadata']
                versions.append({
                    'chunk_index': metadata['chunk_index'],
                    'version': metadata['version'],
                    'category': metadata['category'],
                    'is_archived': metadata['is_archived'],
                    'created_at': metadata.get('created_at', ''),
                    'updated_at': metadata.get('updated_at', ''),
                    'chunk_preview': metadata['chunk_text'][:100] + '...'
                })
            
            # Sort by updated_at descending
            versions.sort(key=lambda x: x['updated_at'], reverse=True)
            
            return versions
            
        except Exception as e:
            print(f"✗ Version history failed: {e}")
            return []
    
    def delete_user_context(self, user_id: str) -> Dict:
        """
        Delete ALL context for user (including archived versions)
        
        Returns:
            Result dict
        """
        try:
            self.index.delete(
                filter={"user_id": {"$eq": user_id}},
                namespace=self.namespace
            )
            
            print(f"✓ Deleted all context for user {user_id}")
            
            return {
                'success': True,
                'message': f'All context deleted for user {user_id}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
_user_context_manager = None


def get_user_context_manager() -> UserContextManager:
    """Get or create singleton user context manager"""
    global _user_context_manager
    
    if _user_context_manager is None:
        _user_context_manager = UserContextManager()
    
    return _user_context_manager