"""
One-time script to create and populate ChromaDB with document embeddings.
Run this once to set up the persistent vector database.

Usage:
    python setup_vector_db.py

This will:
1. Create a persistent ChromaDB instance in ./chroma_db directory
2. Generate embeddings using sentence-transformers
3. Store all documents from vector_db_context.py
4. Create indexes for efficient retrieval
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from core.vector_db_context import VECTOR_DB_DOCUMENTS, DOCUMENT_SUMMARY
from tqdm import tqdm
import json


class VectorDBSetup:
    """Setup and populate ChromaDB with pharma analytics documents"""
    
    def __init__(self, persist_directory="./chroma_db"):
        """
        Initialize vector DB setup
        
        Args:
            persist_directory: Path where ChromaDB will persist data
        """
        self.persist_directory = persist_directory
        self.collection_name = "pharma_analytics_docs"
        
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        print("Loading sentence-transformer model...")
        # Using a model optimized for semantic search
        # all-MiniLM-L6-v2: Fast and efficient, good for general purpose
        # Alternative: all-mpnet-base-v2 (more accurate but slower)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Model loaded: all-MiniLM-L6-v2")
    
    def reset_collection(self):
        """Delete existing collection if it exists"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            print(f"No existing collection to delete: {e}")
    
    def create_collection(self):
        """Create new collection with metadata"""
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "Pharma commercial analytics knowledge base",
                "embedding_model": "all-MiniLM-L6-v2",
                "total_documents": len(VECTOR_DB_DOCUMENTS)
            }
        )
        print(f"Created collection: {self.collection_name}")
        return self.collection
    
    def prepare_documents(self):
        """
        Prepare documents for embedding
        
        Returns:
            Tuple of (ids, documents, metadatas)
        """
        print("\nPreparing documents for embedding...")
        
        ids = []
        documents = []
        metadatas = []
        
        for doc in VECTOR_DB_DOCUMENTS:
            doc_id = doc['doc_id']
            
            # Combine title and content for embedding
            # This gives context to the embedding
            full_text = f"{doc['title']}\n\n{doc['content']}"
            
            # Prepare metadata (ChromaDB requires JSON-serializable values)
            metadata = {
                "doc_id": doc_id,
                "category": doc['category'],
                "title": doc['title'],
                "keywords": json.dumps(doc['keywords']),  # Store as JSON string
            }
            
            # Add metadata fields if they exist
            if 'metadata' in doc and doc['metadata']:
                for key, value in doc['metadata'].items():
                    # Convert complex types to JSON strings
                    if isinstance(value, (dict, list)):
                        metadata[f"meta_{key}"] = json.dumps(value)
                    else:
                        metadata[f"meta_{key}"] = str(value)
            
            ids.append(doc_id)
            documents.append(full_text)
            metadatas.append(metadata)
        
        print(f"Prepared {len(documents)} documents")
        return ids, documents, metadatas
    
    def generate_embeddings(self, documents):
        """
        Generate embeddings for all documents
        
        Args:
            documents: List of document texts
        
        Returns:
            List of embeddings
        """
        print("\nGenerating embeddings...")
        print("This may take a few minutes depending on document count...")
        
        # Generate embeddings with progress bar
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        return embeddings
    
    def populate_collection(self, ids, embeddings, documents, metadatas):
        """
        Add documents to collection
        
        Args:
            ids: Document IDs
            embeddings: Document embeddings
            documents: Original document texts
            metadatas: Document metadata
        """
        print("\nPopulating ChromaDB collection...")
        
        # ChromaDB can handle batch inserts efficiently
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(ids)} documents to collection")
    
    def verify_setup(self):
        """Verify the collection was created correctly"""
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        count = self.collection.count()
        print(f"Total documents in collection: {count}")
        
        # Test query
        print("\nTesting semantic search with query: 'ensemble model performance'")
        results = self.collection.query(
            query_embeddings=self.encoder.encode(
                ["ensemble model performance"],
                convert_to_numpy=True
            ).tolist(),
            n_results=3
        )
        
        print(f"\nTop 3 results:")
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0]), 1):
            metadata = results['metadatas'][0][i-1]
            print(f"\n{i}. {doc_id}: {metadata['title']}")
            print(f"   Category: {metadata['category']}")
            print(f"   Distance: {distance:.4f}")
        
        # Category breakdown
        print("\n" + "-"*80)
        print("Documents by category:")
        categories = {}
        for metadata in self.collection.get()['metadatas']:
            cat = metadata['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} documents")
        
        print("\n" + "="*80)
        print("Setup complete! Vector database is ready for use.")
        print(f"Location: {os.path.abspath(self.persist_directory)}")
        print("="*80)
    
    def run(self):
        """Execute the complete setup process"""
        print("\n" + "="*80)
        print("PHARMA ANALYTICS VECTOR DATABASE SETUP")
        print("="*80 + "\n")
        
        # Step 1: Reset if exists
        self.reset_collection()
        
        # Step 2: Create new collection
        self.create_collection()
        
        # Step 3: Prepare documents
        ids, documents, metadatas = self.prepare_documents()
        
        # Step 4: Generate embeddings
        embeddings = self.generate_embeddings(documents)
        
        # Step 5: Populate collection
        self.populate_collection(ids, embeddings, documents, metadatas)
        
        # Step 6: Verify
        self.verify_setup()
        
        print("\n✓ Vector database setup completed successfully!")
        print("\nYou can now:")
        print("1. Delete this setup script (setup_vector_db.py)")
        print("2. Delete core/vector_db_context.py (no longer needed)")
        print("3. Use the updated context_retrieval_agent in agent/nodes.py")
        print("\nThe vector database will persist in ./chroma_db directory")


def main():
    """Main execution"""
    import sys
    
    # Check if chromadb is installed
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("\n" + "="*80)
        print("ERROR: Missing required packages")
        print("="*80)
        print("\nPlease install required packages:")
        print("\npip install chromadb sentence-transformers")
        print("\nOr update requirements.txt and run:")
        print("pip install -r requirements.txt")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Confirm before proceeding
    print("\n" + "="*80)
    print("PHARMA ANALYTICS VECTOR DATABASE SETUP")
    print("="*80)
    print("\nThis script will:")
    print("1. Create a persistent ChromaDB instance in ./chroma_db")
    print("2. Generate embeddings for all documents using sentence-transformers")
    print("3. Store embeddings for fast semantic retrieval")
    print("\nThis is a ONE-TIME setup process.")
    print("\nEstimated time: 2-5 minutes")
    print("Estimated storage: ~50-100 MB")
    
    response = input("\nProceed with setup? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nSetup cancelled.")
        return
    
    # Run setup
    try:
        setup = VectorDBSetup(persist_directory="./chroma_db")
        setup.run()
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Test the vector DB with a sample query:")
        print("   python -c \"from core.vector_retriever import VectorRetriever; vr = VectorRetriever(); print(vr.search('ensemble performance')[:2])\"")
        print("\n2. Run your main application:")
        print("   python main.py")
        print("\n3. Or run Streamlit app:")
        print("   streamlit run app.py")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR during setup")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()