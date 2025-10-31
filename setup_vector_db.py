"""
One-time script to create and populate Pinecone with document embeddings.
Run this once to set up the Pinecone vector database.

Usage:
    python setup_vector_db.py

This will:
1. Create a Pinecone index
2. Generate embeddings using sentence-transformers
3. Store all documents from vector_db_context.py
4. Create metadata for efficient filtering
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from core.vector_db_context import VECTOR_DB_DOCUMENTS
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv()


class PineconeSetup:
    """Setup and populate Pinecone with pharma analytics documents"""
    
    def __init__(self, index_name="pharma-analytics"):
        """
        Initialize Pinecone setup
        
        Args:
            index_name: Name of the Pinecone index to create
        """
        self.index_name = index_name
        
        # Get API key from environment
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY not found in environment. "
                "Please add it to your .env file"
            )
        
        print("Initializing Pinecone...")
        self.pc = Pinecone(api_key=api_key)
        
        print("Loading sentence-transformer model...")
        # Using the same model as ChromaDB for consistency
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        print(f"Model loaded: all-MiniLM-L6-v2 (dimension: {self.embedding_dimension})")
    
    def delete_index_if_exists(self):
        """Delete existing index if it exists"""
        try:
            existing_indexes = self.pc.list_indexes()
            if self.index_name in [idx.name for idx in existing_indexes]:
                print(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                time.sleep(5)
                print("Index deleted successfully")
        except Exception as e:
            print(f"Note: {e}")
    
    def create_index(self):
        """Create new Pinecone index with serverless spec"""
        print(f"\nCreating Pinecone index: {self.index_name}")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=self.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        print("Waiting for index to be ready...")
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        print(f"Index created successfully: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
        return self.index
    
    def prepare_documents(self):
        """
        Prepare documents for embedding
        
        Returns:
            List of (id, text, metadata) tuples
        """
        print("\nPreparing documents for embedding...")
        
        prepared_docs = []
        
        for doc in VECTOR_DB_DOCUMENTS:
            doc_id = doc['doc_id']
            
            full_text = f"{doc['title']}\n\n{doc['content']}"
            
            metadata = {
                "doc_id": doc_id,
                "category": doc['category'],
                "title": doc['title'],
                "content": doc['content'],
                "keywords": doc['keywords'],
            }
            
            # Flatten nested metadata (if any)
            if 'metadata' in doc and doc['metadata']:
                for key, value in doc['metadata'].items():
                    metadata[f"meta_{key}"] = value
            
            prepared_docs.append((doc_id, full_text, metadata))
        
        print(f"Prepared {len(prepared_docs)} documents")
        return prepared_docs
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for all documents
        
        Args:
            texts: List of document texts
        
        Returns:
            List of embeddings
        """
        print("\nGenerating embeddings...")
        print("This may take a few minutes...")
        
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings


    # ------------------------------------------------------------
    # ✅ FIX: Ensure Pinecone-safe metadata (no nested dicts/lists)
    # ------------------------------------------------------------
    def clean_metadata(self, metadata: dict):
        """
        Convert all nested dicts/lists in metadata into JSON strings,
        ensuring Pinecone compatibility.
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                cleaned[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned


    def upsert_to_pinecone(self, prepared_docs, embeddings):
        """
        Upsert documents to Pinecone
        
        Args:
            prepared_docs: List of (id, text, metadata) tuples
            embeddings: List of embeddings
        """
        print("\nUpserting documents to Pinecone...")
        
        vectors = []
        for (doc_id, text, metadata), embedding in zip(prepared_docs, embeddings):
            safe_metadata = self.clean_metadata(metadata)
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": safe_metadata
            })
        
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size)):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Upserted {len(vectors)} vectors to Pinecone")
        time.sleep(2)
    
    def verify_setup(self):
        """Verify the index was created correctly"""
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        stats = self.index.describe_index_stats()
        print(f"Total vectors in index: {stats['total_vector_count']}")
        print(f"Index dimension: {stats['dimension']}")
        
        print("\nTesting semantic search with query: 'ensemble model performance'")
        query_embedding = self.encoder.encode(
            ["ensemble model performance"],
            convert_to_numpy=True
        ).tolist()[0]
        
        results = self.index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        print(f"\nTop 3 results:")
        for i, match in enumerate(results['matches'], 1):
            print(f"\n{i}. {match['id']}: {match['metadata']['title']}")
            print(f"   Category: {match['metadata']['category']}")
            print(f"   Score: {match['score']:.4f}")
        
        print("\n" + "-"*80)
        print("Documents by category:")
        
        categories = {}
        for doc in VECTOR_DB_DOCUMENTS:
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} documents")
        
        print("\n" + "="*80)
        print("Setup complete! Pinecone index is ready for use.")
        print(f"Index name: {self.index_name}")
        print(f"Region: us-east-1")
        print("="*80)
    
    def run(self):
        """Execute the complete setup process"""
        print("\n" + "="*80)
        print("PHARMA ANALYTICS PINECONE SETUP")
        print("="*80 + "\n")
        
        self.delete_index_if_exists()
        self.create_index()
        prepared_docs = self.prepare_documents()
        texts = [text for _, text, _ in prepared_docs]
        embeddings = self.generate_embeddings(texts)
        self.upsert_to_pinecone(prepared_docs, embeddings)
        self.verify_setup()
        
        print("\n✓ Pinecone setup completed successfully!")
        print("\nYou can now:")
        print("1. Use the Pinecone retriever in your application")
        print("2. Access your index from anywhere (cloud-based)")
        print("3. Delete the local ./chroma_db directory if you want")
        print("\nThe index is hosted on Pinecone's cloud infrastructure")


def main():
    """Main execution"""
    import sys
    
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("\n" + "="*80)
        print("ERROR: Missing required packages")
        print("="*80)
        print("\nPlease install required packages:")
        print("\npip install pinecone-client sentence-transformers")
        print("\nOr update requirements.txt and run:")
        print("pip install -r requirements.txt")
        print("="*80 + "\n")
        sys.exit(1)
    
    load_dotenv()
    if not os.getenv("PINECONE_API_KEY"):
        print("\n" + "="*80)
        print("ERROR: PINECONE_API_KEY not found")
        print("="*80)
        print("\nPlease add your Pinecone API key to .env file:")
        print("\nPINECONE_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://app.pinecone.io/")
        print("="*80 + "\n")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PHARMA ANALYTICS PINECONE SETUP")
    print("="*80)
    print("\nThis script will:")
    print("1. Create a Pinecone index (cloud-based)")
    print("2. Generate embeddings for all documents")
    print("3. Upload embeddings to Pinecone")
    print("\nThis is a ONE-TIME setup process.")
    print("\nEstimated time: 2-5 minutes")
    print("Cloud storage: Pinecone free tier (100K vectors)")
    
    response = input("\nProceed with setup? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nSetup cancelled.")
        return
    
    try:
        setup = PineconeSetup(index_name="pharma-analytics")
        setup.run()
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Update your application to use Pinecone retriever")
        print("2. Test with: python test_pinecone.py")
        print("3. Run your main application: streamlit run app.py")
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
