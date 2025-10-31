import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from core.vector_db_context import VECTOR_DB_DOCUMENTS
import json
import time

def setup_vector_db():
    """Initialize ChromaDB in-memory or local directory (ephemeral)"""
    
    start_time = time.time()
    persist_directory = "./chroma_db"
    collection_name = "pharma_analytics_docs"
    
    os.makedirs(persist_directory, exist_ok=True)
    
    print("=" * 60)
    print("🔧 Initializing Vector Database...")
    print("=" * 60)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Check if collection already exists (in case of restart without rebuild)
    try:
        existing_collection = client.get_collection(name=collection_name)
        count = existing_collection.count()
        if count == len(VECTOR_DB_DOCUMENTS):
            print(f"✓ Vector DB already initialized with {count} documents")
            print(f"⏱️  Completed in {time.time() - start_time:.2f} seconds")
            return
        else:
            print(f"⚠️  Found incomplete collection ({count} docs), rebuilding...")
            client.delete_collection(name=collection_name)
    except:
        print("📦 Creating new collection...")
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Pharma commercial analytics knowledge base"}
    )
    
    # Load sentence transformer
    print("📥 Loading embedding model (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare documents
    print(f"📄 Processing {len(VECTOR_DB_DOCUMENTS)} documents...")
    
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for i, doc in enumerate(VECTOR_DB_DOCUMENTS, 1):
        # Generate embedding
        embedding = encoder.encode([doc['content']], convert_to_numpy=True)[0].tolist()
        
        # Prepare metadata
        metadata = {
            'category': doc['category'],
            'title': doc['title'],
            'keywords': json.dumps(doc['keywords'])
        }
        
        # Flatten nested metadata
        if 'metadata' in doc:
            for key, value in doc['metadata'].items():
                metadata[f'meta_{key}'] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        
        ids.append(doc['doc_id'])
        documents.append(doc['content'])
        metadatas.append(metadata)
        embeddings.append(embedding)
        
        # Progress indicator
        if i % 5 == 0 or i == len(VECTOR_DB_DOCUMENTS):
            print(f"  Processed {i}/{len(VECTOR_DB_DOCUMENTS)} documents...")
    
    # Batch insert
    print("💾 Inserting documents into vector database...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    
    # Verify
    count = collection.count()
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print(f"✅ Vector DB initialized successfully!")
    print(f"   📊 Total documents: {count}")
    print(f"   ⏱️  Time taken: {elapsed:.2f} seconds")
    print("=" * 60)
    
    # Quick test
    test_results = collection.query(
        query_embeddings=[encoder.encode(["NRx forecasting"], convert_to_numpy=True)[0].tolist()],
        n_results=3
    )
    
    print("\n🔍 Test query results for 'NRx forecasting':")
    for i, doc_id in enumerate(test_results['ids'][0], 1):
        title = test_results['metadatas'][0][i-1].get('title', 'Unknown')
        print(f"  {i}. {doc_id}: {title}")
    print()

if __name__ == "__main__":
    try:
        setup_vector_db()
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize vector database: {e}")
        import traceback
        traceback.print_exc()
        exit(1)