"""
One-time setup script to populate Pinecone with schema documents.
Run this after setup_vector_db.py to add schema knowledge.
"""

import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from core.schema_documents import get_schema_documents
from tqdm import tqdm
from dotenv import load_dotenv
import json
import time

load_dotenv()


def clean_metadata(metadata):
    """Convert nested structures to JSON strings for Pinecone"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            cleaned[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def setup_schema_documents():
    """Add schema documents to Pinecone"""
    
    print("\n" + "="*80)
    print("SCHEMA DOCUMENTS SETUP FOR PINECONE")
    print("="*80 + "\n")
    
    # Initialize
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not found in .env file")
        return False
    
    pc = Pinecone(api_key=api_key)
    index_name = "pharma-analytics"
    
    print(f"Connecting to index: {index_name}")
    index = pc.Index(index_name)
    
    # Load encoder
    print("Loading sentence-transformer model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get schema documents
    schema_docs = get_schema_documents()
    print(f"\nFound {len(schema_docs)} schema documents to embed")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    texts = [doc['content'] for doc in schema_docs]
    embeddings = []
    
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_embeddings = encoder.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.extend(batch_embeddings.tolist())
    
    # Prepare vectors
    print("\nPreparing vectors for Pinecone...")
    vectors = []
    for doc, embedding in zip(schema_docs, embeddings):
        metadata = {
            "doc_id": doc['doc_id'],
            "category": doc['category'],
            "content": doc['content'][:40000],  # Pinecone 40KB limit
        }
        
        # Add optional fields
        if 'table_name' in doc:
            metadata['table_name'] = doc['table_name']
        if 'view' in doc:
            metadata['is_view'] = doc['view']
        if 'pattern_name' in doc:
            metadata['pattern_name'] = doc['pattern_name']
        if 'keywords' in doc:
            metadata['keywords'] = json.dumps(doc['keywords'])
        
        # Add metadata fields
        if 'metadata' in doc:
            for key, value in doc['metadata'].items():
                metadata[f"meta_{key}"] = json.dumps(value) if isinstance(value, (dict, list)) else value
        
        vectors.append({
            "id": doc['doc_id'],
            "values": embedding,
            "metadata": clean_metadata(metadata)
        })
    
    # Upsert to Pinecone in schema_knowledge namespace
    print("\nUpserting to Pinecone (namespace: schema_knowledge)...")
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i:i+batch_size]
        index.upsert(
            vectors=batch,
            namespace="schema_knowledge"
        )
    
    time.sleep(2)
    
    # Verify
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    stats = index.describe_index_stats()
    schema_namespace = stats.get('namespaces', {}).get('schema_knowledge', {})
    
    print(f"\nNamespace 'schema_knowledge' stats:")
    print(f"  Vectors: {schema_namespace.get('vector_count', 0)}")
    
    # Test retrieval
    print("\nTesting retrieval with query: 'compare model performance metrics'")
    query_embedding = encoder.encode(["compare model performance metrics"]).tolist()[0]
    
    results = index.query(
        vector=query_embedding,
        top_k=3,
        namespace="schema_knowledge",
        include_metadata=True
    )
    
    print(f"\nTop 3 results:")
    for i, match in enumerate(results['matches'], 1):
        print(f"\n{i}. {match['id']}")
        print(f"   Category: {match['metadata']['category']}")
        print(f"   Score: {match['score']:.4f}")
        if 'table_name' in match['metadata']:
            print(f"   Table: {match['metadata']['table_name']}")
    
    print("\n" + "="*80)
    print("SCHEMA SETUP COMPLETE")
    print("="*80)
    print("\nSchema documents are now available for semantic retrieval.")
    print("The SQL generation agent will automatically retrieve relevant schema based on queries.")
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    print("\nThis script adds schema documents to Pinecone for semantic retrieval.")
    print("This enhances SQL generation by providing only relevant schema context.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        success = setup_schema_documents()
        if success:
            print("\n✓ Success! Schema documents are ready for use.\n")
        else:
            print("\n✗ Setup failed. Please check errors above.\n")
    else:
        print("\nSetup cancelled.\n")