import os
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()


def add_conversation_namespace():
    """Add conversation_memory namespace to existing pharma-analytics index"""
    
    print("\n" + "="*80)
    print("ADDING CONVERSATION MEMORY NAMESPACE")
    print("="*80 + "\n")
    
    # Get API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not found in .env file")
        return False
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Connect to existing index
        index_name = "pharma-analytics"
        print(f"Connecting to index: {index_name}")
        
        index = pc.Index(index_name)
        
        # Check index stats
        stats = index.describe_index_stats()
        print(f"\nCurrent index stats:")
        print(f"  Total vectors: {stats['total_vector_count']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Namespaces: {list(stats.get('namespaces', {}).keys())}")
        
        # Check if conversation_memory namespace exists
        namespaces = stats.get('namespaces', {})
        if 'conversation_memory' in namespaces:
            print(f"\n✓ Namespace 'conversation_memory' already exists")
            print(f"  Vectors in namespace: {namespaces['conversation_memory']['vector_count']}")
            return True
        
        # Create namespace by upserting a dummy vector
        print(f"\nCreating namespace 'conversation_memory'...")
        
        # Upsert dummy vector (must contain at least one non-zero value)
        dummy_vector = {
            "id": "init_dummy",
            "values": [1e-8] + [0.0] * 383,   # *** FIXED HERE ***
            "metadata": {"type": "initialization"}
        }
        
        index.upsert(
            vectors=[dummy_vector],
            namespace="conversation_memory"
        )
        
        print("  Namespace created successfully")
        
        # Wait for namespace to be visible
        time.sleep(2)
        
        # Delete dummy vector
        print("  Cleaning up initialization vector...")
        index.delete(
            ids=["init_dummy"],
            namespace="conversation_memory"
        )
        
        # Verify
        time.sleep(1)
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        
        if 'conversation_memory' in namespaces:
            print(f"\n✓ Namespace 'conversation_memory' ready for use")
            print(f"  Current vector count: {namespaces['conversation_memory'].get('vector_count', 0)}")
        else:
            print(f"\n⚠ Namespace created but not yet visible (may take a moment)")
        
        print("\n" + "="*80)
        print("SETUP COMPLETE")
        print("="*80)
        print("\nThe conversation_memory namespace is ready.")
        print("You can now run the application and it will store conversation history.")
        print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nThis script adds the conversation_memory namespace to your Pinecone index.")
    print("This is a ONE-TIME operation.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        success = add_conversation_namespace()
        if success:
            print("\n✓ Success! You can now start using conversation memory.\n")
        else:
            print("\n✗ Setup failed. Please check errors above.\n")
    else:
        print("\nSetup cancelled.\n")
