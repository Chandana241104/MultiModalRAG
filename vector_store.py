# vector_store.py - Updated for Serverless
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key or not index_name:
    raise ValueError("Missing Pinecone API Key or Index Name in .env")

pc = Pinecone(api_key=api_key)

# Try to list indexes
try:
    indexes = pc.list_indexes()
    existing_indexes = [idx.name for idx in indexes]
    print(f"Found {len(existing_indexes)} existing indexes")
except Exception as e:
    print(f"Error listing indexes: {e}")
    existing_indexes = []

if index_name not in existing_indexes:
    print(f"Creating serverless index: {index_name}...")
    try:
        # Use a different region that might have quota
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-west-2"  # Try different region
            )
        )
        print(f"✅ Index {index_name} created in us-west-2")
        time.sleep(30)  # Serverless indexes take longer
    except Exception as e:
        print(f"❌ Failed to create index in us-west-2: {e}")
        
        # Try another region
        try:
            print("Trying eu-west-1 region...")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", 
                    region="eu-west-1"  # Another region
                )
            )
            print(f"✅ Index {index_name} created in eu-west-1")
            time.sleep(30)
        except Exception as e2:
            print(f"❌ Failed to create index in any region: {e2}")
            print("⚠️  You may need to:")
            print("   1. Wait for quota reset (free tier daily limits)")
            print("   2. Delete unused indexes at app.pinecone.io")
            print("   3. Upgrade to paid plan")
            print("   4. Use mock mode for testing")
            
            # Fall back to mock index
            class MockIndex:
                def query(self, vector, top_k=20, include_metadata=True):
                    class Match:
                        def __init__(self, i):
                            import random
                            self.score = 0.9 - (i * 0.1)
                            self.metadata = {
                                "text": f"Mock document chunk {i+1} for testing without Pinecone.",
                                "page": random.randint(1, 3),
                                "type": "text"
                            }
                    
                    class Result:
                        def __init__(self):
                            self.matches = [Match(i) for i in range(min(top_k, 5))]
                    
                    return Result()
            
            index = MockIndex()
            print("⚠️  Using mock index for testing")
else:
    print(f"✅ Index {index_name} already exists")
    index = pc.Index(index_name)

def query_vector(query_embedding, top_k=20):
    # Handle mock index
    if hasattr(index, '__class__') and index.__class__.__name__ == 'MockIndex':
        return index.query(vector=query_embedding, top_k=top_k).matches
    
    # Real Pinecone index
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    
    if isinstance(query_embedding, list) and query_embedding:
        if isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
    
    try:
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return response.matches
    except Exception as e:
        print(f"Query error: {e}")
        return []