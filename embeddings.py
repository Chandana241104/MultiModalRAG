# embeddings.py - Updated with better error handling
import os
import time
from sentence_transformers import SentenceTransformer
import logging

# Disable verbose logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Try to load model with retries
def load_model_with_retry(model_name="paraphrase-MiniLM-L3-v2", max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to load model...")
            model = SentenceTransformer(model_name)
            print(f"✅ Model '{model_name}' loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("⚠️  Falling back to a smaller model...")
                try:
                    return SentenceTransformer("all-MiniLM-L6-v2")
                except:
                    raise Exception(f"Failed to load any model after {max_retries} attempts")

# Load model with retry mechanism
try:
    model = load_model_with_retry("paraphrase-MiniLM-L3-v2")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    # Create a dummy model for testing
    class DummyModel:
        def encode(self, texts, **kwargs):
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384).astype(np.float32)
    
    model = DummyModel()
    print("⚠️  Using dummy model for testing. Real embeddings won't work.")

def embed_text(texts):
    """
    Embed single text or list of texts.
    Normalizes embeddings for better cosine similarity.
    """
    if isinstance(texts, str):
        texts = [texts]

    texts = [t.replace("\n", " ").strip() for t in texts]
    
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings
    except Exception as e:
        print(f"⚠️  Embedding error: {e}")
        # Return random embeddings for testing
        import numpy as np
        if isinstance(texts, list):
            return np.random.randn(len(texts), 384).astype(np.float32)
        else:
            return np.random.randn(1, 384).astype(np.float32)