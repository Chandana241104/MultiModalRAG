# retriever_local.py - Local retriever with multi-modal support
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import ast
import warnings
warnings.filterwarnings('ignore')

class LocalRetriever:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.metadata = {}
        self.load_data()
    
    def load_data(self):
        """Load data from local storage"""
        storage_dir = Path("data/local_store")
        
        if not storage_dir.exists():
            print("⚠️  No local data found. Run ingest_local.py first.")
            return
        
        try:
            # Load documents
            with open(storage_dir / "documents.json", "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            # Load embeddings
            self.embeddings = np.load(storage_dir / "embeddings.npy")
            
            # Load metadata
            with open(storage_dir / "metadata.json", "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            print(f"✅ Loaded {len(self.documents)} documents from local storage")
            print(f"✅ Embeddings shape: {self.embeddings.shape}")
            
            # Show document types
            if 'document_types' in self.metadata:
                for doc_type, count in self.metadata['document_types'].items():
                    print(f"   - {doc_type}: {count}")
                    
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("⚠️  Run ingest_local.py first to create the data")
    
    def clean_text(self, text: str) -> str:
        """Clean text from dictionary strings"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Try to parse as dict
        if text.startswith("{") and "text" in text:
            try:
                parsed = ast.literal_eval(text)
                return parsed.get("text", text)
            except:
                pass
        
        # Simple cleanup
        clean = text.replace("{'text':", "").replace('{"text":', "")
        clean = clean.replace("'}", "").replace('"馊', "")
        return clean.strip()
    
    def retrieve_answer_chunks(self, query: str, top_k: int = 20, **kwargs) -> List[str]:
        """Retrieve relevant chunks using cosine similarity"""
        from embeddings import embed_text
        
        if not self.documents or self.embeddings is None:
            print("⚠️  No data loaded")
            return []
        
        # Embed query
        query_embedding = embed_text(query)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        
        # Flatten if needed
        if isinstance(query_embedding, list) and query_embedding:
            if isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
        
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        similarities = []
        for i, doc_vec in enumerate(self.embeddings):
            if hasattr(doc_vec, "tolist"):
                doc_vec = doc_vec.tolist()
            
            vec = np.array(doc_vec)
            
            # Cosine similarity
            norm_vec = np.linalg.norm(vec)
            norm_query = np.linalg.norm(query_vec)
            
            if norm_vec > 0 and norm_query > 0:
                similarity = np.dot(vec, query_vec) / (norm_vec * norm_query)
            else:
                similarity = 0
            
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top chunks
        results = []
        for idx, score in similarities[:top_k]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                text = self.clean_text(doc["text"])
                
                if text and len(text) > 10:
                    # Format the chunk with source info
                    source_info = f"[Page {doc.get('page', '?')} | {doc.get('type', 'text').upper()}]"
                    results.append(f"{source_info} {text[:500]}...")
        
        return results
    
    def get_document_statistics(self) -> Dict:
        """Get statistics about the loaded documents"""
        return self.metadata.get('document_types', {})

# Global instance
_retriever_instance = None

def get_retriever():
    """Get singleton retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LocalRetriever()
    return _retriever_instance

def retrieve_answer_chunks(query, top_k=20):
    """For backward compatibility"""
    retriever = get_retriever()
    return retriever.retrieve_answer_chunks(query, top_k=top_k)