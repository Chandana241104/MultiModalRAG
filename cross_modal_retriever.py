# cross_modal_retriever.py
import numpy as np
import json
from typing import List, Dict, Tuple
from pathlib import Path
from vision_embeddings import get_vision_embedder

class CrossModalRetriever:
    def __init__(self, data_dir: str = "data/local_store"):
        """Initialize cross-modal retriever"""
        self.data_dir = Path(data_dir)
        self.documents = []
        self.text_embeddings = None
        self.vision_embeddings = None
        self.chart_data = []
        self.vision_embedder = get_vision_embedder()
        
        self.load_data()
    
    def load_data(self):
        """Load text and vision embeddings"""
        try:
            # Load text documents
            with open(self.data_dir / "documents.json", "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            # Load text embeddings
            self.text_embeddings = np.load(self.data_dir / "embeddings.npy")
            
            # Load vision embeddings if available
            vision_emb_path = self.data_dir / "vision_embeddings.npy"
            if vision_emb_path.exists():
                self.vision_embeddings = np.load(vision_emb_path)
            
            # Extract chart data
            self.chart_data = [
                {"index": i, **doc} 
                for i, doc in enumerate(self.documents) 
                if doc.get("type") == "chart"
            ]
            
            print(f"✅ Loaded {len(self.documents)} documents")
            print(f"✅ Loaded {len(self.chart_data)} charts")
            if self.vision_embeddings is not None:
                print(f"✅ Loaded {len(self.vision_embeddings)} vision embeddings")
                
        except Exception as e:
            print(f"⚠️  Error loading data: {e}")
    
    def hybrid_retrieve(self, query: str, top_k: int = 10, 
                       text_weight: float = 0.7, vision_weight: float = 0.3) -> List[Dict]:
        """Hybrid retrieval combining text and vision similarity"""
        
        if not self.documents or self.text_embeddings is None:
            return []
        
        # Get query embedding for text
        from embeddings import embed_text
        query_text_embedding = embed_text(query)
        if hasattr(query_text_embedding, "tolist"):
            query_text_embedding = query_text_embedding.tolist()
        
        # Get query embedding for vision (using CLIP)
        query_vision_embedding = self.vision_embedder.embed_text(query)
        
        # Calculate text similarities
        text_similarities = []
        for i, doc_emb in enumerate(self.text_embeddings):
            if hasattr(doc_emb, "tolist"):
                doc_emb = doc_emb.tolist()
            
            # Cosine similarity
            similarity = self.cosine_similarity(query_text_embedding, doc_emb)
            text_similarities.append((i, similarity))
        
        # Calculate vision similarities for charts
        vision_similarities = []
        if self.vision_embeddings is not None and len(self.vision_embeddings) > 0:
            for chart in self.chart_data:
                idx = chart["index"]
                if idx < len(self.vision_embeddings):
                    vision_emb = self.vision_embeddings[idx]
                    similarity = self.cosine_similarity(query_vision_embedding, vision_emb)
                    vision_similarities.append((idx, similarity))
        
        # Combine scores using RRF (Reciprocal Rank Fusion)
        combined_results = self.reciprocal_rank_fusion(
            text_similarities, vision_similarities, k=60
        )
        
        # Get top results
        results = []
        for idx, score in combined_results[:top_k]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "score": score,
                    "type": doc.get("type", "text"),
                    "is_chart": doc.get("type") == "chart"
                })
        
        return results
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def reciprocal_rank_fusion(self, list1: List[Tuple], list2: List[Tuple], k: int = 60) -> List[Tuple]:
        """Implement Reciprocal Rank Fusion for combining multiple rankings"""
        
        scores = {}
        
        # Rank list1
        list1.sort(key=lambda x: x[1], reverse=True)
        for rank, (idx, score) in enumerate(list1, 1):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        
        # Rank list2
        list2.sort(key=lambda x: x[1], reverse=True)
        for rank, (idx, score) in enumerate(list2, 1):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items
    
    def retrieve_visual_content(self, query: str, top_k: int = 5) -> List[Dict]:
        """Specifically retrieve visual content (charts, images)"""
        if not self.chart_data:
            return []
        
        # Use vision embedding for query
        query_embedding = self.vision_embedder.embed_text(query)
        
        similarities = []
        for chart in self.chart_data:
            idx = chart["index"]
            if idx < len(self.vision_embeddings):
                chart_emb = self.vision_embeddings[idx]
                similarity = self.cosine_similarity(query_embedding, chart_emb)
                
                # Boost score if query contains visual terms
                visual_terms = ["chart", "graph", "figure", "image", "diagram", "plot", "visual"]
                if any(term in query.lower() for term in visual_terms):
                    similarity *= 1.5
                
                similarities.append((idx, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "text": doc.get("text", ""),
                    "page": doc.get("page", 0),
                    "type": "chart",
                    "chart_type": doc.get("chart_type", "unknown"),
                    "description": doc.get("description", ""),
                    "score": score,
                    "metadata": json.loads(doc.get("metadata", "{}"))
                })
        
        return results

# Singleton instance
_cross_modal_retriever = None

def get_cross_modal_retriever():
    """Get singleton cross-modal retriever"""
    global _cross_modal_retriever
    if _cross_modal_retriever is None:
        _cross_modal_retriever = CrossModalRetriever()
    return _cross_modal_retriever