import numpy as np
from typing import List, Dict, Optional
import logging

# FIXED: Added proper import with try/except
try:
    from unified_embeddings import UnifiedEmbedder
except ImportError:
    UnifiedEmbedder = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossModalReranker:
    def __init__(self):
        """Initialize cross-modal reranker with unified embedder"""
        try:
            from unified_embeddings import UnifiedEmbedder
            self.unified_embedder = UnifiedEmbedder()
            logger.info("CrossModalReranker initialized successfully")
        except Exception as e:
            logger.warning(f"UnifiedEmbedder unavailable: {e}")
            # Create a dummy embedder for fallback
            self.unified_embedder = None
    
    def cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Convert to numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Flatten arrays
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Ensure same shape
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    
    def embed_query_by_modality(self, query: str, modality: str):
        """Get query embedding for specific modality"""
        if self.unified_embedder is None:
            # Return random embedding as fallback
            if modality == "text":
                return np.random.randn(384)
            else:
                return np.random.randn(512)
        
        if modality == "text":
            return self.unified_embedder.embed(query, "text")
        elif modality == "image":
            # Use vision processor if available
            if hasattr(self.unified_embedder, 'vision_processor'):
                return self.unified_embedder.vision_processor.embed_text_for_vision(query)
            else:
                return self.unified_embedder.embed(query, "text")  # Fallback
        elif modality == "table":
            return self.unified_embedder.embed(f"table data about {query}", "text")
        else:
            return self.unified_embedder.embed(query, "text")
    
    def rerank(self, query: str, results: List[Dict], modality_weights: Optional[Dict] = None, top_k: int = 10) -> List[Dict]:
        """Rerank results using cross-modal similarity"""
        if not results:
            return []
        
        if modality_weights is None:
            modality_weights = {"text": 0.4, "table": 0.3, "image": 0.3}
        
        # Get query embedding for each modality
        query_embeddings = {}
        for modality in modality_weights.keys():
            query_embeddings[modality] = self.embed_query_by_modality(query, modality)
        
        # Calculate combined scores
        reranked = []
        for result in results:
            modality = result.get("type", "text")
            content = result.get("content", "")
            
            # Get similarity score
            if modality in query_embeddings and content:
                try:
                    # Get result embedding
                    if self.unified_embedder:
                        result_embedding = self.unified_embedder.embed(content, modality)
                    else:
                        result_embedding = np.random.randn(384)
                    
                    # Calculate similarity
                    similarity = self.cosine_similarity(
                        query_embeddings[modality], 
                        result_embedding
                    )
                    
                    # Calculate combined score
                    original_score = result.get("score", 0.5)
                    weight = modality_weights.get(modality, 0.3)
                    combined_score = original_score * 0.7 + similarity * weight * 0.3
                    
                    # Update result
                    result["cross_modal_score"] = combined_score
                    result["similarity"] = similarity
                    result["original_score"] = original_score
                    reranked.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    # Keep original score
                    result["cross_modal_score"] = result.get("score", 0.5)
                    reranked.append(result)
            else:
                # Keep original score for unhandled modalities
                result["cross_modal_score"] = result.get("score", 0.5)
                reranked.append(result)
        
        # Sort by cross-modal score
        reranked.sort(key=lambda x: x.get("cross_modal_score", 0), reverse=True)
        
        # Return top_k results
        return reranked[:top_k]
    
    def get_modality_breakdown(self, results: List[Dict]) -> Dict:
        """Analyze modality distribution in results"""
        modalities = {}
        for result in results:
            modality = result.get("type", "text")
            modalities[modality] = modalities.get(modality, 0) + 1
        
        total = len(results)
        if total > 0:
            for modality in modalities:
                modalities[modality] = modalities[modality] / total * 100
        
        return modalities

    def hybrid_rerank(self, query: str, text_results: List[Dict],
                      image_results: Optional[List[Dict]] = None, top_k: int = 10) -> List[Dict]:
        """Backward-compatible helper for separate text and image result lists."""
        combined = []
        for rank, result in enumerate(text_results or []):
            item = dict(result)
            item["type"] = item.get("type", "text")
            item["score"] = float(item.get("score", 1.0 / (rank + 1)))
            combined.append(item)

        for rank, result in enumerate(image_results or []):
            item = dict(result)
            item["type"] = item.get("type", "image")
            item["score"] = float(item.get("score", 1.0 / (rank + 1)))
            combined.append(item)

        return self.rerank(query, combined, top_k=top_k)

# ========== SINGLETON PATTERN ==========

_reranker_instance = None

def get_reranker():
    """Get singleton reranker instance"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossModalReranker()
    return _reranker_instance

def create_cross_modal_reranker():
    """Alternative function name for backward compatibility"""
    return get_reranker()