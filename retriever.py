# retriever.py
from vector_store import query_vector
from embeddings import embed_text
import ast
import hashlib
from typing import List, Dict
import numpy as np
from sentence_transformers import util
import torch

class EnhancedRetriever:
    def __init__(self):
        self.cache = {}
        
    def clean_text(self, text: str) -> str:
        """
        Robustly extracts text from dictionary-like strings.
        """
        text = text.strip()
        
        # 1. Try parsing as a Python dictionary (The most accurate method)
        if text.startswith("{") and "text" in text:
            try:
                parsed = ast.literal_eval(text)
                return parsed.get("text", text)
            except:
                # If parsing fails, fall back to string manipulation
                pass

        # 2. Fallback: Strip brackets and common keys
        clean = text.replace("{'text':", "").replace('{"text":', "")
        clean = clean.replace("'}", "").replace('"馊', "")
        
        return clean.strip()
    
    def query_expansion(self, query: str) -> List[str]:
        """Expand query for better retrieval"""
        expansions = [query]
        
        # Add variations for common question types
        query_lower = query.lower()
        
        if "what" in query_lower:
            expansions.append(query + " definition")
            expansions.append(query + " meaning")
        
        if "how" in query_lower:
            expansions.append(query + " process")
            expansions.append(query + " steps")
        
        if "why" in query_lower:
            expansions.append(query + " reason")
            expansions.append(query + " cause")
        
        # Add document-specific expansions
        if "weight" in query_lower or "score" in query_lower:
            expansions.append(query + " percentage")
            expansions.append(query + " criteria")
        
        return expansions
    
    def deduplicate_results(self, results: List) -> List[Dict]:
        """Remove duplicate or highly similar results"""
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            # Create content hash
            content = result.metadata.get('text', '') + str(result.metadata.get('page', ''))
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append({
                    'text': result.metadata.get('text', ''),
                    'page': result.metadata.get('page', 0),
                    'type': result.metadata.get('type', 'text'),
                    'score': result.score,
                    'id': result.id,
                    'original_text': result.metadata.get('text', '')  # Keep original for citations
                })
        
        return unique_results
    
    def relevance_filter(self, results: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Filter results by relevance score"""
        return [r for r in results if r.get('score', 0) > threshold]
    
    def hybrid_rerank(self, query: str, text_results: List[Dict], 
                     vision_weight: float = 0.3, text_weight: float = 0.7, 
                     top_k: int = 10) -> List[Dict]:
        """Rerank results using hybrid scores"""
        
        # Get query embedding for text similarity
        query_embedding = embed_text(query)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        if isinstance(query_embedding, list) and len(query_embedding) > 0:
            if isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
        
        # Calculate scores
        reranked = []
        
        for result in text_results:
            # Get result embedding (re-embed for consistency)
            result_text = result.get('text', '')
            if len(result_text) > 10:  # Only embed meaningful text
                try:
                    # Simple text similarity using embeddings if available
                    result_embed = embed_text(result_text)
                    if hasattr(result_embed, "tolist"):
                        result_embed = result_embed.tolist()
                    
                    # Calculate similarity
                    if isinstance(query_embedding, list) and isinstance(result_embed, list):
                        # Convert to numpy arrays for similarity calculation
                        query_arr = np.array(query_embedding)
                        result_arr = np.array(result_embed)
                        
                        # Calculate cosine similarity
                        norm_query = np.linalg.norm(query_arr)
                        norm_result = np.linalg.norm(result_arr)
                        
                        if norm_query > 0 and norm_result > 0:
                            text_sim = np.dot(query_arr, result_arr) / (norm_query * norm_result)
                        else:
                            text_sim = 0
                    else:
                        text_sim = result.get('score', 0)  # Fallback to original score
                    
                    # Combine with original score
                    original_score = result.get('score', 0)
                    combined_score = (text_sim * text_weight) + (original_score * (1 - text_weight))
                    
                    result['combined_score'] = combined_score
                    result['text_similarity'] = text_sim
                    reranked.append(result)
                    
                except Exception as e:
                    # Fallback to original scoring
                    result['combined_score'] = result.get('score', 0)
                    reranked.append(result)
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return reranked[:top_k]
    
    def retrieve_answer_chunks(self, query: str, top_k: int = 20, 
                              rerank: bool = True, deduplicate: bool = True) -> List[str]:
        """
        Enhanced retrieval with query expansion, deduplication, and reranking.
        Returns clean text chunks for LLM consumption.
        """
        # Query expansion
        expanded_queries = self.query_expansion(query)
        
        all_matches = []
        for exp_query in expanded_queries[:2]:  # Limit to 2 expansions for performance
            query_embedding = embed_text(exp_query)
            
            # Flatten embedding
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()
            if isinstance(query_embedding, list) and len(query_embedding) > 0:
                if isinstance(query_embedding[0], list):
                    query_embedding = query_embedding[0]
            
            try:
                matches = query_vector(query_embedding, top_k=min(top_k * 2, 50))
                all_matches.extend(matches)
            except Exception as e:
                print(f"Error querying Pinecone for '{exp_query}': {e}")
        
        if not all_matches:
            return []
        
        # Process and deduplicate results
        processed_results = []
        for match in all_matches:
            if match and match.metadata:
                raw_text = match.metadata.get("text", "")
                page = match.metadata.get("page", "Unknown")
                source_type = match.metadata.get("type", "text")
                
                # Apply cleaning
                final_text = self.clean_text(raw_text)
                
                if len(final_text) > 10:
                    processed_results.append({
                        'text': final_text,
                        'page': page,
                        'type': source_type,
                        'score': match.score,
                        'id': match.id,
                        'original_text': raw_text
                    })
        
        # Deduplicate
        if deduplicate:
            processed_results = self.deduplicate_results_for_matches(processed_results)
        
        # Filter by relevance
        processed_results = self.relevance_filter(processed_results, threshold=0.3)
        
        # Rerank
        if rerank and len(processed_results) > 1:
            processed_results = self.hybrid_rerank(
                query, processed_results, top_k=top_k
            )
        else:
            # Sort by original score
            processed_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            processed_results = processed_results[:top_k]
        
        # Format chunks for LLM with citation markers
        chunks = []
        for i, result in enumerate(processed_results):
            text = result.get('text', '')
            if text.strip():
                chunks.append(f"[Page {result.get('page', '?')} | {result.get('type', 'text')}] {text}")
        
        return chunks
    
    def deduplicate_results_for_matches(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results with similarity threshold"""
        if len(results) <= 1:
            return results
        
        unique_results = []
        seen_texts = set()
        
        for result in results:
            text = result.get('text', '').lower().strip()
            if len(text) < 20:  # Very short texts can't be meaningfully compared
                unique_results.append(result)
                continue
            
            # Check for near-duplicates
            is_duplicate = False
            for seen_text in seen_texts:
                # Simple similarity check (can be enhanced with embeddings)
                if self.text_similarity(text, seen_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(text)
                unique_results.append(result)
        
        return unique_results
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (can be replaced with embedding similarity)"""
        # Simple Jaccard similarity of words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

# Initialize singleton instance
_retriever_instance = None

def get_retriever():
    """Get singleton retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EnhancedRetriever()
    return _retriever_instance

# Backward compatibility function
def retrieve_answer_chunks(query, top_k=20):
    """Legacy function for backward compatibility"""
    retriever = get_retriever()
    return retriever.retrieve_answer_chunks(query, top_k=top_k)