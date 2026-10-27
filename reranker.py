"""Lightweight cross-modal reranking utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class CrossModalReranker:
    """Combine text and image retrieval results into one ranked list."""

    def cosine_similarity(self, vec1, vec2) -> float:
        if vec1 is None or vec2 is None:
            return 0.0

        arr1 = np.asarray(vec1, dtype=np.float32).flatten()
        arr2 = np.asarray(vec2, dtype=np.float32).flatten()

        if arr1.size == 0 or arr2.size == 0:
            return 0.0

        if arr1.size != arr2.size:
            size = min(arr1.size, arr2.size)
            arr1 = arr1[:size]
            arr2 = arr2[:size]

        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def hybrid_rerank(
        self,
        query: str,
        text_results: List[Dict],
        image_results: Optional[List[Dict]] = None,
        top_k: int = 10,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
    ) -> List[Dict]:
        """Merge text and image results and sort by a normalized hybrid score."""
        merged: List[Dict] = []

        for rank, result in enumerate(text_results or []):
            item = dict(result)
            base_score = float(item.get("score", 1.0 / (rank + 1)))
            item["type"] = item.get("type", "text")
            item["hybrid_score"] = text_weight * base_score
            merged.append(item)

        for rank, result in enumerate(image_results or []):
            item = dict(result)
            base_score = float(item.get("score", 1.0 / (rank + 1)))
            item["type"] = item.get("type", "image")
            item["hybrid_score"] = image_weight * base_score
            merged.append(item)

        merged.sort(key=lambda item: item.get("hybrid_score", item.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def rerank(
        self,
        query: str,
        results: List[Dict],
        modality_weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """Rerank a single mixed result list by modality-weighted scores."""
        modality_weights = modality_weights or {"text": 0.7, "table": 0.6, "image": 0.5, "image_ocr": 0.5}
        reranked = []

        for rank, result in enumerate(results or []):
            item = dict(result)
            modality = str(item.get("type", "text")).lower()
            base_score = float(item.get("score", 1.0 / (rank + 1)))
            item["cross_modal_score"] = base_score * modality_weights.get(modality, 0.5)
            reranked.append(item)

        reranked.sort(key=lambda item: item.get("cross_modal_score", 0.0), reverse=True)
        return reranked[:top_k]


_reranker_instance = None


def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossModalReranker()
    return _reranker_instance


def create_cross_modal_reranker():
    return get_reranker()
