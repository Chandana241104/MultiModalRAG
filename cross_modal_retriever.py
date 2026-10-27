from dotenv import load_dotenv 
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from embeddings import embed_text
from vision_embeddings import get_vision_embedder, VISION_EMBEDDING_DIMENSION

load_dotenv()

class CrossModalRetriever:
    def __init__(self, data_dir: str = "data/local_store"):
        self.data_dir = Path(data_dir)

        self.documents = []
        self.text_embeddings = None

        self.vision_documents = []
        self.vision_embeddings = None

        self.vision_embedder = get_vision_embedder()
        self.load_data()

    def load_data(self):
        """Load text and image collections independently."""
        try:
            documents_path = self.data_dir / "documents.json"
            embeddings_path = self.data_dir / "embeddings.npy"

            if documents_path.exists() and embeddings_path.exists():
                with open(documents_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)

                self.text_embeddings = np.load(embeddings_path)

                if len(self.documents) != len(self.text_embeddings):
                    raise ValueError(
                        "Text document count does not match text embedding count."
                    )

                if self.text_embeddings.ndim != 2:
                    raise ValueError("Text embeddings must be a 2D matrix.")

                print(f"Loaded {len(self.documents)} text documents")
                print(f"Text embedding shape: {self.text_embeddings.shape}")
            else:
                print("Text collection files not found.")

            vision_docs_path = self.data_dir / "vision_documents.json"
            vision_emb_path = self.data_dir / "vision_embeddings.npy"

            if vision_docs_path.exists() and vision_emb_path.exists():
                with open(vision_docs_path, "r", encoding="utf-8") as f:
                    self.vision_documents = json.load(f)

                self.vision_embeddings = np.load(vision_emb_path)

                if len(self.vision_documents) != len(self.vision_embeddings):
                    raise ValueError(
                        "Image metadata count does not match image embedding count."
                    )

                if (
                    self.vision_embeddings.ndim != 2
                    or self.vision_embeddings.shape[1]
                    != VISION_EMBEDDING_DIMENSION
                ):
                    raise ValueError(
                        "Expected image embeddings with shape (N, 512)."
                    )

                print(f"Loaded {len(self.vision_documents)} image records")
                print(
                    f"Vision embedding shape: {self.vision_embeddings.shape}"
                )
            else:
                print("Vision collection files not found.")

        except Exception as exc:
            raise RuntimeError(f"Failed to load retrieval data: {exc}") from exc

    @staticmethod
    def cosine_similarity(vec1, vec2) -> float:
        """Calculate cosine similarity without silently changing dimensions."""
        a = np.asarray(vec1, dtype=np.float32).reshape(-1)
        b = np.asarray(vec2, dtype=np.float32).reshape(-1)

        if a.shape != b.shape:
            raise ValueError(
                f"Embedding dimension mismatch: {a.shape} vs {b.shape}"
            )

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _require_real_clip(self):
        """Prevent meaningless fallback vectors from being used for search."""
        if not self.vision_embedder._ensure_model():
            raise RuntimeError(
                "Real CLIP is unavailable. Check ENABLE_CLIP_EMBEDDINGS=1, "
                "the transformers installation, and model loading. "
                "Image search cannot work reliably with fallback vectors."
            )

    def retrieve_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant text chunks using the text embedding model."""
        if not self.documents or self.text_embeddings is None:
            return []

        query_vector = np.asarray(embed_text(query), dtype=np.float32).reshape(-1)

        if query_vector.shape[0] != self.text_embeddings.shape[1]:
            raise ValueError(
                f"Query text embedding has {query_vector.shape[0]} dimensions, "
                f"but stored text embeddings have {self.text_embeddings.shape[1]}."
            )

        scored = []
        for idx, vector in enumerate(self.text_embeddings):
            score = self.cosine_similarity(query_vector, vector)
            scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "document": self.documents[idx],
                "score": score,
                "type": self.documents[idx].get("type", "text"),
                "is_chart": self.documents[idx].get("type") == "chart",
            }
            for idx, score in scored[:top_k]
        ]

    def retrieve_visual_content(
        self, query: str, top_k: int = 5
    ) -> List[Dict]:
        """Retrieve saved PDF images using CLIP text-image similarity."""
        if not self.vision_documents or self.vision_embeddings is None:
            return []

        self._require_real_clip()

        query_vector = self.vision_embedder.embed_text(query)

        if len(query_vector) != self.vision_embeddings.shape[1]:
            raise ValueError("Query CLIP vector dimension does not match image vectors.")

        scored = []

        for idx, image_record in enumerate(self.vision_documents):
            score = self.cosine_similarity(
                query_vector, self.vision_embeddings[idx]
            )

            # Only return image files that still exist.
            image_path = Path(image_record["image_path"])
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            if image_path.exists():
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            record = self.vision_documents[idx]
            results.append({
                "type": "image",
                "image_id": record.get("image_id", f"image_{idx + 1}"),
                "image_path": record["image_path"],
                "page": record.get("page"),
                "image_index": record.get("image_index"),
                "ocr_text": record.get("ocr_text", ""),
                "score": score,
            })

        return results

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 10,
        text_weight: float = 0.7,
        vision_weight: float = 0.3,
    ) -> List[Dict]:
        """Return a ranked mixture of text chunks and actual PDF images."""
        text_results = self.retrieve_text(query, top_k=top_k)
        image_results = self.retrieve_visual_content(query, top_k=top_k)

        # Normalize weights so their total is 1.
        total_weight = text_weight + vision_weight
        if total_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive.")

        text_weight /= total_weight
        vision_weight /= total_weight

        combined = []

        for item in text_results:
            combined.append({
                **item,
                "combined_score": text_weight * max(0.0, item["score"]),
            })

        for item in image_results:
            combined.append({
                **item,
                "combined_score": vision_weight * max(0.0, item["score"]),
                "is_chart": False,
            })

        combined.sort(
            key=lambda item: item["combined_score"],
            reverse=True,
        )

        return combined[:top_k]


_cross_modal_retriever = None


def get_cross_modal_retriever():
    """Return a reusable retriever instance."""
    global _cross_modal_retriever

    if _cross_modal_retriever is None:
        _cross_modal_retriever = CrossModalRetriever()

    return _cross_modal_retriever
