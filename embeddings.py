"""Text embedding helpers for the RAG pipeline."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Iterable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised only when dependency is absent
    SentenceTransformer = None


logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

EMBEDDING_DIMENSION = 384
_model = None


class DummyModel:
    """Deterministic fallback used when sentence-transformers is unavailable."""

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([_deterministic_vector(text) for text in texts]).astype(np.float32)


def _deterministic_vector(text: str, dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def load_model_with_retry(model_name: str = "paraphrase-MiniLM-L3-v2", max_retries: int = 3):
    if SentenceTransformer is None:
        print("sentence-transformers is not installed. Using dummy embeddings.")
        return DummyModel()

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to load model...")
            model = SentenceTransformer(model_name)
            print(f"Model '{model_name}' loaded successfully.")
            return model
        except Exception as exc:
            print(f"Attempt {attempt + 1} failed: {exc}")
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

    try:
        fallback_name = "all-MiniLM-L6-v2"
        print(f"Falling back to '{fallback_name}'.")
        return SentenceTransformer(fallback_name)
    except Exception as exc:
        print(f"Could not load any embedding model: {exc}")
        print("Using dummy embeddings. Retrieval quality will be limited.")
        return DummyModel()


def get_model():
    global _model
    if _model is None:
        _model = load_model_with_retry()
    return _model


def _clean_texts(texts: Iterable[str]) -> list[str]:
    return [str(text).replace("\n", " ").strip() for text in texts]


def embed_text(texts):
    """
    Embed a single string or a list of strings.

    Returns a 1D vector for a single string and a 2D array for a list, matching
    the expectations of the retrievers and tests.
    """
    single_input = isinstance(texts, str)
    batch = [texts] if single_input else list(texts)
    batch = _clean_texts(batch)

    try:
        embeddings = get_model().encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
    except Exception as exc:
        print(f"Embedding error: {exc}")
        embeddings = np.vstack([_deterministic_vector(text) for text in batch]).astype(np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    return embeddings[0] if single_input else embeddings
