"""Vision embedding helpers backed by CLIP with a deterministic fallback."""

from __future__ import annotations

import hashlib
import os
import warnings

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

VISION_EMBEDDING_DIMENSION = 512


def _deterministic_vector(value: str, dimension: int = VISION_EMBEDDING_DIMENSION) -> np.ndarray:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class VisionEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_clip = os.getenv("ENABLE_CLIP_EMBEDDINGS", "0").lower() in {"1", "true", "yes"}
        self._load_attempted = False

    def _ensure_model(self) -> bool:
        if not self.enable_clip:
            return False
        if self.model is not None and self.processor is not None:
            return True
        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            from transformers import CLIPModel, CLIPProcessor

            print(f"Loading CLIP model on {self.device}...")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            print(f"CLIP model '{self.model_name}' loaded successfully.")
            return True
        except ImportError:
            print("transformers is not installed. Using dummy vision embeddings.")
        except Exception as exc:
            print(f"Could not load CLIP model: {exc}")
            print("Using dummy vision embeddings.")
        return False

    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate a 512-dimensional embedding for an image path."""
        if not self._ensure_model():
            return _deterministic_vector(f"image:{image_path}")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            embeddings = image_features.detach().cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(embeddings)
            return embeddings / norm if norm else embeddings
        except Exception as exc:
            print(f"Image embedding error for {image_path}: {exc}")
            return np.zeros(VISION_EMBEDDING_DIMENSION, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a CLIP text embedding for cross-modal retrieval."""
        if not self._ensure_model():
            return _deterministic_vector(f"text:{text}")

        try:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            embeddings = text_features.detach().cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(embeddings)
            return embeddings / norm if norm else embeddings
        except Exception as exc:
            print(f"Text embedding error: {exc}")
            return np.zeros(VISION_EMBEDDING_DIMENSION, dtype=np.float32)

    def image_text_similarity(self, image_path: str, text: str) -> float:
        """Calculate similarity between an image and text."""
        if not self._ensure_model():
            image_embedding = self.embed_image(image_path)
            text_embedding = self.embed_text(text)
            return float(np.dot(image_embedding, text_embedding))

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                inputs = self.processor(
                    text=[text],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                return float(torch.sigmoid(outputs.logits_per_image).item())
        except Exception as exc:
            print(f"Similarity calculation error: {exc}")
            return 0.0


_vision_embedder = None


def get_vision_embedder():
    global _vision_embedder
    if _vision_embedder is None:
        _vision_embedder = VisionEmbedder()
    return _vision_embedder
