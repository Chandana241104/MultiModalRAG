"""Compatibility wrapper for CLIP vision/text embeddings."""

from __future__ import annotations

import numpy as np

from vision_embeddings import VISION_EMBEDDING_DIMENSION, VisionEmbedder


class VisionProcessor:
    def __init__(self):
        self.embedder = VisionEmbedder()

    def embed_image(self, image_path):
        return self.embedder.embed_image(image_path)

    def embed_text_for_vision(self, text):
        embedding = self.embedder.embed_text(text)
        return np.asarray(embedding, dtype=np.float32).reshape(VISION_EMBEDDING_DIMENSION)
