import numpy as np
from embeddings import embed_text
from vision_processor import VisionProcessor

class UnifiedEmbedder:
    def __init__(self):
        self.text_embedder = embed_text
        self.vision_processor = VisionProcessor()
    
    def embed(self, content, content_type="text"):
        if content_type == "text":
            return self.text_embedder(content)
        elif content_type == "image":
            return self.vision_processor.embed_image(content)
        elif content_type == "table":
            # Convert table to descriptive text
            table_desc = f"Table showing: {content}"
            return self.text_embedder(table_desc)
        else:
            return np.zeros(384)  # Default dimension