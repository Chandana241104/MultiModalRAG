# vision_embeddings.py - Enhanced version
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class VisionEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model for vision embeddings"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖼️  Loading CLIP model on {self.device}...")
            
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            print(f"✅ CLIP model '{model_name}' loaded successfully")
            
        except ImportError:
            print("⚠️  transformers not installed. Install with: pip install transformers")
            self.model = None
        except Exception as e:
            print(f"⚠️  Could not load CLIP model: {e}")
            print("   Using dummy vision embeddings")
            self.model = None
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate embeddings for images"""
        if self.model is None:
            # Return dummy embeddings (512 dimensions like CLIP)
            return np.random.randn(512).astype(np.float32)
        
        try:
            image = Image.open(image_path)
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            embeddings = image_features.cpu().numpy().flatten()
            embeddings = embeddings / np.linalg.norm(embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️  Image embedding error for {image_path}: {e}")
            return np.zeros(512)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embeddings using CLIP for cross-modal retrieval"""
        if self.model is None:
            return np.random.randn(512).astype(np.float32)
        
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            embeddings = text_features.cpu().numpy().flatten()
            embeddings = embeddings / np.linalg.norm(embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️  Text embedding error: {e}")
            return np.zeros(512)
    
    def image_text_similarity(self, image_path: str, text: str) -> float:
        """Calculate similarity between image and text"""
        if self.model is None:
            return 0.5
        
        try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = torch.sigmoid(logits_per_image).item()
            
            return similarity
            
        except Exception as e:
            print(f"⚠️  Similarity calculation error: {e}")
            return 0.0

# Singleton instance
_vision_embedder = None

def get_vision_embedder():
    """Get singleton vision embedder instance"""
    global _vision_embedder
    if _vision_embedder is None:
        _vision_embedder = VisionEmbedder()
    return _vision_embedder