import torch
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor

class VisionProcessor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.numpy()
    
    def embed_text_for_vision(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.numpy()