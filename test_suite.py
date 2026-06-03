# test_suite.py
import unittest
import sys
import os
sys.path.append('.')

class TestMultiModalRAG(unittest.TestCase):
    
    def setUp(self):
        from embeddings import embed_text
        from vision_embeddings import VisionEmbedder
        
        self.text_embedder = embed_text
        self.vision_embedder = VisionEmbedder()
    
    def test_text_embedding(self):
        """Test text embedding generation"""
        embedding = self.text_embedder("test text")
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 384)  # MiniLM dimension
    
    def test_vision_embedding(self):
        """Test vision embedding generation"""
        # Create a test image
        from PIL import Image
        import numpy as np
        
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save('test_image.png')
        
        embedding = self.vision_embedder.embed_image('test_image.png')
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 512)  # CLIP dimension
        
        # Cleanup
        os.remove('test_image.png')
    
    def test_reranker(self):
        """Test cross-modal reranking"""
        from reranker import CrossModalReranker
        
        reranker = CrossModalReranker()
        
        # Mock results
        text_results = [{'text': 'test', 'embedding': [0.1]*384}]
        image_results = [{'vision_embedding': [0.1]*512}]
        
        reranked = reranker.hybrid_rerank("query", text_results, image_results, top_k=5)
        self.assertIsInstance(reranked, list)
    
    def test_memory_manager(self):
        """Test memory management"""
        from memory_manager import MemoryManager
        
        manager = MemoryManager()
        self.assertTrue(manager.check_memory())
    
    def test_evaluator(self):
        """Test evaluation metrics"""
        from evaluator import MultiModalEvaluator
        
        evaluator = MultiModalEvaluator()
        self.assertIn('text_queries', evaluator.test_queries)

if __name__ == '__main__':
    unittest.main()