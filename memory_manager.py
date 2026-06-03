# memory_manager.py
import psutil
import gc
import threading
from queue import Queue
import time

class MemoryManager:
    def __init__(self, memory_threshold_mb: int = 1024):
        self.threshold = memory_threshold_mb
        self.queue = Queue()
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.threshold:
            self.queue.put("cleanup")
            return False
        
        return True
    
    def _cleanup_worker(self):
        """Background thread for memory cleanup"""
        while True:
            if not self.queue.empty():
                self.queue.get()
                self.force_cleanup()
            time.sleep(5)
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        
        # Clear caches if needed
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def process_large_pdf(self, pdf_path: str, batch_size: int = 10):
        """Process PDF in batches to manage memory"""
        import fitz
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for start_page in range(0, total_pages, batch_size):
            end_page = min(start_page + batch_size, total_pages)
            
            # Check memory before processing batch
            if not self.check_memory():
                print(f"Memory high, pausing...")
                time.sleep(2)
                self.force_cleanup()
            
            # Process batch
            batch_texts = []
            for page_num in range(start_page, end_page):
                page = doc.load_page(page_num)
                text = page.get_text()
                batch_texts.append(text)
            
            yield batch_texts
            
            # Cleanup after batch
            del batch_texts
            gc.collect()
        
        doc.close()