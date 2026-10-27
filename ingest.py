# ingest.py
import os
import sys
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings("ignore")

from ingest_utils import (
    extract_text,
    extract_tables,
    extract_images_ocr,
    extract_chart_metadata_from_image
)
from embeddings import embed_text
from vector_store import index
from chunking import chunk_text

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "data/documents/sample.pdf")

if not os.path.exists(PDF_PATH):
    print(f"PDF file not found at: {PDF_PATH}")
    sys.exit(1)

print(f"Starting ingestion for: {PDF_PATH}")

# -------------------------
# 1. Extraction Phase
# -------------------------
print("\n--- Phase 1: Extraction ---")

# A. Text
print("Extracting text...")
text_pages = extract_text(PDF_PATH)
print(f"   Found {len(text_pages)} pages.")

# B. Tables
print("Extracting tables...")
tables = extract_tables(PDF_PATH)
print(f"   Found {len(tables)} tables.")

# C. Images/OCR
print("Extracting images and OCR...")
images = extract_images_ocr(PDF_PATH)
print(f"   Found {len(images)} images with text.")

all_chunks = []
all_metadata = []

# -------------------------
# 2. Chunking Phase
# -------------------------
print("\n--- Phase 2: Chunking ---")

# Process Text Pages
for page_num, page_text in enumerate(text_pages, start=1):
    if not page_text or not page_text.strip():
        print(f"   Page {page_num} is empty (scanned?). Relying on OCR/Tables.")
        continue
        
    chunks = chunk_text(page_text, chunk_size=500)
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            all_chunks.append(chunk)
            all_metadata.append({
                "text": chunk,
                "source": "pdf",
                "page": page_num,
                "type": "text",
                "chunk_id": i
            })
print(f"   Processed text chunks: {len([m for m in all_metadata if m['type']=='text'])}")

# Process Tables
for table_id, table_info in enumerate(tables, start=1):
    t_text = table_info.get("text", "")
    page_num = table_info.get("page", 0)
    
    if t_text.strip():
        # Keep tables largely intact or split if huge
        table_chunks = chunk_text(t_text, chunk_size=400)
        for i, chunk in enumerate(table_chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "text": chunk,
                "source": "pdf",
                "page": page_num,
                "type": "table",
                "table_id": table_id
            })

# Process Images
for img_info in images:
    ocr_text = img_info.get("text", "")
    page_num = img_info.get("page", 0)
    
    if ocr_text.strip():
        all_chunks.append(ocr_text)
        all_metadata.append({
            "text": ocr_text,
            "source": "pdf",
            "page": page_num,
            "type": "image_ocr"
        })

if not all_chunks:
    print("Critical error: No data extracted. Check PDF or OCR setup.")
    sys.exit(1)

print(f"Total chunks prepared: {len(all_chunks)}")

# -------------------------
# 3. Embedding Phase
# -------------------------
print("\n--- Phase 3: Embedding ---")
vectors = []
BATCH_SIZE = 32

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i+BATCH_SIZE]
    print(f"   Embedding batch {i // BATCH_SIZE + 1}...")
    
    try:
        embeddings = embed_text(batch)
        for emb in embeddings:
            if hasattr(emb, "tolist"):
                vectors.append(emb.tolist())
            else:
                vectors.append(emb)
    except Exception as e:
        print(f"Error embedding batch: {e}")

# In ingest.py, update the upsert section (around line 150):

# -------------------------
# 4. Upsert Phase
# -------------------------
print("\n--- Phase 4: Upserting to Pinecone ---")
UPSERT_BATCH = 100

for i in range(0, len(vectors), UPSERT_BATCH):
    batch_vecs = vectors[i:i+UPSERT_BATCH]
    batch_meta = all_metadata[i:i+UPSERT_BATCH]
    
    pinecone_vectors = []
    for j, (vec, meta) in enumerate(zip(batch_vecs, batch_meta)):
        # Create a unique ID
        uid = f"doc_p{meta.get('page')}_{meta.get('type')}_{i+j}"
        
        # Convert vector to list if needed
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        
        # Pinecone 8.0.0 format
        pinecone_vectors.append({
            "id": uid,
            "values": vec,
            "metadata": meta
        })
        
    try:
        index.upsert(vectors=pinecone_vectors)
        print(f"   Upserted vectors {i} to {i+len(batch_vecs)}")
    except Exception as e:
        print(f"Upsert failed: {e}")
        # Try with tuples format
        try:
            tuple_vectors = [(v["id"], v["values"], v["metadata"]) for v in pinecone_vectors]
            index.upsert(vectors=tuple_vectors)
            print("   Upserted with tuple format")
        except Exception as e2:
            print(f"   Tuple format also failed: {e2}")
print("\nIngestion complete!")
