# ingest_local.py 
import os
import sys
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Import modules
try:
    from ingest_utils import extract_text, extract_tables, extract_images_ocr
    from embeddings import embed_text
    from chunking import chunk_text
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "data/documents/sample.pdf")

if not os.path.exists(PDF_PATH):
    print(f"❌ PDF not found: {PDF_PATH}")
    sys.exit(1)

print(f"📄 Processing: {PDF_PATH}")

# 1. Extract content
print("\n1. Extracting content...")
text_pages = extract_text(PDF_PATH)
tables = extract_tables(PDF_PATH)
images = extract_images_ocr(PDF_PATH)

print(f"   Pages: {len(text_pages)}")
print(f"   Tables: {len(tables)}")
print(f"   Images with OCR: {len(images)}")

# 2. Prepare data
all_data = []
texts_for_embedding = []

# Text pages
for page_num, text in enumerate(text_pages, 1):
    if text.strip():
        chunks = chunk_text(text, 500)
        for chunk in chunks:
            all_data.append({
                "text": chunk,
                "page": page_num,
                "type": "text",
                "source": "text_extraction"
            })
            texts_for_embedding.append(chunk)

# Tables
for table in tables:
    if table.get("text", "").strip():
        all_data.append({
            "text": table["text"],
            "page": table.get("page", 0),
            "type": "table",
            "source": "table_extraction"
        })
        texts_for_embedding.append(table["text"])

# Images OCR
for image in images:
    if image.get("text", "").strip():
        all_data.append({
            "text": image["text"],
            "page": image.get("page", 0),
            "type": "image_ocr",
            "source": "image_ocr"
        })
        texts_for_embedding.append(image["text"])

if not all_data:
    print("❌ No data extracted!")
    sys.exit(1)

print(f"📊 Total documents: {len(all_data)}")

# 3. Create text embeddings
print("\n2. Creating embeddings...")
text_embeddings = embed_text(texts_for_embedding)

# 4. Save locally
print("\n3. Saving to local storage...")

# Create directory
storage_dir = Path("data/local_store")
storage_dir.mkdir(parents=True, exist_ok=True)

# Save documents
with open(storage_dir / "documents.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

# Save text embeddings
np.save(storage_dir / "embeddings.npy", text_embeddings)

# Save metadata
metadata = {
    "total_documents": len(all_data),
    "embedding_dimension": text_embeddings.shape[1] if len(text_embeddings.shape) > 1 else 384,
    "document_types": {
        "text": len([d for d in all_data if d["type"] == "text"]),
        "table": len([d for d in all_data if d["type"] == "table"]),
        "image_ocr": len([d for d in all_data if d["type"] == "image_ocr"])
    }
}

with open(storage_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved {len(all_data)} documents to {storage_dir}/")
print(f"✅ Embedding shape: {text_embeddings.shape}")

print("\n🎉 Ingestion complete!")
print("Now run: streamlit run app.py")