
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from ingest_utils import extract_images_ocr
from vision_embeddings import get_vision_embedder

load_dotenv()

PDF_PATH = Path("data/documents/sample.pdf")
OUTPUT_DIR = Path("data/local_store")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

print("Extracting and saving PDF images...")
images = extract_images_ocr(str(PDF_PATH))

if not images:
    raise RuntimeError("No embedded images were extracted from the PDF.")

embedder = get_vision_embedder()

if not embedder.enable_clip:
    raise RuntimeError(
        "CLIP is disabled. Add ENABLE_CLIP_EMBEDDINGS=1 to .env "
        "and restart Python."
    )

records = []
embeddings = []

for i, image in enumerate(images, start=1):
    image_path = Path(image["image_path"])

    print(f"Embedding image {i}/{len(images)}: {image_path}")

    vector = embedder.embed_image(str(image_path))
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    if vector.shape != (512,):
        raise ValueError(
            f"Expected a 512-dimensional CLIP vector, got {vector.shape}"
        )

    embeddings.append(vector)
    records.append({
        "image_id": f"image_{i}",
        "image_path": str(image_path),
        "page": image["page"],
        "image_index": image["image_index"],
        "ocr_text": image.get("text", ""),
        "embedding_dimension": 512,
    })

embedding_matrix = np.stack(embeddings)

np.save(OUTPUT_DIR / "vision_embeddings.npy", embedding_matrix)

with open(
    OUTPUT_DIR / "vision_documents.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(records, file, indent=2, ensure_ascii=False)

print("\nVision indexing complete!")
print(f"Images indexed: {len(records)}")
print(f"Embedding shape: {embedding_matrix.shape}")
print(f"Metadata saved: {OUTPUT_DIR / 'vision_documents.json'}")
print(f"Embeddings saved: {OUTPUT_DIR / 'vision_embeddings.npy'}")
