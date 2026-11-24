"""Ingest uploaded PDFs into per-session vector and visual indexes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from chunking import chunk_text
from embeddings import embed_text
from ingest_utils import extract_images_ocr, extract_tables, extract_text
from vector_store import delete_vectors, index


UPLOAD_ROOT = Path("data/uploads")
UPSERT_BATCH_SIZE = 100


def _document_filter(session_id: str, document_id: str) -> dict:
    return {
        "session_id": {"$eq": session_id},
        "document_id": {"$eq": document_id},
    }


def _asset_directory(session_id: str, document_id: str) -> Path:
    session_key = uuid.UUID(session_id).hex
    document_key = uuid.UUID(document_id).hex
    return UPLOAD_ROOT / session_key / document_key


def _remove_document_assets(session_id: str, document_id: str) -> None:
    target = _asset_directory(session_id, document_id).resolve()
    session_directory = (UPLOAD_ROOT / uuid.UUID(session_id).hex).resolve()
    if target.parent != session_directory:
        raise ValueError("Upload asset path is outside the session directory.")
    if target.exists():
        shutil.rmtree(target)


def _collect_chunks(pdf_path: str) -> tuple[list[dict], list[dict]]:
    chunks = []
    text_pages = extract_text(pdf_path)
    tables = extract_tables(pdf_path)

    for page_number, page_text in enumerate(text_pages, start=1):
        if page_text and page_text.strip():
            for chunk_number, text in enumerate(chunk_text(page_text, chunk_size=500)):
                if text.strip():
                    chunks.append({
                        "text": text,
                        "page": page_number,
                        "type": "text",
                        "chunk_id": chunk_number,
                    })

    for table_number, table in enumerate(tables, start=1):
        table_text = table.get("text", "")
        if table_text.strip():
            for text in chunk_text(table_text, chunk_size=400):
                if text.strip():
                    chunks.append({
                        "text": text,
                        "page": table.get("page", 0),
                        "type": "table",
                        "table_id": table_number,
                    })

    return chunks, tables


def _save_vision_index(image_records: list[dict], asset_directory: Path) -> bool:
    if not image_records:
        return False

    from vision_embeddings import get_vision_embedder

    vision_embedder = get_vision_embedder()
    if not vision_embedder.enable_clip or not vision_embedder._ensure_model():
        return False

    records = []
    vectors = []
    for image_number, image_record in enumerate(image_records, start=1):
        vector = np.asarray(
            vision_embedder.embed_image(image_record["image_path"]),
            dtype=np.float32,
        ).reshape(-1)
        if vector.shape != (512,) or not np.linalg.norm(vector):
            raise RuntimeError(
                f"Could not create a valid CLIP embedding for image {image_number}."
            )

        vectors.append(vector)
        records.append({
            "image_id": f"image_{image_number}",
            "image_path": image_record["image_path"],
            "page": image_record["page"],
            "image_index": image_record["image_index"],
            "ocr_text": image_record.get("text", ""),
            "embedding_dimension": 512,
        })

    np.save(asset_directory / "vision_embeddings.npy", np.stack(vectors))
    with open(asset_directory / "vision_documents.json", "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    return True


def ingest_uploaded_pdf(
    pdf_bytes: bytes,
    session_id: str,
    filename: str,
    previous_document_id: str | None = None,
) -> dict:
    """Index a PDF for one session, then remove that session's previous PDF."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    session_id = uuid.UUID(session_id).hex
    document_id = uuid.uuid4().hex
    asset_directory = _asset_directory(session_id, document_id)
    asset_directory.mkdir(parents=True, exist_ok=False)

    temporary_pdf_path = None
    vectors_started = False
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_pdf:
            temporary_pdf.write(pdf_bytes)
            temporary_pdf_path = temporary_pdf.name

        chunks, _ = _collect_chunks(temporary_pdf_path)
        image_directory = asset_directory / "images"
        image_records = extract_images_ocr(
            temporary_pdf_path,
            output_dir=image_directory,
        )
        for image_record in image_records:
            ocr_text = image_record.get("text", "")
            if ocr_text.strip():
                chunks.append({
                    "text": ocr_text,
                    "page": image_record.get("page", 0),
                    "type": "image_ocr",
                })

        if not chunks:
            raise ValueError("No readable text, tables, or image text was found in the PDF.")

        vectors = []
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            embeddings = np.asarray(
                embed_text([chunk["text"] for chunk in batch]),
                dtype=np.float32,
            )
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if embeddings.shape != (len(batch), 384):
                raise ValueError(
                    "The generated text embeddings do not match the extracted PDF content."
                )

            for offset, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                metadata = {
                    **chunk,
                    "source": "uploaded_pdf",
                    "session_id": session_id,
                    "document_id": document_id,
                }
                vector_number = start + offset
                vectors.append({
                    "id": f"{document_id}-{vector_number}",
                    "values": embedding.tolist(),
                    "metadata": metadata,
                })

        vision_indexed = _save_vision_index(image_records, asset_directory)

        os.unlink(temporary_pdf_path)
        temporary_pdf_path = None
        vectors_started = True
        for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
            index.upsert(vectors=vectors[start:start + UPSERT_BATCH_SIZE])

        if previous_document_id:
            delete_vectors(_document_filter(session_id, previous_document_id))
    except Exception as exc:
        if vectors_started:
            try:
                delete_vectors(_document_filter(session_id, document_id))
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"PDF indexing failed and the new vectors could not be rolled back: "
                    f"{cleanup_error}"
                ) from exc
        try:
            _remove_document_assets(session_id, document_id)
        except Exception as cleanup_error:
            raise RuntimeError(
                f"PDF indexing failed and temporary upload files could not be removed: "
                f"{cleanup_error}"
            ) from exc
        raise
    finally:
        if temporary_pdf_path and os.path.exists(temporary_pdf_path):
            os.unlink(temporary_pdf_path)

    cleanup_warning = None
    if previous_document_id:
        try:
            _remove_document_assets(session_id, previous_document_id)
        except Exception as exc:
            cleanup_warning = f"Previous PDF image files could not be removed: {exc}"

    return {
        "document_id": document_id,
        "filename": filename,
        "asset_directory": str(asset_directory),
        "chunk_count": len(chunks),
        "image_count": len(image_records),
        "vision_indexed": vision_indexed,
        "cleanup_warning": cleanup_warning,
    }
