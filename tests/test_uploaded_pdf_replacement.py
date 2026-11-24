import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

original_api_key = os.environ.get("PINECONE_API_KEY")
original_index_name = os.environ.get("PINECONE_INDEX_NAME")
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_INDEX_NAME"] = ""

ingest_stubs = types.ModuleType("ingest_utils")
ingest_stubs.extract_images_ocr = None
ingest_stubs.extract_tables = None
ingest_stubs.extract_text = None
with patch.dict(sys.modules, {"ingest_utils": ingest_stubs}):
    import upload_manager
sys.modules.pop("upload_manager", None)

if original_api_key is None:
    os.environ.pop("PINECONE_API_KEY", None)
else:
    os.environ["PINECONE_API_KEY"] = original_api_key
if original_index_name is None:
    os.environ.pop("PINECONE_INDEX_NAME", None)
else:
    os.environ["PINECONE_INDEX_NAME"] = original_index_name

from vector_store import MockIndex


class UploadedPdfReplacementTests(unittest.TestCase):
    def test_replacement_deletes_only_previous_pdf_for_current_session(self):
        session_id = uuid.uuid4().hex
        previous_document_id = uuid.uuid4().hex
        other_session_id = uuid.uuid4().hex
        self.working_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.working_directory.cleanup)
        upload_root = Path(self.working_directory.name) / "uploads"
        previous_assets = upload_root / session_id / previous_document_id
        previous_assets.mkdir(parents=True)
        (previous_assets / "old-image.png").write_bytes(b"old")

        index = MockIndex.__new__(MockIndex)
        index.vectors = [
            {
                "id": "previous",
                "values": [1.0] * 384,
                "metadata": {
                    "session_id": session_id,
                    "document_id": previous_document_id,
                    "text": "old uploaded PDF",
                },
            },
            {
                "id": "other-user",
                "values": [1.0] * 384,
                "metadata": {
                    "session_id": other_session_id,
                    "document_id": "other-document",
                    "text": "other user's PDF",
                },
            },
        ]

        def delete_matching(metadata_filter):
            return index.delete(filter=metadata_filter)

        with (
            patch.object(upload_manager, "UPLOAD_ROOT", upload_root),
            patch.object(upload_manager, "index", index),
            patch.object(upload_manager, "delete_vectors", side_effect=delete_matching),
            patch.object(
                upload_manager,
                "_collect_chunks",
                return_value=([{"text": "new PDF contents", "page": 1, "type": "text"}], []),
            ),
            patch.object(upload_manager, "extract_images_ocr", return_value=[]),
            patch.object(
                upload_manager,
                "embed_text",
                return_value=np.ones((1, 384), dtype=np.float32),
            ),
            patch.object(upload_manager, "_save_vision_index", return_value=False),
        ):
            result = upload_manager.ingest_uploaded_pdf(
                b"test pdf bytes",
                session_id=session_id,
                filename="new.pdf",
                previous_document_id=previous_document_id,
            )

        self.assertFalse(previous_assets.exists())
        self.assertEqual(len(index.vectors), 2)
        self.assertEqual(
            {vector["metadata"]["session_id"] for vector in index.vectors},
            {session_id, other_session_id},
        )
        new_vector = next(
            vector for vector in index.vectors
            if vector["metadata"]["document_id"] == result["document_id"]
        )
        self.assertEqual(new_vector["metadata"]["text"], "new PDF contents")
        self.assertEqual(result["filename"], "new.pdf")


if __name__ == "__main__":
    unittest.main()
