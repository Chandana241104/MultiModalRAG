import os
import unittest
from unittest.mock import patch

original_api_key = os.environ.get("PINECONE_API_KEY")
original_index_name = os.environ.get("PINECONE_INDEX_NAME")
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_INDEX_NAME"] = ""

import vector_store
from vector_store import MockIndex

if original_api_key is None:
    os.environ.pop("PINECONE_API_KEY", None)
else:
    os.environ["PINECONE_API_KEY"] = original_api_key
if original_index_name is None:
    os.environ.pop("PINECONE_INDEX_NAME", None)
else:
    os.environ["PINECONE_INDEX_NAME"] = original_index_name


class SessionScopedVectorTests(unittest.TestCase):
    def setUp(self):
        self.index = MockIndex.__new__(MockIndex)
        self.index.vectors = []
        self.index.upsert(vectors=[
            {
                "id": "a-old",
                "values": [1.0, 0.0],
                "metadata": {
                    "session_id": "session-a",
                    "document_id": "old",
                    "text": "old document",
                },
            },
            {
                "id": "b-current",
                "values": [1.0, 0.0],
                "metadata": {
                    "session_id": "session-b",
                    "document_id": "current",
                    "text": "other user's document",
                },
            },
            {
                "id": "a-current",
                "values": [1.0, 0.0],
                "metadata": {
                    "session_id": "session-a",
                    "document_id": "current",
                    "text": "new document",
                },
            },
        ])
        self.index_patcher = patch.object(vector_store, "index", self.index)
        self.index_patcher.start()

    def tearDown(self):
        self.index_patcher.stop()

    def test_scoped_delete_and_query_preserve_other_sessions(self):
        vector_store.delete_vectors({
            "session_id": {"$eq": "session-a"},
            "document_id": {"$eq": "old"},
        })

        current_user_results = vector_store.query_vector(
            [1.0, 0.0],
            metadata_filter={
                "session_id": {"$eq": "session-a"},
                "document_id": {"$eq": "current"},
            },
        )
        other_user_results = vector_store.query_vector(
            [1.0, 0.0],
            metadata_filter={
                "session_id": {"$eq": "session-b"},
                "document_id": {"$eq": "current"},
            },
        )

        self.assertEqual([match.id for match in current_user_results], ["a-current"])
        self.assertEqual([match.id for match in other_user_results], ["b-current"])

    def test_scoped_query_never_returns_unscoped_fallback_chunks(self):
        self.index.vectors = []

        matches = vector_store.query_vector(
            [1.0, 0.0],
            metadata_filter={"session_id": {"$eq": "unknown-session"}},
        )

        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
