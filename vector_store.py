"""Pinecone vector-store adapter with a local mock fallback."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None


load_dotenv()


class MockMatch:
    def __init__(self, i: int, metadata: dict | None = None, score: float | None = None, item_id: str | None = None):
        self.id = item_id or f"mock-{i}"
        self.score = 0.9 - (i * 0.08) if score is None else score
        self.metadata = metadata or {
            "text": f"Mock document chunk {i + 1} for testing without Pinecone.",
            "page": (i % 3) + 1,
            "type": "text",
        }


class MockResult:
    def __init__(self, matches):
        self.matches = matches


class MockIndex:
    def __init__(self):
        self.vectors = []
        self._load_mock_data()

    def _load_mock_data(self):
        """Load mock data from local storage if available"""
        try:
            import json
            from pathlib import Path
            
            storage_dir = Path("data/local_store")
            if storage_dir.exists():
                with open(storage_dir / "documents.json", "r", encoding="utf-8") as f:
                    documents = json.load(f)
                
                # Load embeddings
                import numpy as np
                embeddings = np.load(storage_dir / "embeddings.npy")
                
                for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()
                    self.vectors.append({
                        "id": f"doc_{i}",
                        "values": emb,
                        "metadata": doc
                    })
                print(f"Loaded {len(self.vectors)} vectors from local storage")
        except Exception as e:
            print(f"Could not load mock data: {e}")

    def upsert(self, vectors):
        self.vectors.extend(vectors or [])
        return {"upserted_count": len(vectors or [])}

    def query(self, vector, top_k=20, include_metadata=True):
        if self.vectors:
            matches = []
            for i, item in enumerate(self.vectors):
                if isinstance(item, dict):
                    metadata = item.get("metadata", {})
                    item_id = item.get("id", f"mock-{i}")
                    # Calculate similarity if vector is provided
                    if vector and "values" in item:
                        try:
                            import numpy as np
                            vec1 = np.array(vector)
                            vec2 = np.array(item["values"])
                            norm1 = np.linalg.norm(vec1)
                            norm2 = np.linalg.norm(vec2)
                            if norm1 > 0 and norm2 > 0:
                                score = float(np.dot(vec1, vec2) / (norm1 * norm2))
                            else:
                                score = 0.5 - (i * 0.03)
                        except:
                            score = 0.5 - (i * 0.03)
                    else:
                        score = 0.5 - (i * 0.03)
                    
                    match = MockMatch(i, metadata=metadata, score=score, item_id=item_id)
                    matches.append(match)
            
            # Sort by score and return top_k
            matches.sort(key=lambda x: x.score, reverse=True)
            return MockResult(matches[:top_k])

        # Fallback mock data
        return MockResult([MockMatch(i) for i in range(min(top_k, 5))])


def _create_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        print("Pinecone is not configured. Using mock vector index.")
        return MockIndex()

    if Pinecone is None or ServerlessSpec is None:
        print("pinecone package is not installed. Using mock vector index.")
        return MockIndex()

    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        existing_indexes = [idx.name for idx in indexes]
        print(f"Found {len(existing_indexes)} existing Pinecone indexes.")

        if index_name not in existing_indexes:
            print(f"Creating serverless index: {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(30)

        print(f"Using Pinecone index: {index_name}")
        return pc.Index(index_name)
    except Exception as exc:
        print(f"Pinecone unavailable: {exc}")
        print("Using mock vector index.")
        return MockIndex()


index = _create_index()


def query_vector(query_embedding, top_k=20):
    """Query the vector store and return matches list."""
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    if isinstance(query_embedding, list) and query_embedding and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    try:
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        matches = getattr(response, "matches", [])
        if matches is None:
            return []
        return matches
    except Exception as exc:
        print(f"Query error: {exc}")
        return []