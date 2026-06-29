import os
import pytest
from app.rag.vector_store import VectorStore


def test_vector_store_search_returns_list():
    index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        pytest.skip("FAISS index not built yet -- run scripts/ingest.py first")
    vs = VectorStore()
    vs.load(index_path)
    results = vs.search([0.0] * 1536, top_k=3)
    assert isinstance(results, list)


def test_vector_store_score_ordering():
    index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        pytest.skip("FAISS index not built yet -- run scripts/ingest.py first")
    vs = VectorStore()
    vs.load(index_path)
    results = vs.search([0.0] * 1536, top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
