import json
import os
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class SearchResult:
    text: str
    source: str
    chunk_id: str
    score: float
    cadres: list = None  # cadre codes this chunk is relevant to (["all"] if general)


class VectorStore:
    def __init__(self) -> None:
        self._index = None
        self._metadata: list[dict] = []

    def load(self, index_dir: str) -> None:
        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "metadata.json")
        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def search(self, embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("Vector store not loaded. Call load() at startup.")
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                text=meta["text"],
                source=meta["source"],
                chunk_id=meta["chunk_id"],
                score=float(score),
                cadres=meta.get("cadres", ["all"]),
            ))
        return results


# Module-level singleton — loaded once at startup, shared across all handlers
vector_store = VectorStore()
