"""Per-tenant hybrid index: BM25 (k1=1.25, b=0.75) + FAISS HNSW (efSearch=64). Built from the chunk store."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import faiss
import numpy as np

from .tokenize import tokenize


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    tenant_id: str
    text: str
    source: str = "vault"


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.25, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(query_tokens):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, f in plist:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1.0))
                scores[i] += idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


@dataclass
class TenantIndex:
    chunks: list[Chunk]
    bm25: BM25
    faiss_index: faiss.Index
    version: int = 1
    by_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray, version: int = 1) -> "TenantIndex":
        bm = BM25([tokenize(c.text) for c in chunks])
        dim = embeddings.shape[1]
        ix = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction, ix.hnsw.efSearch = 80, 64
        if len(chunks):
            ix.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return cls(chunks, bm, ix, version, {c.chunk_id: i for i, c in enumerate(chunks)})

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(tokenize(query), top_k)

    def search_dense(self, q_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.faiss_index.ntotal == 0:
            return []
        k = min(top_k, self.faiss_index.ntotal)
        d, i = self.faiss_index.search(np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32), k)
        return [(int(j), float(s)) for j, s in zip(i[0], d[0]) if j >= 0]


class IndexRegistry:
    """tenant -> index. `set` is an atomic swap (blue/green promotion)."""

    def __init__(self) -> None:
        self._ix: dict[str, TenantIndex] = {}

    def get(self, tenant: str) -> TenantIndex | None:
        return self._ix.get(tenant)

    def set(self, tenant: str, index: TenantIndex) -> None:
        old = self._ix.get(tenant)
        index.version = (old.version + 1) if old else 1
        self._ix[tenant] = index
