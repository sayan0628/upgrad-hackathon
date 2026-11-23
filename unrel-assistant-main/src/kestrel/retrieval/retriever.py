"""3-stage hybrid retrieval (design.md ADR-001): BM25 ∥ dense -> RRF -> cross-encoder, with abstention."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from .index import Chunk, IndexRegistry
from .rerank import Reranker


@dataclass
class ScoredChunk:
    chunk: Chunk
    rerank_score: float
    rrf_rank: int | None = None
    bm25: float | None = None
    cosine: float | None = None


@dataclass
class Retrieval:
    chunks: list[ScoredChunk]
    top_score: float
    abstain: bool
    stage_ms: dict[str, float] = field(default_factory=dict)
    # chunk_ids of the fused candidate pool actually sent to the reranker, RRF order (empty if the pre-rerank
    # gate abstained). Diagnostic only: used by tools/retrieval_metrics.py for KPI-10 (recall@pool).
    pool_ids: list[str] = field(default_factory=list)


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, int]]:
    """Reciprocal rank fusion; returns (doc, rank) best-first, deduplicated."""
    s: dict[int, float] = {}
    for lst in rank_lists:
        for r, d in enumerate(lst, 1):
            s[d] = s.get(d, 0.0) + 1.0 / (k + r)
    return [(d, i + 1) for i, (d, _) in enumerate(sorted(s.items(), key=lambda x: (-x[1], x[0])))]


class HybridRetriever:
    def __init__(self, registry: IndexRegistry, reranker: Reranker, *, bm25_top_k: int = 50, dense_top_k: int = 50,
                 rrf_k: int = 60, pool: int = 8, rerank_k: int = 4, threshold: float = 0.42,
                 prefilter_floor: float | None = 0.20):
        self.registry, self.reranker = registry, reranker
        self.bm25_top_k, self.dense_top_k, self.rrf_k, self.pool, self.rerank_k = bm25_top_k, dense_top_k, rrf_k, pool, rerank_k
        self.threshold = threshold      # server-side config only (never caller-settable)
        # Pre-rerank abstain gate: no lexical hit AND best cosine below the floor => skip the (expensive) reranker.
        # Conservative and provisional: calibrate on the golden set (a false abstain here is a recall loss).
        self.prefilter_floor = prefilter_floor

    async def retrieve(self, tenant: str, query: str, q_emb: np.ndarray) -> Retrieval:
        ix = self.registry.get(tenant)
        if ix is None or not ix.chunks:
            return Retrieval([], 0.0, True, {})
        ms: dict[str, float] = {}
        t0 = time.perf_counter()

        async def timed(name, fn, *a):
            t = time.perf_counter()
            r = await asyncio.to_thread(fn, *a)
            ms[name] = (time.perf_counter() - t) * 1000
            return r

        bm, dn = await asyncio.gather(timed("bm25", ix.search_bm25, query, self.bm25_top_k),
                                      timed("dense", ix.search_dense, q_emb, self.dense_top_k))
        if self.prefilter_floor is not None and not bm and (not dn or max(s for _, s in dn) < self.prefilter_floor):
            ms["retrieval_total"] = (time.perf_counter() - t0) * 1000
            ms["prefilter_abstain"] = 1.0
            return Retrieval([], 0.0, True, ms)
        t = time.perf_counter()
        bm_s, dn_s = dict(bm), dict(dn)
        fused = rrf([[d for d, _ in bm], [d for d, _ in dn]], self.rrf_k)[: self.pool]
        ms["rrf"] = (time.perf_counter() - t) * 1000
        if not fused:
            return Retrieval([], 0.0, True, ms)
        t = time.perf_counter()
        scores = await self.reranker.score(query, [ix.chunks[d].text for d, _ in fused])
        ms["rerank"] = (time.perf_counter() - t) * 1000
        scored = sorted((ScoredChunk(ix.chunks[d], sc, rank, bm_s.get(d), dn_s.get(d))
                         for (d, rank), sc in zip(fused, scores)), key=lambda c: -c.rerank_score)[: self.rerank_k]
        top = scored[0].rerank_score if scored else 0.0
        ms["retrieval_total"] = (time.perf_counter() - t0) * 1000
        return Retrieval(scored, top, top < self.threshold, ms, [ix.chunks[d].chunk_id for d, _ in fused])
