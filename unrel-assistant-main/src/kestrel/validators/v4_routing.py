"""V4 Routing & retrieval (FR-04, FR-05, FR-06; KPI-04, 05, 07, 11, 12, 13, 23).

Quick profile: fixed samples (25 golden, 10 unanswerable, 64 triage rows) with the deterministic HashEmbedder and
LexicalReranker. The retrieval numbers are therefore indicative, not the production models' numbers (design §2.7:
real-model golden runs take minutes and are not part of quick). Relevance is document-level: a citation is relevant
when its source_uri is the golden item's evidence document (the HTTP response carries chunk ids, not chunk text).
"""
from __future__ import annotations

import math
import time


from ..triage import FixedTriage
from .base import Env, Fault, Outcome, Validator
from .fixtures import golden_set, jsonl, sse_events

GOLD_STRIDE, UNANS_STRIDE, TRIAGE_STRIDE = 10, 15, 5


class _AlwaysZeroReranker:
    """Seeded fault: the reranker scores every candidate 0, so nothing clears the abstention threshold."""

    async def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0 for _ in texts]


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]


class V4Routing(Validator):
    id = "V4"
    title = "Routing & retrieval"
    features = "FR-04, FR-05, FR-06"
    kpis = "KPI-04, KPI-05, KPI-07, KPI-11, KPI-12, KPI-13, KPI-23"
    faults = (
        Fault("triage_always_direct", "triage routes every query to DIRECT_ANSWER",
              build={"triage": FixedTriage("DIRECT_ANSWER")}),
        Fault("reranker_scores_zero", "reranker returns 0 for every candidate",
              build={"reranker": _AlwaysZeroReranker()}),
    )
    skip_reasons = (
        "KPI-10 recall@pool: the candidate pool is not exposed by the public API",
        "KPI-08 and KPI-09 stage latencies: quick runs use the in-process fakes, so they are not measured",
    )

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        sb = await env.sandbox(tenant, fault)
        try:
            # ---- triage: the orchestrator's own classifier, called directly (no pipeline side effects)
            rows = jsonl("triage.jsonl")[::TRIAGE_STRIDE]
            hits, tri_ms = 0, []
            for row in rows:
                t0 = time.perf_counter()
                route = sb.orch.triage.classify(row["text"]).route
                tri_ms.append((time.perf_counter() - t0) * 1000)
                hits += route == row["label"]
            out.metric("KPI-04 triage routing accuracy", hits / len(rows), 0.97, ">=", len(rows))
            out.metric("KPI-05 triage latency p95 (ms)", _p95(tri_ms), 25.0, "<=", len(rows))

            # ---- answerable golden set through the public API
            gold = [g for g in golden_set() if g["answerable"]][::GOLD_STRIDE]
            mrr, ndcg, p4, bm25_ms = [], [], [], []
            for i, g in enumerate(gold):
                doc = g["evidence"][0]["doc"]
                fin, meta = await self._ask(sb, tenant, f"user_v4g{i}", g["question"])
                srcs = [c["source_uri"] for c in (fin or {}).get("citations", [])]
                out.check(len(srcs) <= 4, f"golden {g['id']} cited {len(srcs)} chunks (max 4)")
                rel = [s == doc for s in srcs[:4]]
                p4.append(sum(rel) / 4)
                first = next((k for k, r in enumerate(rel, 1) if r), None)
                mrr.append(1 / first if first else 0.0)
                ndcg.append(1.0 / math.log2(first + 1) if first else 0.0)   # one relevant document: ideal DCG = 1
                if "mcp.rag.bm25" in (meta.get("timings_ms") or {}):
                    bm25_ms.append(float(meta["timings_ms"]["mcp.rag.bm25"]))
            n = len(gold)
            out.metric("KPI-11 precision@4 (doc-level)", sum(p4) / n, 0.88, ">=", n)
            out.metric("KPI-12 MRR@4 (doc-level)", sum(mrr) / n, 0.85, ">=", n)
            out.metric("KPI-13 nDCG@4 (doc-level)", sum(ndcg) / n, 0.86, ">=", n)
            if bm25_ms:
                out.metric("KPI-07 BM25 stage latency p95 (ms)", _p95(bm25_ms), 12.0, "<=", len(bm25_ms))

            # ---- unanswerable set: must abstain (or be refused as out of scope)
            unans = jsonl("unanswerable.jsonl")[::UNANS_STRIDE]
            abst = 0
            for i, u in enumerate(unans):
                fin, _ = await self._ask(sb, tenant, f"user_v4u{i}", u["question"])
                good = (fin or {}).get("status") in ("ABSTAINED", "REFUSED")
                abst += out.check(good, f"unanswerable {u['id']} was answered (status {(fin or {}).get('status')})",
                                  severity="major")
            out.metric("KPI-23 abstention accuracy", abst / len(unans), 0.98, ">=", len(unans))
            return out
        finally:
            await sb.aclose()

    @staticmethod
    async def _ask(sb, tenant: str, user: str, query: str) -> tuple[dict | None, dict]:
        r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant, "Accept": "text/event-stream"},
                                 json={"tenant_id": tenant, "user_id": user, "query": query})
        ev = sse_events(r.text)
        return ev.get("final"), ev.get("meta", {})
