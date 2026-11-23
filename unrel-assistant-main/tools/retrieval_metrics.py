"""KPI-10..13 on the insurance golden set, measured through the REAL retrieval stack (real embedder, BM25 + FAISS HNSW,
RRF, cross-encoder reranker, tenant index built by the same ingestion the runtime uses).

    python tools/retrieval_metrics.py --reranker torch           # production default pool (Settings.rerank_pool = 4)
    python tools/retrieval_metrics.py --reranker onnx-int8 --pool 4 --max-length 128

Gold labels (chunk level): for each answerable golden question, every evidence {doc, sentence} must be contained (after
whitespace normalisation) in chunks of that doc in the tenant index. Gold chunk ids = those chunks. Labels are written
to bench_results/retrieval_gold.jsonl. Evidence sentences that land in exactly one chunk are the norm; sentences that
sit in the 10-word overlap of two adjacent windows land in two chunks. The primary metrics use the union of matching
chunks; the `strict` block drops every question with a multi-chunk evidence sentence (exactly-one labels only).

Metrics (binary relevance, per question, macro-averaged over answerable questions with >= 1 gold chunk):
  KPI-10 recall@pool      |gold & candidate pool sent to the reranker| / |gold|      threshold >= 0.92
  KPI-11 precision@4      |gold & top-4| / |top-4 returned|                             threshold >= 0.88
  KPI-12 MRR@4            1 / rank of the first gold chunk in top-4 (0 if none)        threshold >= 0.85
  KPI-13 nDCG@4           DCG@4 / IDCG@4, rel in {0,1}                                 threshold >= 0.86
Questions where the pre-rerank gate abstains have an empty pool and top-4, so they score 0 on every metric.
Diagnostics: recall at pool 4/8/16/20/40/50 on the fused list (no reranker needed), and precision@4 with the denominator
min(4, |gold|), because with one gold chunk the spec's denominator caps precision@4 at 0.25.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.config import Settings  # noqa: E402
from kestrel.embed import SentenceTransformerEmbedder  # noqa: E402
from kestrel.retrieval import CrossEncoderReranker, HybridRetriever, IndexRegistry, ingest_texts  # noqa: E402
from kestrel.retrieval.rerank import OnnxCrossEncoderReranker  # noqa: E402
from kestrel.runtime import DEMO_TENANT, GOLDEN, configure_torch, load_corpus  # noqa: E402
from kestrel.stores import InMemoryStateStore  # noqa: E402

BENCH = ROOT / "bench_results"
THRESH = {"recall_at_pool": 0.92, "precision_at4": 0.88, "mrr_at4": 0.85, "ndcg_at4": 0.86}


class _NullReranker:
    """Zero scores: produces the real fused candidate pool without paying for the cross-encoder (diagnostics)."""

    async def score(self, query, texts):
        return [0.0] * len(texts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_gold(ix, golden_rows):
    """Returns (labels_for_file, gold_union {qid: set}, gold_strict {qid: set}) for answerable questions."""
    labels, union, strict = [], {}, {}
    for r in golden_rows:
        if not r.get("answerable") or not r.get("evidence"):
            continue
        ev_out, ids_union, ambiguous = [], set(), False
        for e in r["evidence"]:
            s = _norm(e["sentence"])
            ids = [c.chunk_id for c in ix.chunks if c.source == e["doc"] and s in _norm(c.text)]
            ev_out.append({"doc": e["doc"], "sentence": e["sentence"], "chunk_ids": ids, "n_chunks": len(ids)})
            if len(ids) > 1:
                ambiguous = True
            ids_union |= set(ids)
        union[r["id"]] = ids_union
        if not ambiguous and ids_union:
            strict[r["id"]] = ids_union
        labels.append({"id": r["id"], "question": r["question"], "gold_chunk_ids": sorted(ids_union),
                       "exactly_one_chunk_per_evidence": not ambiguous, "evidence": ev_out})
    return labels, union, strict


def score_question(gold: set, pool_ids: list[str], top_ids: list[str], k: int = 4) -> dict:
    top = top_ids[:k]
    hits = [c in gold for c in top]
    rec = len(gold & set(pool_ids)) / len(gold)
    prec = sum(hits) / len(top) if top else 0.0
    mrr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"recall_at_pool": rec, "precision_at4": prec, "mrr_at4": mrr, "ndcg_at4": dcg / idcg,
            "precision_at4_capped_diag": sum(hits) / min(k, len(gold))}


def aggregate(rows: list[dict]) -> dict:
    keys = rows[0].keys() if rows else []
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}


async def main(a):
    configure_torch()
    settings = Settings(openrouter_api_key=None, openrouter_base_url="", generation_model="offline/none",
                        fallback_model=None, judge_model=None, rerank_pool=a.pool)
    emb, reg, st = SentenceTransformerEmbedder(), IndexRegistry(), InMemoryStateStore()
    await ingest_texts(DEMO_TENANT, load_corpus(), emb, reg, st, words=70, overlap=10)   # same as build_live_orchestrator
    ix = reg.get(DEMO_TENANT)
    golden = [json.loads(l) for l in (GOLDEN / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    labels, union, strict = build_gold(ix, golden)
    BENCH.mkdir(exist_ok=True)
    (BENCH / "retrieval_gold.jsonl").write_text("\n".join(json.dumps(x) for x in labels) + "\n")

    if a.reranker == "torch":
        rr = CrossEncoderReranker(max_length=a.max_length)
    else:
        rr = OnnxCrossEncoderReranker(model_file="model_int8.onnx" if a.reranker == "onnx-int8" else "model_fp32.onnx",
                                      max_length=a.max_length)
    ret = HybridRetriever(reg, rr, pool=a.pool, rerank_k=4, threshold=settings.abstain_threshold)
    diag = HybridRetriever(reg, _NullReranker(), pool=50, rerank_k=4, threshold=settings.abstain_threshold)

    per_q, per_q_strict, recall_diag, gate = [], [], {n: [] for n in (4, 8, 16, 20, 40, 50)}, 0
    stats = Counter()
    for r in golden:
        if r["id"] not in union:
            continue
        q = (await emb.embed([r["question"]]))[0]
        res = await ret.retrieve(DEMO_TENANT, r["question"], q)
        gate += "prefilter_abstain" in res.stage_ms
        top_ids = [c.chunk.chunk_id for c in res.chunks]
        gold = union[r["id"]]
        m = score_question(gold, res.pool_ids, top_ids)
        per_q.append(m)
        if r["id"] in strict:
            per_q_strict.append(m)
        stats["abstain_flag"] += res.abstain
        full = (await diag.retrieve(DEMO_TENANT, r["question"], q)).pool_ids
        for n in recall_diag:
            recall_diag[n].append(len(gold & set(full[:n])) / len(gold))

    out = {"reranker": a.reranker, "pool": a.pool, "max_length": a.max_length,
           "n_questions": len(per_q), "n_gold_chunks_mean": round(sum(len(union[k]) for k in union) / len(union), 3),
           "prefilter_gate_abstain": gate, "retrieval_abstain_flag": stats["abstain_flag"],
           "metrics": aggregate(per_q), "strict_exactly_one_labels": {"n_questions": len(per_q_strict),
                                                                    "metrics": aggregate(per_q_strict)},
           "thresholds": THRESH,
           "pass": {k: aggregate(per_q)[k] >= v for k, v in THRESH.items()},
           "diag_recall_by_pool": {str(n): round(sum(v) / len(v), 4) for n, v in recall_diag.items()},
           "labels": {"questions_with_gold": len(union), "evidence_total": sum(len(x["evidence"]) for x in labels),
                      "evidence_exactly_one_chunk": sum(1 for x in labels for e in x["evidence"] if e["n_chunks"] == 1),
                      "evidence_two_chunks": sum(1 for x in labels for e in x["evidence"] if e["n_chunks"] == 2),
                      "evidence_zero_chunks": sum(1 for x in labels for e in x["evidence"] if e["n_chunks"] == 0),
                      "questions_excluded_from_strict": len(union) - len(strict)}}
    path = BENCH / f"retrieval_metrics_{a.reranker}_pool{a.pool}_len{a.max_length}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reranker", choices=["torch", "onnx", "onnx-int8"], default="torch")
    p.add_argument("--pool", type=int, default=Settings.rerank_pool)
    p.add_argument("--max-length", type=int, default=128)
    asyncio.run(main(p.parse_args()))
