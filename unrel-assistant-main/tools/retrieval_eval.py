"""Retrieval quality vs latency on the insurance golden set: does the cited evidence document appear in the
top-4 chunks? Reports hit@4, abstain rate on answerable items, and rerank p50. Used to pick rerank_pool/max_length.

    python tools/retrieval_eval.py --pool 4 --max-length 128 --reranker torch
    python tools/retrieval_eval.py --pool 4 --max-length 128 --reranker onnx        # models/reranker/model_fp32.onnx
    python tools/retrieval_eval.py --pool 4 --max-length 128 --reranker onnx-int8   # models/reranker/model_int8.onnx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.embed import SentenceTransformerEmbedder  # noqa: E402
from kestrel.retrieval import CrossEncoderReranker, HybridRetriever, IndexRegistry, ingest_texts  # noqa: E402
from kestrel.retrieval.rerank import OnnxCrossEncoderReranker  # noqa: E402
from kestrel.runtime import GOLDEN, configure_torch, load_corpus  # noqa: E402
from kestrel.stores import InMemoryStateStore  # noqa: E402


async def main(a):
    configure_torch()
    if a.reranker == "torch":
        rr = CrossEncoderReranker(max_length=a.max_length)
    else:
        rr = OnnxCrossEncoderReranker(model_file="model_int8.onnx" if a.reranker == "onnx-int8" else "model_fp32.onnx",
                                      max_length=a.max_length)
    emb, reg, st = SentenceTransformerEmbedder(), IndexRegistry(), InMemoryStateStore()
    await ingest_texts("tenant_eval", load_corpus(), emb, reg, st, words=a.words, overlap=a.overlap)
    ret = HybridRetriever(reg, rr, pool=a.pool, threshold=0.42)
    rows = [json.loads(l) for l in (GOLDEN / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    hits, abst, ms = 0, 0, []
    for r in rows:
        q = await emb.embed([r["question"]])
        t = time.perf_counter()
        res = await ret.retrieve("tenant_eval", r["question"], q[0])
        ms.append((time.perf_counter() - t) * 1000)
        docs = {c.chunk.source for c in res.chunks}
        if any(e["doc"] in docs for e in r["evidence"]):
            hits += 1
        abst += res.abstain
    out = {"reranker": a.reranker, "items": len(rows), "pool": a.pool, "max_length": a.max_length, "words": a.words,
           "evidence_doc_hit_at4": round(hits / len(rows), 3), "abstain_rate_on_answerable": round(abst / len(rows), 3),
           "retrieve_p50_ms": round(statistics.median(ms), 1), "retrieve_p95_ms": round(sorted(ms)[int(0.95 * len(ms)) - 1], 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--reranker", choices=["torch", "onnx", "onnx-int8"], default="torch")
    p.add_argument("--words", type=int, default=70)
    p.add_argument("--overlap", type=int, default=10)
    asyncio.run(main(p.parse_args()))
