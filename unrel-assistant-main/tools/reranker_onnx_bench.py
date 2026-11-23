"""Torch vs ONNX (fp32, dynamic INT8) cross-encoder: score agreement on the golden set, and per-call latency.

    python tools/reranker_onnx_bench.py                  # agreement + latency, writes bench_results/reranker_onnx_bench.json
    python tools/reranker_onnx_bench.py --skip-latency   # agreement only

Agreement uses the REAL candidate pool from HybridRetriever (fused BM25 + dense, RRF order) for every golden question,
so the pairs are the ones the reranker actually sees. Latency times one `score()` call per request-sized batch
(4 or 8 pairs) after warm-up, at max_length 128 and 256. Both use 2 intra-op threads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.embed import SentenceTransformerEmbedder  # noqa: E402
from kestrel.retrieval import CrossEncoderReranker, HybridRetriever, IndexRegistry, ingest_texts  # noqa: E402
from kestrel.retrieval.rerank import OnnxCrossEncoderReranker  # noqa: E402
from kestrel.runtime import GOLDEN, configure_torch, load_corpus  # noqa: E402
from kestrel.stores import InMemoryStateStore  # noqa: E402

OUT = ROOT / "bench_results" / "reranker_onnx_bench.json"


class _NullReranker:
    """Returns zeros: lets the real retriever produce its candidate pool without spending rerank time."""

    async def score(self, query, texts):
        return [0.0] * len(texts)


def make_reranker(kind: str, max_length: int):
    if kind == "torch":
        return CrossEncoderReranker(max_length=max_length)
    f = "model_int8.onnx" if kind == "onnx-int8" else "model_fp32.onnx"
    return OnnxCrossEncoderReranker(model_file=f, max_length=max_length, intra_op_threads=2)


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p)) if len(xs) else float("nan")


async def build_pools(pool: int):
    emb, reg, st = SentenceTransformerEmbedder(), IndexRegistry(), InMemoryStateStore()
    await ingest_texts("tenant_eval", load_corpus(), emb, reg, st, words=70, overlap=10)
    ret = HybridRetriever(reg, _NullReranker(), pool=pool, threshold=0.42)
    rows = [json.loads(l) for l in (GOLDEN / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    ix = reg.get("tenant_eval")
    pools = []
    for r in rows:
        q = await emb.embed([r["question"]])
        res = await ret.retrieve("tenant_eval", r["question"], q[0])
        pools.append((r["question"], [ix.chunks[ix.by_id[c]].text for c in res.pool_ids]))
    return pools


async def agreement(pools, max_length: int):
    ref = make_reranker("torch", max_length)
    engines = {"onnx-fp32": make_reranker("onnx", max_length), "onnx-int8": make_reranker("onnx-int8", max_length)}
    stats = {k: {"max_abs": 0.0, "sum_abs": 0.0, "n_pairs": 0, "over_0.01": 0, "top4_set_overlap": [],
                 "top4_identical": 0, "top1_identical": 0, "pool4_order_identical": 0, "n_q": 0} for k in engines}
    for q, texts in pools:
        if not texts:
            continue
        s_ref = np.asarray(await ref.score(q, texts))
        for name, eng in engines.items():
            s = np.asarray(await eng.score(q, texts))
            d = np.abs(s - s_ref)
            st = stats[name]
            st["max_abs"] = max(st["max_abs"], float(d.max()))
            st["sum_abs"] += float(d.sum())
            st["over_0.01"] += int((d > 0.01).sum())
            st["n_pairs"] += len(d)
            st["n_q"] += 1
            # top-4 agreement at the pool the retriever actually uses (pool 8 here: at pool 4 the top-4 is the whole pool)
            k = min(4, len(texts))
            t4, v4 = set(np.argsort(-s_ref)[:k].tolist()), set(np.argsort(-s)[:k].tolist())
            st["top4_set_overlap"].append(len(t4 & v4) / k)
            st["top4_identical"] += int(t4 == v4)
            st["top1_identical"] += int(int(np.argmax(s_ref)) == int(np.argmax(s)))
            # ordering agreement on the 4 pairs the configured pool (rerank_pool=4) sends to the reranker
            p4 = texts[:4]
            if len(p4) == 4:
                o_ref = np.argsort(-s_ref[:4], kind="stable").tolist()
                o_eng = np.argsort(-s[:4], kind="stable").tolist()
                st["pool4_order_identical"] += int(o_ref == o_eng)
    out = {}
    for name, st in stats.items():
        n = max(st["n_q"], 1)
        out[name] = {
            "max_len": max_length, "questions": st["n_q"], "pairs": st["n_pairs"],
            "max_abs_score_diff": round(st["max_abs"], 6),
            "mean_abs_score_diff": round(st["sum_abs"] / max(st["n_pairs"], 1), 6),
            "pairs_with_abs_diff_over_0.01": st["over_0.01"],
            "top4_agreement_pool8": round(float(np.mean(st["top4_set_overlap"])), 4),
            "top4_identical_sets_pool8": round(st["top4_identical"] / n, 4),
            "top1_agreement_pool8": round(st["top1_identical"] / n, 4),
            "pool4_order_identical": round(st["pool4_order_identical"] / n, 4),
        }
    return out


async def latency(pools, reps: int, warm: int = 10):
    """Per-call p50/p95 of one score() call on 4 and 8 real candidate pairs (rotating golden questions)."""
    res = {}
    for L in (128, 256):
        for kind in ("torch", "onnx-fp32", "onnx-int8"):
            rr = make_reranker("torch" if kind == "torch" else ("onnx-int8" if kind == "onnx-int8" else "onnx"), L)
            rr.warm() if hasattr(rr, "warm") else None
            for k in (4, 8):
                usable = [(q, t[:k]) for q, t in pools if len(t) >= k]
                for i in range(warm):
                    q, t = usable[i % len(usable)]
                    await rr.score(q, t)
                ms = []
                for i in range(reps):
                    q, t = usable[(i * 7 + 3) % len(usable)]
                    t0 = time.perf_counter()
                    await rr.score(q, t)
                    ms.append((time.perf_counter() - t0) * 1000)
                key = f"{kind}|pairs={k}|max_len={L}"
                res[key] = {"n": len(ms), "p50_ms": round(pct(ms, 50), 1), "p95_ms": round(pct(ms, 95), 1)}
                print(key, res[key], flush=True)
    return res


async def main(a):
    configure_torch()
    pools = await build_pools(a.pool)
    print(f"pools built: {len(pools)} questions, candidates per question at pool {a.pool}: "
          f"{min(len(t) for _, t in pools)}..{max(len(t) for _, t in pools)}", flush=True)
    out = {"pool": a.pool, "agreement": {}, "latency_ms": {}}
    for L in (128, 256):
        out["agreement"][f"max_len={L}"] = await agreement(pools, L)
        print(json.dumps(out["agreement"][f"max_len={L}"], indent=1), flush=True)
    if not a.skip_latency:
        out["latency_ms"] = await latency(pools, a.reps)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--reps", type=int, default=150)
    p.add_argument("--skip-latency", action="store_true")
    asyncio.run(main(p.parse_args()))
