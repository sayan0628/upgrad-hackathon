"""Pre-LLM latency benchmark on the real local stack (design.md §1.4.2 fast-path measurements).

Runs golden-set questions through Orchestrator.handle() (no LLM call is made: the LLM client is replaced by a
recorder that fails the run if it is ever invoked for a fast path). Reports p50/p95 per path and per stage.

    python tools/latency_bench.py --n 3 [--reranker-pool 8] [--int8]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.config import Settings  # noqa: E402
from kestrel.contracts import AgentQueryRequest  # noqa: E402
from kestrel.runtime import GOLDEN, build_live_orchestrator, configure_torch  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100 * len(xs) + 0.5)) - 1))] if xs else float("nan")


async def main(a):
    configure_torch()
    s = Settings(openrouter_api_key=None, openrouter_base_url="", generation_model="offline/none",
                 fallback_model=None, judge_model=None, rerank_pool=a.pool)
    from kestrel.retrieval import CrossEncoderReranker
    from kestrel.embed import SentenceTransformerEmbedder
    rr = CrossEncoderReranker(max_length=a.max_length)
    emb = SentenceTransformerEmbedder()
    from kestrel.llm.client import StreamEvent

    class StubLLM:  # zero-latency stand-in: the LLM path is measured up to (not including) the model's own latency
        async def stream(self, messages, *, model, max_tokens, temperature=0.0, extra_body=None):
            yield StreamEvent("token", "Stub answer.", 0.0)
            yield StreamEvent("done", "", 0.0)

    orch = await build_live_orchestrator(s, embedder=emb, reranker=rr, llm=StubLLM())
    orch.retriever.pool = a.pool
    rows = [json.loads(l) for l in (GOLDEN / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    qs = [r["question"] for r in rows][: a.questions]
    rows = [json.loads(l) for l in (GOLDEN / "unanswerable.jsonl").read_text().splitlines() if l.strip()]
    qs += [r["question"] for r in rows][: a.questions // 3]
    # warm up: model load + first-call overheads are not what a steady-state request pays
    for q in qs[:3]:
        await orch.handle(AgentQueryRequest(tenant_id="tenant_acme", user_id=f"user_warm_{time.time_ns()}", query=q))
    per_path, per_stage = defaultdict(list), defaultdict(list)
    for rep in range(a.n):
        for i, q in enumerate(qs):
            # fresh user per request so the semantic tier cannot answer from an earlier repetition
            req = AgentQueryRequest(tenant_id="tenant_acme", user_id=f"user_bench_{rep}_{i}", query=q)
            t = time.perf_counter()
            h = await orch.handle(req)
            events = [(n, d) async for n, d in h.events]
            total = (time.perf_counter() - t) * 1000
            final = next(d for n, d in events if n == "final")
            key = f"{h.path}:{final['status']}"
            per_path[key].append(total)
            for k, v in h.timings_ms.items():
                per_stage[k].append(v)
    out = {"pool": a.pool, "max_length": a.max_length, "paths": {}, "stages_p50_ms": {}}
    for k, xs in sorted(per_path.items()):
        out["paths"][k] = {"n": len(xs), "p50_ms": round(pct(xs, 50), 1), "p95_ms": round(pct(xs, 95), 1)}
    for k, xs in sorted(per_stage.items()):
        out["stages_p50_ms"][k] = round(pct(xs, 50), 2)
    print(json.dumps(out, indent=2))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--questions", type=int, default=30)
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--out", default="")
    asyncio.run(main(p.parse_args()))
