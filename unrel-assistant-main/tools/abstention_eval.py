"""KPI-23 abstention accuracy and answerable false-abstain rate, through the real Orchestrator.handle() path.

    python tools/abstention_eval.py --reranker torch

* unanswerable.jsonl (150 items, all should abstain)  -> KPI-23 = share ending in status ABSTAINED (threshold >= 98%)
* golden_set.jsonl answerable items (213)            -> false-abstain = share ending in status ABSTAINED
Both are also reported with REFUSED counted as a decline (lenient), and with a per-path breakdown.

The LLM is a stub that is recorded and must not be called on fast paths (the run asserts this per request). The stub
answers with the first context passage verbatim when a context is present, so a grounded answer is produced and the
LLM path only abstains through the real grounding veto. This means LLM-path outcomes here reflect retrieval, triage
and verification, not generation quality. Unanswerable items that reach the LLM path are counted as non-abstentions
(pessimistic); they are reported separately because a real model could still decline them.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.config import Settings  # noqa: E402
from kestrel.contracts import AgentQueryRequest  # noqa: E402
from kestrel.llm.client import StreamEvent  # noqa: E402
from kestrel.retrieval import CrossEncoderReranker  # noqa: E402
from kestrel.retrieval.rerank import OnnxCrossEncoderReranker  # noqa: E402
from kestrel.runtime import GOLDEN, build_live_orchestrator, configure_torch  # noqa: E402

BENCH = ROOT / "bench_results"


class StubLLM:
    def __init__(self):
        self.calls = 0

    async def stream(self, messages, *, model, max_tokens, temperature=0.0, extra_body=None):
        self.calls += 1
        content = messages[-1]["content"]
        line = next((l for l in content.splitlines() if l.startswith("[1] ")), None)
        yield StreamEvent("token", line[4:] if line else "Stub answer.", 0.0)
        yield StreamEvent("done", "", 0.0)


def load(name):
    return [json.loads(l) for l in (GOLDEN / name).read_text().splitlines() if l.strip()]


async def run(orch, llm, rows, qtext, tag):
    out = []
    for i, r in enumerate(rows):
        before = llm.calls
        req = AgentQueryRequest(tenant_id="tenant_acme", user_id=f"user_abst_{tag}_{i}", query=qtext(r))
        t = time.perf_counter()
        h = await orch.handle(req)
        events = [(n, d) async for n, d in h.events]
        ms = (time.perf_counter() - t) * 1000
        final = next(d for n, d in events if n == "final")
        llm_called = llm.calls - before
        if h.path == "fast" and llm_called:
            raise AssertionError(f"LLM called on a fast path: {r['id'] if 'id' in r else i}")
        out.append({"id": r.get("id", str(i)), "kind": r.get("kind", r.get("category")), "path": h.path,
                    "route": h.route, "status": final["status"], "llm_called": llm_called, "ms": round(ms, 1)})
    return out


def summarise(res):
    n = len(res)
    strict = sum(x["status"] == "ABSTAINED" for x in res)
    lenient = sum(x["status"] in ("ABSTAINED", "REFUSED") for x in res)
    by = collections.Counter((x["path"], x["route"], x["status"]) for x in res)
    return {"n": n, "abstained_strict": strict, "rate_strict": round(strict / n, 4) if n else None,
            "declined_lenient": lenient, "rate_lenient": round(lenient / n, 4) if n else None,
            "breakdown_path_route_status": {"|".join(k): v for k, v in sorted(by.items())}}


async def main(a):
    configure_torch()
    settings = Settings(openrouter_api_key=None, openrouter_base_url="", generation_model="offline/none",
                        fallback_model=None, judge_model=None)     # rerank_pool / threshold: production defaults
    from kestrel.embed import SentenceTransformerEmbedder
    if a.reranker == "torch":
        rr = CrossEncoderReranker()
    else:
        rr = OnnxCrossEncoderReranker(model_file="model_int8.onnx" if a.reranker == "onnx-int8" else "model_fp32.onnx")
    llm = StubLLM()
    orch = await build_live_orchestrator(settings, embedder=SentenceTransformerEmbedder(), reranker=rr, llm=llm)

    unans = await run(orch, llm, load("unanswerable.jsonl"), lambda r: r["question"], "u")
    golden = load("golden_set.jsonl")
    ans = await run(orch, llm, [r for r in golden if r["answerable"]], lambda r: r["question"], "a")
    neg_golden = await run(orch, llm, [r for r in golden if not r["answerable"]], lambda r: r["question"], "g")

    out = {"reranker": a.reranker, "threshold_KPI23": 0.98,
           "unanswerable_jsonl": summarise(unans),
           "answerable_golden_false_abstain": summarise(ans),
           "golden_unanswerable_rows": summarise(neg_golden),
           "unanswerable_reaching_llm_path": sum(x["path"] == "llm" for x in unans),
           "unanswerable_direct_answer_route": sum(x["route"] == "DIRECT_ANSWER" for x in unans),
           "p50_ms_unanswerable": sorted(x["ms"] for x in unans)[len(unans) // 2]}
    out["kpi23_pass"] = out["unanswerable_jsonl"]["rate_strict"] >= 0.98
    BENCH.mkdir(exist_ok=True)
    (BENCH / f"abstention_{a.reranker}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reranker", choices=["torch", "onnx", "onnx-int8"], default="torch")
    asyncio.run(main(p.parse_args()))
