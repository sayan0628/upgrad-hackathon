"""V9 Load & chaos (FR-12; KPI-31..34 at smoke scale).

Quick profile = smoke only: a few concurrent virtual users drive the in-process app for ~35 s, then each seeded fault
gets a short 4 s run. The smoke proves the harness and the percentile plumbing; it is not the 50K-VU SLA result.
The 50K-VU k6/JMeter run, the 1-hour soak (KPI-36), the chaos suite and the llm-sim distribution check need external
infrastructure and are reported SKIPPED. The Full profile's last SKIPPED result is shown with its own date in the UI.
"""
from __future__ import annotations

import asyncio
import math
import time

from .base import Env, Fault, Outcome, Validator
from .fixtures import StubLLM, golden_set

SMOKE_S = 35.0
FAULT_SMOKE_S = 4.0
VUS = 4


class SlowLLM(StubLLM):
    """Seeded fault: every generation takes 2.5 s (an LLM latency spike)."""

    async def stream(self, messages, **kw):
        await asyncio.sleep(2.5)
        async for ev in super().stream(messages, **kw):
            yield ev


class FlakyLLM(StubLLM):
    """Seeded fault: every second generation raises (5xx surface)."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def stream(self, messages, **kw):
        self.attempts += 1
        if self.attempts % 2 == 0:
            raise RuntimeError("injected upstream failure")
        async for ev in super().stream(messages, **kw):
            yield ev


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(p * len(s)) - 1)]


class V9Load(Validator):
    id = "V9"
    title = "Load & chaos (smoke)"
    features = "FR-12"
    kpis = "KPI-31, KPI-32, KPI-33, KPI-34 (smoke scale)"
    faults = (
        Fault("llm_latency_spike", "generation takes 2.5 s", build={"llm": SlowLLM()}),
        Fault("llm_intermittent_5xx", "every second generation raises", build={"llm": FlakyLLM()}),
    )
    skip_reasons = (
        "50K-VU k6/JMeter run (KPI-31..35): needs the external load rig and llm-sim",
        "1-hour soak (KPI-36): needs a 1-hour run window",
        "chaos suite (Redis kill, MCP hang, reranker slowdown): needs the production topology",
    )

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        duration = FAULT_SMOKE_S if fault else SMOKE_S
        sb = await env.sandbox(tenant, fault)
        try:
            base = [g["question"] for g in golden_set() if g["answerable"]][:20]
            # plain factoids mostly take the extractive fast path; "Explain ..." is high complexity, so it reaches the LLM
            qs = [q if k % 2 else f"Explain {q}" for k, q in enumerate(base)]
            lat: dict[str, list[float]] = {"fast": [], "llm": []}
            counts = {"total": 0, "5xx": 0}
            deadline = time.perf_counter() + duration

            async def vu(k: int) -> None:
                i = 0
                while time.perf_counter() < deadline:
                    q = qs[(k * 7 + i) % len(qs)]
                    i += 1
                    t0 = time.perf_counter()
                    counts["total"] += 1
                    try:
                        r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                                 json={"tenant_id": tenant, "user_id": f"user_v9_{k}", "query": q})
                    except Exception:  # noqa: BLE001 - an escaped exception is a 5xx from the client's side
                        counts["5xx"] += 1
                        continue
                    ms = (time.perf_counter() - t0) * 1000
                    if r.status_code >= 500:
                        counts["5xx"] += 1
                    elif r.headers.get("x-path") in lat:
                        lat[r.headers["x-path"]].append(ms)

            await asyncio.gather(*(vu(k) for k in range(VUS)))
            total = max(1, counts["total"])
            err = counts["5xx"] / total
            p50, p95, p99 = (_pct(lat["fast"], p) for p in (0.50, 0.95, 0.99))
            out.check(p50 <= p95 <= p99, "percentiles are not monotonic", severity="critical")
            fast_n, llm_n = len(lat["fast"]), len(lat["llm"])
            out.check(llm_n > 0 and fast_n > 0, f"smoke produced {fast_n} fast and {llm_n} LLM-path samples",
                      severity="critical")
            out.metric("KPI-32 fast-path p95 (ms, smoke)", _pct(lat["fast"], 0.95), 250.0, "<=", max(fast_n, 1))
            out.metric("KPI-32 LLM-path p95 (ms, smoke)", _pct(lat["llm"], 0.95), 2200.0, "<=", max(llm_n, 1))
            out.metric("KPI-34 HTTP 5xx rate (smoke)", err, 0.0005, "<=", total)
            return out
        finally:
            await sb.aclose()
