"""V10 Telemetry (FR-11 OTel; KPI-1, 25, 30 in the matrix).

No OpenTelemetry exporter exists in this build, so span emission is checked through its in-band proxy: the SSE `meta`
event carries one timing per hop (design §2.3 span names). Checked here:
  * traceparent continuity: the trace id sent in `traceparent` comes back in the meta and final payloads (20 requests)
  * hop coverage: a RAG query reports timings for every hop that must emit a span
The collector-down check (the request path must not block) needs a collector, so it is SKIPPED.
"""
from __future__ import annotations

import secrets

from .base import Env, Fault, Outcome, Validator
from .fixtures import golden_set, sse_events

REQUIRED_HOPS = ("hook.pre", "cache.lookup", "triage.classify", "mcp.rag.bm25", "mcp.rag.rerank")


def _drop_trace_id(orch) -> None:
    """Seeded fault: the gateway's trace id is replaced by a fresh one (traceparent continuity broken)."""
    real = orch.handle

    async def handle(req, trace_id=None):
        h = await real(req, None)
        inner = h.events

        async def events():
            new = secrets.token_hex(16)
            async for name, data in inner:
                if isinstance(data, dict) and "trace_id" in data:
                    data = {**data, "trace_id": new}
                yield name, data
        h.events = events()
        return h
    orch.handle = handle


def _drop_rerank_timing(orch) -> None:
    """Seeded fault: the cross-encoder hop emits no timing (its span is missing)."""
    real = orch.handle

    async def handle(req, trace_id=None):
        h = await real(req, trace_id)
        inner = h.events

        async def events():
            async for name, data in inner:
                if name == "meta":
                    data = {**data, "timings_ms": {k: v for k, v in data["timings_ms"].items() if k != "mcp.rag.rerank"}}
                yield name, data
        h.events = events()
        return h
    orch.handle = handle


class V10Telemetry(Validator):
    id = "V10"
    title = "Telemetry"
    features = "FR-11 (OTel)"
    kpis = "KPI-1, KPI-25, KPI-30"
    faults = (
        Fault("trace_id_not_propagated", "gateway ignores traceparent and mints a new trace id", post=_drop_trace_id),
        Fault("rerank_span_missing", "cross-encoder hop reports no timing", post=_drop_rerank_timing),
    )
    skip_reasons = ("collector-down request (the request path must not block): no OTel collector in this build",)

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        sb = await env.sandbox(tenant, fault)
        try:
            qs = [g["question"] for g in golden_set() if g["answerable"]]
            cont = 0
            for i in range(20):
                tid = secrets.token_hex(16)
                r = await sb.client.post("/api/v1/agent/query",
                                         headers={"X-Tenant-Id": tenant, "traceparent": f"00-{tid}-{secrets.token_hex(8)}-01",
                                                  "Accept": "text/event-stream"},
                                         json={"tenant_id": tenant, "user_id": f"user_v10{i}", "query": qs[i]})
                ev = sse_events(r.text)
                ok = ev.get("meta", {}).get("trace_id") == tid and ev.get("final", {}).get("trace_id") == tid
                cont += out.check(ok, f"traceparent {tid[:8]} not carried through request {i}", severity="major")
            out.metric("traceparent continuity", cont / 20, 1.0, "==", 20)

            hop_ok = 0
            for i, q in enumerate(qs[20:25]):   # fresh questions: cache hits carry no RAG timings
                r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant, "Accept": "text/event-stream"},
                                         json={"tenant_id": tenant, "user_id": f"user_v10h{i}", "query": q})
                timings = sse_events(r.text).get("meta", {}).get("timings_ms", {})
                missing = [h for h in REQUIRED_HOPS if h not in timings]
                hop_ok += out.check(not missing, f"RAG query {i} missing hop timings: {missing}", severity="major")
            out.metric("hop timing coverage (RAG queries)", hop_ok / 5, 1.0, "==", 5)
            return out
        finally:
            await sb.aclose()
