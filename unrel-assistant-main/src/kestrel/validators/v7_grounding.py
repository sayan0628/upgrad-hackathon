"""V7 Grounding (FR-10; KPI-20, 21, 22, 24).

Deterministic part (runs in quick): the inline veto must retract an ungrounded draft, and verified answers must
carry a grounding score at or above the configured minimum. The RAGAS (KPI-20..22) and claim-level (KPI-24) measures
need the pinned judge model, which is not available locally, so they are reported SKIPPED.
"""
from __future__ import annotations

from typing import Any

from .base import Env, Fault, Outcome, Validator
from .fixtures import golden_set, sse_events
from .v5_loop import HALLUCINATION, HallucinatingLLM


def _drop_retract_event(orch) -> None:
    """Seeded fault: the veto still withholds the draft but never tells the client (`retract` event suppressed)."""
    real = orch.handle

    async def handle(req, trace_id=None):
        h = await real(req, trace_id)
        inner = h.events

        async def events():
            async for name, data in inner:
                if name != "retract":
                    yield name, data
        h.events = events()
        return h
    orch.handle = handle


class V7Grounding(Validator):
    id = "V7"
    title = "Grounding"
    features = "FR-10"
    kpis = "KPI-20, KPI-21, KPI-22, KPI-24"
    faults = (
        Fault("grounding_veto_off", "grounding veto disabled (grounding_min = 0)", settings={"grounding_min": 0.0}),
        Fault("retract_event_suppressed", "veto withholds the draft but emits no retract event", post=_drop_retract_event),
    )
    skip_reasons = ("KPI-20..22 RAGAS and KPI-24 claim-level rate: pinned judge not configured locally (no judge quota)",)

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        sb = await env.sandbox(tenant, fault, llm=HallucinatingLLM())
        try:
            q = "What is the retention period for supplier certificates?"
            r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant, "Accept": "text/event-stream"},
                                     json={"tenant_id": tenant, "user_id": "user_v7h", "query": q})
            ev = sse_events(r.text)
            final = ev.get("final") or {}
            out.check("event: retract" in r.text, "ungrounded draft produced no retract event", severity="critical")
            out.check(final.get("status") == "ABSTAINED" and final.get("answer") != HALLUCINATION,
                      f"ungrounded draft was returned (status {final.get('status')})", severity="critical")
        finally:
            await sb.aclose()

        sb2 = await env.sandbox(tenant, fault)
        try:
            qs = [g["question"] for g in golden_set() if g["answerable"]][:10]
            floor = env.settings.grounding_min
            for i, q in enumerate(qs):
                r = await sb2.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                          json={"tenant_id": tenant, "user_id": f"user_v7g{i}", "query": q})
                body: dict[str, Any] = r.json() if r.status_code == 200 else {}
                if body.get("status") == "OK":
                    g = body.get("grounding_score")
                    out.check(g is not None and g >= floor, f"verified answer {i} has grounding {g} < {floor}",
                              severity="major")
                else:
                    out.check(False, f"golden question {i} was not answered (HTTP {r.status_code})", severity="major")
        finally:
            await sb2.aclose()
        return out
