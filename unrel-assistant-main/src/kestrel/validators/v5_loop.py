"""V5 Loop & breaker (FR-08; KPI-16, KPI-17).

Scenarios (fault-injection through the public LLM / orchestrator constructors):
  * breaker: the same failing tool call twice in one request opens the breaker; the third is refused (BREAKER_OPEN)
  * malformed LLM output (empty) and hallucinated output are retracted, not returned as OK
  * LLM 429 is surfaced as 429, not retried in a loop
  * steps per completed golden query (1 retrieval + LLM calls) are <= 5 each and mean <= 2.2

Limitation: the breaker is only reachable by repeated dispatch inside one request; the HTTP path does not retry a
raised dispatch error, so the breaker scenario calls `orch.harness.call` (the harness's public method) directly.
`HALTED_BUDGET` before the request deadline is not exercised: this build has no per-request deadline setting.
"""
from __future__ import annotations

import uuid

from ..contracts import ErrorCode, PlatformError
from ..llm.client import LLMUsage, StreamEvent
from .base import Env, Fault, Outcome, Validator
from .fixtures import StubLLM, golden_set
from .v3_guardrails import RecordingLLM

BAD_TRANSITION = {"entity": "CLM-00001", "from_state": "SUBMITTED", "to_state": "PAID"}
HALLUCINATION = "Quantum flux capacitors are sold to Mars colonies at a discount [1]."


class EmptyLLM(StubLLM):
    """Malformed output: the stream ends with no tokens."""

    async def stream(self, messages, **kw):
        self.calls += 1
        yield StreamEvent("done", usage=LLMUsage(0, 0))


class HallucinatingLLM(StubLLM):
    """Ungrounded output: a fluent answer with no support in the retrieved chunks."""

    async def stream(self, messages, **kw):
        self.calls += 1
        for i, w in enumerate(HALLUCINATION.split()):
            yield StreamEvent("token", (" " if i else "") + w, float(i + 1))
        yield StreamEvent("done", usage=LLMUsage(0, 0))


class RateLimitedLLM(StubLLM):
    async def stream(self, messages, **kw):
        self.calls += 1
        raise PlatformError(ErrorCode.UPSTREAM_RATE_LIMITED, "upstream 429")
        yield  # pragma: no cover  (makes this an async generator)


class V5Loop(Validator):
    id = "V5"
    title = "Loop & breaker"
    features = "FR-08"
    kpis = "KPI-16, KPI-17"
    faults = (
        Fault("breaker_threshold_99", "breaker opens only after 99 failures",
              post=lambda o: setattr(o.harness.breaker, "max_failures", 99)),
        Fault("grounding_veto_off", "grounding veto disabled (grounding_min = 0)", settings={"grounding_min": 0.0}),
    )
    skip_reasons = ("HALTED_BUDGET before the request deadline: no deadline setting in this build",)

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()

        # ---- breaker: two identical failing dispatches in one request, the third must be refused
        llm = RecordingLLM()
        sb = await env.sandbox(tenant, fault, llm=llm)
        try:
            req_id = uuid.uuid4().hex
            args = {"tenant_id": tenant, **BAD_TRANSITION}
            seen: list[str] = []
            for _ in range(3):
                try:
                    await sb.orch.harness.call(caller="execution", tenant_id=tenant, session_id="validator-v5",
                                               request_id=req_id, tool="transition_workflow", args=dict(args))
                    seen.append("OK")
                except PlatformError as e:
                    seen.append(e.code.name)
                except Exception as e:  # noqa: BLE001
                    seen.append(type(e).__name__)
            sb.orch.harness.breaker.end_request(req_id)
            out.check(seen[2] == "BREAKER_OPEN", f"third identical failing call gave {seen[2]}, expected BREAKER_OPEN",
                      severity="critical")
            out.check(seen[1] != "BREAKER_OPEN", "breaker opened before the second failure", severity="major")

            # ---- steps per completed golden query
            gold = [g for g in golden_set() if g["answerable"]][::25]
            steps = []
            for i, g in enumerate(gold):
                before = llm.calls
                r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant, "Accept": "text/event-stream"},
                                         json={"tenant_id": tenant, "user_id": f"user_v5s{i}", "query": g["question"]})
                after = llm.calls
                if "event: final" not in r.text:
                    out.check(False, f"golden query {g['id']} did not complete", severity="major")
                    continue
                steps.append(1 + (after - before))                  # 1 retrieval + generation calls
            for s in steps:
                out.check(s <= 5, f"a query took {s} steps (max 5)", severity="major")
            mean = sum(steps) / len(steps) if steps else 0.0
            out.metric("KPI-17 mean ReAct steps per completed task", mean, 2.2, "<=", len(steps))
        finally:
            await sb.aclose()

        # ---- malformed / hallucinated output must be retracted; 429 must surface as 429
        escapes = 0
        for label, llm, want in (("empty", EmptyLLM(), "ABSTAINED"), ("hallucinated", HallucinatingLLM(), "ABSTAINED")):
            sb2 = await env.sandbox(tenant, fault, llm=llm)
            try:
                r = await sb2.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                          json={"tenant_id": tenant, "user_id": f"user_v5{label}",
                                                "query": "What is the retention period for supplier certificates?"})
                status = r.json().get("status") if r.status_code == 200 else f"HTTP {r.status_code}"
                ok = status == want
                escapes += not ok
                out.check(ok, f"{label} LLM output gave {status}, expected {want}", severity="critical")
            finally:
                await sb2.aclose()
        sb3 = await env.sandbox(tenant, fault, llm=RateLimitedLLM())
        try:
            r = await sb3.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                      json={"tenant_id": tenant, "user_id": "user_v5rl", "query": "What is the retention period?"})
            ok = r.status_code == 429
            escapes += not ok
            out.check(ok, f"upstream 429 gave HTTP {r.status_code}, expected 429")
        finally:
            await sb3.aclose()

        # KPI-16: every scenario terminates with a terminal response (no escape). Scenarios counted here are the
        # three LLM-output cases plus the breaker case.
        breaker_escape = 0 if seen[2] == "BREAKER_OPEN" else 1
        n_scen = 4
        out.metric("KPI-16 loop-escape rate (escapes = 0)", escapes + breaker_escape, 0, "<=", n_scen)
        return out
