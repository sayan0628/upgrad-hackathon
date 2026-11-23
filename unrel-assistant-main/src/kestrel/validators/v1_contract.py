"""V1 Contract & harness (FR-01, FR-07; KPI-14, KPI-15). Driven only through POST /api/v1/agent/query."""
from __future__ import annotations

from pydantic import ValidationError

from ..contracts import HTTP_STATUS, AgentQueryResponse, ErrorCode, PlatformError
from .base import Env, Fault, Outcome, Validator
from .fixtures import golden_set
from .models import Finding


def _valid_cases(tenant: str, n: int) -> list[tuple[dict, dict]]:
    qs = [g["question"] for g in golden_set() if g["answerable"]][:n]
    return [({"X-Tenant-Id": tenant}, {"tenant_id": tenant, "user_id": f"user_v1{i}", "query": q})
            for i, q in enumerate(qs)]


def _malformed_cases(tenant: str) -> list[tuple[str, dict, dict, ErrorCode]]:
    """(label, headers, body, expected code). Each must be rejected with exactly the documented code."""
    ok = {"tenant_id": tenant, "user_id": "user_v1x", "query": "What is the retention period?"}
    h = {"X-Tenant-Id": tenant}
    return [
        ("no tenant header", {}, ok, ErrorCode.TENANT_MISMATCH),
        ("body tenant differs", h, {**ok, "tenant_id": "tenant_validator_other"}, ErrorCode.TENANT_MISMATCH),
        ("empty query", h, {**ok, "query": ""}, ErrorCode.INVALID_PARAMS),
        ("query too long", h, {**ok, "query": "x" * 4001}, ErrorCode.INVALID_PARAMS),
        ("bad user id", h, {**ok, "user_id": "bob"}, ErrorCode.INVALID_PARAMS),
        ("bad session id", h, {**ok, "session_id": "not-a-uuid"}, ErrorCode.INVALID_PARAMS),
        ("unknown field", h, {**ok, "admin": True}, ErrorCode.INVALID_PARAMS),
        ("missing user id", h, {k: v for k, v in ok.items() if k != "user_id"}, ErrorCode.INVALID_PARAMS),
        ("query not a string", h, {**ok, "query": 42}, ErrorCode.INVALID_PARAMS),
    ]


# ---- seeded faults (applied to the orchestrator of the sandbox)
def _drop_trace_id(orch) -> None:
    real = orch.handle

    async def handle(req, trace_id=None):
        h = await real(req, trace_id)
        inner = h.events

        async def events():
            async for name, data in inner:
                if name == "final":
                    data = {k: v for k, v in data.items() if k != "trace_id"}
                yield name, data
        h.events = events()
        return h
    orch.handle = handle


def _reject_all(orch) -> None:
    real = orch.handle

    async def handle(req, trace_id=None):
        if req.query.startswith(("What", "How")):
            raise PlatformError(ErrorCode.INVALID_PARAMS, "rejected by fault")
        return await real(req, trace_id)
    orch.handle = handle


class V1Contract(Validator):
    id = "V1"
    title = "Contract & harness"
    features = "FR-01, FR-07"
    kpis = "KPI-14, KPI-15"
    faults = (
        Fault("response_missing_trace_id", "final payload drops trace_id (schema break)", post=_drop_trace_id),
        Fault("valid_queries_rejected", "handler rejects What/How queries with INVALID_PARAMS", post=_reject_all),
    )

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        sb = await env.sandbox(tenant, fault)
        try:
            ok_n = 0
            valid = _valid_cases(tenant, 20)
            for hdrs, body in valid:
                r = await sb.client.post("/api/v1/agent/query", headers=hdrs, json=body)
                try:
                    data = r.json()
                    if "mode" in data:            # extractive answers carry `mode`, which §1.2.6 does not declare
                        if not any("'mode'" in f.summary for f in out.findings):
                            out.findings.append(Finding(summary="undeclared response field 'mode' on extractive answers "
                                                                "(not in AgentQueryResponse)", severity="minor"))
                        data = {k: v for k, v in data.items() if k != "mode"}
                    AgentQueryResponse.model_validate(data)
                    schema_ok = r.status_code in (200, 202)
                except (ValidationError, ValueError):
                    schema_ok = False
                ok_n += out.check(schema_ok, f"valid query returned HTTP {r.status_code} or bad schema")
            out.metric("KPI-14 well-formed responses", ok_n / len(valid), 0.96, ">=", len(valid))

            rej_n = 0
            cases = _malformed_cases(tenant)
            for label, hdrs, body, code in cases:
                r = await sb.client.post("/api/v1/agent/query", headers=hdrs, json=body)
                is_json = r.headers.get("content-type", "").startswith("application/json")
                err = (r.json() or {}).get("error", {}) if is_json else {}
                good = r.status_code == HTTP_STATUS[code] and err.get("code") == int(code)
                rej_n += good
                out.check(good, f"malformed '{label}' gave HTTP {r.status_code} {err.get('name')}, expected {code.name}")
            out.metric("KPI-15 malformed inputs rejected with documented code", rej_n / len(cases), 0.90, ">=", len(cases))
            return out
        finally:
            await sb.aclose()
