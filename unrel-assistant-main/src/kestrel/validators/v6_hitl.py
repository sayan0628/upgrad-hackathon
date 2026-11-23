"""V6 HITL gate (FR-09; KPI-19 = 100%).

A Tier 3 request must never change state without an approved record. Scenarios run through the public API:
the query route creates a PENDING_APPROVAL, the reviewer routes approve/reject. The state store is read only to
confirm whether an entity changed (InMemoryStateStore.get_entity is the public read used by the tests too).
"""
from __future__ import annotations

from ..stores import InMemoryStateStore
from .base import Env, Fault, Outcome, Validator

REQ, REV = "user_v6req", "user_v6rev"
REV_H = {"X-User-Id": REV}


def _reviewer_check_removed(orch) -> None:
    """Seeded fault: a requester may approve their own request (the reviewer != requester check is gone)."""
    real = orch.approvals.approve

    async def approve(tenant, approval_id, reviewer):
        view = await orch.approvals.state.get_approval(tenant, approval_id)
        if view and view["requested_by"] == reviewer:
            reviewer = reviewer + "~"          # the real check compares against requested_by; this skips it
        return await real(tenant, approval_id, reviewer)
    orch.approvals.approve = approve


def _replay_conflicts(orch) -> None:
    """Seeded fault: replaying an approved request returns a conflict instead of the stored result."""
    from ..contracts import ErrorCode, PlatformError
    real = orch.approvals.approve

    async def approve(tenant, approval_id, reviewer):
        view = await orch.approvals.state.get_approval(tenant, approval_id)
        if view and view["status"] == "APPROVED":
            raise PlatformError(ErrorCode.IDEMPOTENCY_CONFLICT, "approval is APPROVED")
        return await real(tenant, approval_id, reviewer)
    orch.approvals.approve = approve


class V6HITL(Validator):
    id = "V6"
    title = "HITL gate"
    features = "FR-09"
    kpis = "KPI-19"
    faults = (
        Fault("reviewer_check_removed", "requester may approve own request", post=_reviewer_check_removed),
        Fault("replay_not_idempotent", "replay of an approved request returns 409", post=_replay_conflicts),
    )

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        sb = await env.sandbox(tenant, fault)
        try:
            ent = "CLM-00001"
            before = sb.state.get_entity(tenant, ent)["state"]
            pending = await self._request(sb, tenant, REQ, f"Move claim {ent} to UNDER_REVIEW")
            out.check(pending.status_code == 202 and pending.json().get("approval_id"), "T3 request did not return a PENDING_APPROVAL")
            aid = (pending.json() or {}).get("approval_id")
            out.check(sb.state.get_entity(tenant, ent)["state"] == before, "T3 request changed state before approval",
                      severity="critical")

            # self-approval must be rejected
            r = await sb.client.post(f"/api/v1/approvals/{aid}/approve",
                                     headers={"X-Tenant-Id": tenant, "X-User-Id": REQ})
            out.check(r.status_code == 403, f"self-approval returned HTTP {r.status_code}, expected 403", severity="critical")
            # approve without a caller identity must be rejected (reviewer != requester cannot be evaluated otherwise)
            r = await sb.client.post(f"/api/v1/approvals/{aid}/approve", headers={"X-Tenant-Id": tenant})
            out.check(r.status_code >= 400, f"anonymous approval returned HTTP {r.status_code}", severity="critical")
            out.check(sb.state.get_entity(tenant, ent)["state"] == before,
                      "state changed after a rejected approval attempt", severity="critical")

            # reviewer list contains the request
            r = await sb.client.get("/api/v1/approvals", headers={"X-Tenant-Id": tenant, **REV_H})
            out.check(r.status_code == 200 and any(a.get("approval_id") == aid for a in r.json().get("approvals", [])),
                      "pending approval missing from the reviewer list")

            # a correct approval commits exactly once; a replay returns the stored result
            r = await sb.client.post(f"/api/v1/approvals/{aid}/approve", headers={"X-Tenant-Id": tenant, **REV_H})
            out.check(r.status_code == 200 and r.json().get("status") == "APPROVED",
                      f"reviewer approval returned HTTP {r.status_code}", severity="critical")
            out.check(sb.state.get_entity(tenant, ent)["state"] == "UNDER_REVIEW",
                      "approved T3 request did not commit", severity="critical")
            r2 = await sb.client.post(f"/api/v1/approvals/{aid}/approve", headers={"X-Tenant-Id": tenant, **REV_H})
            out.check(r2.status_code == 200 and r2.json().get("replayed") is True,
                      f"replay of the same approval returned HTTP {r2.status_code}, expected an idempotent 200", severity="major")
            audits = [e for e in sb.state.export_audit(tenant) if e.get("event_type") == "action.committed"]
            out.check(len(audits) == 1, f"expected one action.committed audit event, found {len(audits)}", severity="critical")

            # rejection leaves state alone
            p2 = await self._request(sb, tenant, REQ, f"Move claim {ent} to APPROVED")
            aid2 = (p2.json() or {}).get("approval_id")
            r = await sb.client.post(f"/api/v1/approvals/{aid2}/reject", headers={"X-Tenant-Id": tenant, **REV_H},
                                     json={"reason": "validator scenario"})
            out.check(r.status_code == 200, f"reject returned HTTP {r.status_code}")
            out.check(sb.state.get_entity(tenant, ent)["state"] == "UNDER_REVIEW", "rejected request changed state",
                      severity="critical")

            # repaired arguments are shown to the requester and the reviewer
            p3 = await self._request(sb, tenant, REQ, "Move claim CLM-2 to INFO_REQUESTED")
            body3 = p3.json() or {}
            out.check("CLM-00002" in (body3.get("remediation") or "") or "CLM-00002" in (body3.get("answer") or ""),
                      "repaired identifier CLM-2 -> CLM-00002 was not shown to the requester", severity="major")

            # expiry never auto-approves: with a zero TTL the pending request must not commit
            sb_exp = await env.sandbox(tenant, fault, state=InMemoryStateStore(approval_ttl_s=0.0))
            try:
                before_e = sb_exp.state.get_entity(tenant, ent)["state"]
                p4 = await self._request(sb_exp, tenant, REQ, f"Move claim {ent} to UNDER_REVIEW")
                aid4 = (p4.json() or {}).get("approval_id")
                r = await sb_exp.client.post(f"/api/v1/approvals/{aid4}/approve", headers={"X-Tenant-Id": tenant, **REV_H})
                out.check(r.status_code != 200 and sb_exp.state.get_entity(tenant, ent)["state"] == before_e,
                          "an expired approval was applied", severity="critical")
            finally:
                await sb_exp.aclose()
            return out
        finally:
            await sb.aclose()

    @staticmethod
    async def _request(sb, tenant: str, user: str, query: str):
        return await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                    json={"tenant_id": tenant, "user_id": user, "query": query})
