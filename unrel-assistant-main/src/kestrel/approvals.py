"""Reviewer-side approval service (design.md FR-09, §1.2.12, D-3, D-26).

approve(): re-validate the stored request against CURRENT state, then consume the vault entry and apply it.
The commit is a compare-and-set, so a request that went stale (someone else moved the claim) fails with
STATE_CONFLICT and stays PENDING for the reviewer to reject. Replaying an approval returns the stored result.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from .contracts import ErrorCode, PlatformError
from .forms import validate_form
from .stores import InMemoryStateStore
from .workflows import check_transition


class ApprovalService:
    def __init__(self, state: InMemoryStateStore):
        self.state = state
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def list_pending(self, tenant: str) -> list[dict]:
        return await self.state.list_pending(tenant)

    async def get(self, tenant: str, approval_id: str) -> dict | None:
        return await self.state.get_approval(tenant, approval_id)

    async def approve(self, tenant: str, approval_id: str, reviewer: str) -> dict[str, Any]:
        async with self._locks[tenant]:
            view = await self.state.get_approval(tenant, approval_id)
            if view is None:
                raise PlatformError(ErrorCode.INVALID_PARAMS, "unknown approval")
            if view["status"] == "APPROVED":                                  # idempotent replay
                return {"approval_id": approval_id, "status": "APPROVED", "committed": view["result"], "replayed": True}
            if view["status"] != "PENDING":
                raise PlatformError(ErrorCode.IDEMPOTENCY_CONFLICT, f"approval is {view['status']}")
            if reviewer == view["requested_by"]:
                raise PlatformError(ErrorCode.RBAC_DENIED, "reviewer must differ from requester")
            payload = await self.state.peek_payload(tenant, approval_id)
            if payload is None:
                raise PlatformError(ErrorCode.INVALID_PARAMS, "approval payload is no longer available")
            self._precheck(tenant, view["tool"], payload)                     # raises STATE_CONFLICT / INVALID_PARAMS, stays PENDING
            await self.state.decide(tenant, approval_id, reviewer, approve=True)
            committed = self._apply(tenant, view["tool"], payload, approval_id, view["requested_by"])
            self.state.set_result(tenant, approval_id, committed)
            await self.state.append_audit(tenant, "action.committed", {"approval_id": approval_id, **{k: str(v) for k, v in committed.items() if k != "payload"}})
            return {"approval_id": approval_id, "status": "APPROVED", "committed": committed, "replayed": False}

    async def reject(self, tenant: str, approval_id: str, reviewer: str, reason: str = "") -> dict[str, Any]:
        async with self._locks[tenant]:
            await self.state.decide(tenant, approval_id, reviewer, approve=False, reason=reason)
            return {"approval_id": approval_id, "status": "REJECTED"}

    # ------------------------------------------------------------------
    def _precheck(self, tenant: str, tool: str, p: dict) -> None:
        if tool == "transition_workflow":
            e = self.state.get_entity(tenant, p["entity"])
            check_transition(p["entity"], e["state"] if e else None, p["from_state"], p["to_state"])
        elif tool == "submit_form":
            validate_form(p["form_id"], p["payload"])
            if self.state.get_entity(tenant, p["payload"]["policy_id"]) is None:
                raise PlatformError(ErrorCode.STATE_CONFLICT, "policy no longer exists")

    def _apply(self, tenant: str, tool: str, p: dict, approval_id: str, requester: str) -> dict[str, Any]:
        if tool == "transition_workflow":
            self.state.apply_transition(tenant, p["entity"], p["from_state"], p["to_state"], approval_id)
            return {"kind": "transition", "entity": p["entity"], "from": p["from_state"], "to": p["to_state"]}
        sid = self.state.record_submission(tenant, p["form_id"], p["payload"], approval_id)
        out: dict[str, Any] = {"kind": "form", "form_id": p["form_id"], "submission_id": sid}
        if p["form_id"] == "fnol":
            cid = self.state.new_claim_id(tenant)
            self.state.create_entity(tenant, cid, "SUBMITTED", requester)
            out["claim_id"] = cid
        return out
