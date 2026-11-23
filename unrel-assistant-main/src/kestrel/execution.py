"""Execution subagent for TRANSACTION_EXEC (design.md §2.2, §1.2.5, FR-07/FR-09).

Deterministic by design: a natural-language request is parsed with a small grammar into a typed tool call. No LLM
decides what gets mutated. The call goes through the ToolExecutionHarness (strict validation, repair, caller x tool
RBAC, harness-minted idempotency, breaker). Tier 3 tools NEVER mutate: they validate and then a PENDING_APPROVAL
record is written; only a reviewer's approval (approvals.py) applies the change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .contracts import AgentQueryRequest, ErrorCode, PlatformError
from .forms import FORMS, redact_form, validate_form
from .harness import ToolExecutionHarness, repair_identifier
from .stores import InMemoryStateStore
from .workflows import check_transition, machine_for, normalise_state

_ENTITY = re.compile(r"(?i)\b(?:(?:clm|pol)[-_ ]?\d+|(?:claim|policy)\s*#?\s*\d+)\b")
_FROM_TO = re.compile(r"(?i)\bfrom\s+(?P<frm>[A-Za-z_ ]+?)\s+to\s+(?P<to>[A-Za-z_ ]+?)\s*[.!?]?\s*$")
_TO = re.compile(r"(?i)\bto\s+(?P<to>[A-Za-z_ ]+?)\s*[.!?]?\s*$")
_MOVE = re.compile(r"(?i)\b(move|transition|change|set|update|mark)\b")
_PAIRS = re.compile(r"(?:^|,\s*)([a-z_]+)=(.*?)(?=,\s*[a-z_]+=|$)", re.S)
_FORM_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fnol", re.compile(r"(?i)\b(first notice of loss|fnol|file (?:a |the )?claim|submit (?:a |the )?claim)\b")),
    ("address_change", re.compile(r"(?i)\baddress change\b")),
    ("beneficiary_change", re.compile(r"(?i)\bbeneficiary change\b")),
    ("cancellation_request", re.compile(r"(?i)\bcancellation request\b")),
)


@dataclass
class Parsed:
    tool: str
    args: dict[str, Any]
    summary: str
    repaired: dict[str, tuple[str, str]] = field(default_factory=dict)


def _bad(msg: str, **data) -> PlatformError:
    return PlatformError(ErrorCode.INVALID_PARAMS, msg, {
        "remediation": ("I can: (1) move a claim or policy to a new state, e.g. 'Move claim CLM-00002 to APPROVED'; "
                        "(2) submit a form: " + ", ".join(sorted(FORMS)) + " with field=value pairs."), **data})


def parse(text: str) -> Parsed:
    """Pure function: text -> typed tool call (or INVALID_PARAMS with remediation)."""
    for form_id, rx in _FORM_WORDS:
        if rx.search(text) and re.search(r"(?i)\bwith\b", text):
            return _parse_form(form_id, text)
    if m := _ENTITY.search(text):
        if not _MOVE.search(text):
            raise _bad("I could not tell what to do with that record")
        raw = m.group(0)
        entity = repair_identifier(raw)
        rest = text[m.end():]
        ft = _FROM_TO.search(rest)
        if ft:
            frm, to = normalise_state(ft.group("frm")), normalise_state(ft.group("to"))
        elif t := _TO.search(rest):
            frm, to = None, normalise_state(t.group("to"))
        else:
            raise _bad("Which state should it move to?", example="Move claim CLM-00002 to APPROVED")
        args = {"entity": entity, "from_state": frm, "to_state": to}
        rep = {"entity": (raw, entity)} if entity != raw else {}
        return Parsed("transition_workflow", args, f"move {entity} " + (f"from {frm} " if frm else "") + f"to {to}", rep)
    raise _bad("I could not understand that request")


def _parse_form(form_id: str, text: str) -> Parsed:
    head, _, tail = re.split(r"(?i)\bwith\b", text, maxsplit=1)[0], None, re.split(r"(?i)\bwith\b", text, maxsplit=1)[1]
    payload = {k.strip().lower(): v.strip().rstrip(".") for k, v in _PAIRS.findall(tail.strip())}
    rep: dict[str, tuple[str, str]] = {}
    if "policy_id" not in payload and (m := _ENTITY.search(head)):
        payload["policy_id"] = m.group(0)
    if "policy_id" in payload:
        fixed = repair_identifier(payload["policy_id"])
        if fixed != payload["policy_id"]:
            rep["policy_id"] = (payload["policy_id"], fixed)
            payload["policy_id"] = fixed
    return Parsed("submit_form", {"form_id": form_id, "payload": payload}, f"submit a {form_id.replace('_', ' ')} form", rep)


class ExecutionBackend:
    """The harness dispatch target: VALIDATES against live state, never mutates (Tier 3 -> PENDING_APPROVAL)."""

    def __init__(self, state: InMemoryStateStore):
        self.state = state

    async def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        tenant = args["tenant_id"]
        if tool == "transition_workflow":
            e = self.state.get_entity(tenant, args["entity"])
            check_transition(args["entity"], e["state"] if e else None, args["from_state"], args["to_state"])
            return {"validated": True, "entity": args["entity"], "from": args["from_state"], "to": args["to_state"]}
        if tool == "submit_form":
            clean = validate_form(args["form_id"], args["payload"])
            if self.state.get_entity(tenant, clean["policy_id"]) is None:
                raise PlatformError(ErrorCode.INVALID_PARAMS, f"unknown policy {clean['policy_id']}", {"bad": [{"field": "policy_id", "reason": "unknown policy"}]})
            return {"validated": True, "form_id": args["form_id"], "payload": clean}
        raise PlatformError(ErrorCode.METHOD_NOT_FOUND, f"{tool} is not an execution tool")


class ExecutionAgent:
    def __init__(self, harness: ToolExecutionHarness, state: InMemoryStateStore):
        self.harness, self.state = harness, state

    async def request(self, req: AgentQueryRequest, text: str, request_id: str) -> dict[str, Any]:
        t, u = req.tenant_id, req.user_id
        p = parse(text)
        args = dict(p.args)
        if p.tool == "transition_workflow" and args["from_state"] is None:      # read current state (platform read, not a model tool)
            e = self.state.get_entity(t, args["entity"])
            if e is None:
                raise PlatformError(ErrorCode.INVALID_PARAMS, f"unknown entity {args['entity']}", {"remediation": "Check the claim or policy number."})
            args["from_state"] = e["state"]
        args["tenant_id"] = t
        for attempt in range(5):                                       # a rejected/approved earlier request must not block a new one
            r = await self.harness.call(caller="execution", tenant_id=t, session_id=f"{req.session_id or 'none'}#{attempt}",
                                        request_id=request_id, tool=p.tool, args=args)
            existing = await self.state.get_approval_by_key(t, r.idempotency_key)  # type: ignore[arg-type]
            if existing is None or existing["status"] == "PENDING":
                break
        else:
            raise PlatformError(ErrorCode.IDEMPOTENCY_CONFLICT, "too many identical requests")
        payload = r.value if p.tool == "transition_workflow" else {"form_id": args["form_id"], "payload": r.value["payload"]}
        stored = {k: v for k, v in args.items() if k != "tenant_id"} if p.tool == "transition_workflow" else payload
        red = (stored if p.tool == "transition_workflow" else {"form_id": args["form_id"], "payload": redact_form(args["form_id"], stored["payload"])})
        repaired = {**p.repaired, **{k: tuple(v) for k, v in r.repaired.items()}}
        aid = await self.state.create_approval(t, p.tool, red, stored, r.idempotency_key, u, repaired)  # type: ignore[arg-type]
        return {"approval_id": aid, "summary": p.summary, "repaired": {k: list(v) for k, v in repaired.items()},
                "deduplicated": existing is not None}
