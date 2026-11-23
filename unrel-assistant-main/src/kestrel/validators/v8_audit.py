"""V8 Audit chain (FR-11; KPI-19 audit-integrity side).

Checks, per validator tenant:
  * the store's own verify_chain passes, sequence numbers are contiguous, and the anchor matches the exported head;
  * an INDEPENDENT verifier (re-implemented here from the design's chain definition, D-15: SHA-256 over
    prev_hash + canonical JSON of {tenant, seq, type, payload}) accepts the exported chain;
  * the number of audit events equals the number of actions the validator caused (no silent drops);
  * a tamper canary: a modified copy of one event is rejected by the independent verifier.

Seeded faults: an append path that silently drops events, and a hash that ignores the payload.
"""
from __future__ import annotations

import hashlib
import json

from .base import Env, Fault, Outcome, Validator
from .fixtures import golden_set

GENESIS = "0" * 64


def _independent_verify(events: list[dict]) -> int | None:
    """Return the first bad seq, or None if the chain verifies. Written from the spec, not from the store."""
    prev = GENESIS
    for ev in events:
        blob = json.dumps({"tenant": ev["tenant_id"], "seq": ev["seq"], "type": ev["event_type"],
                           "payload": ev["payload_redacted"]}, sort_keys=True, separators=(",", ":"))
        expect = hashlib.sha256((prev + blob).encode()).hexdigest()
        if ev["prev_hash"] != prev or ev["hash"] != expect:
            return ev["seq"]
        prev = ev["hash"]
    return None


def _drop_every_third_append(orch) -> None:
    real = orch.state.append_audit
    count = {"n": 0}

    async def append_audit(tenant, event_type, payload_redacted):
        count["n"] += 1
        if count["n"] % 3 == 0:                       # silently dropped
            return {"seq": None, "dropped": True}
        return await real(tenant, event_type, payload_redacted)
    orch.state.append_audit = append_audit


def _payload_not_hashed(orch) -> None:
    orch.state._hash = staticmethod(lambda prev, tenant, seq, event_type, payload: hashlib.sha256(
        f"{prev}{tenant}{seq}{event_type}".encode()).hexdigest())


class V8Audit(Validator):
    id = "V8"
    title = "Audit chain"
    features = "FR-11"
    kpis = "KPI-19 (audit side)"
    faults = (
        Fault("audit_event_dropped", "every third audit append is silently dropped", post=_drop_every_third_append),
        Fault("hash_ignores_payload", "chain hash covers seq/type but not the payload", post=_payload_not_hashed),
    )

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        sb = await env.sandbox(tenant, fault)
        try:
            qs = [g["question"] for g in golden_set() if g["answerable"]][:5]
            for i, q in enumerate(qs):
                r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                         json={"tenant_id": tenant, "user_id": f"user_v8{i}", "query": q})
                out.check(r.status_code == 200, f"query {i} returned HTTP {r.status_code}")
            # a valid Tier 3 move (CLM-00002 is UNDER_REVIEW -> APPROVED), then a reviewer approval that commits it
            p = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant},
                                     json={"tenant_id": tenant, "user_id": "user_v8req",
                                           "query": "Move claim CLM-00002 to APPROVED"})
            aid = (p.json() or {}).get("approval_id")
            a = await sb.client.post(f"/api/v1/approvals/{aid}/approve",
                                     headers={"X-Tenant-Id": tenant, "X-User-Id": "user_v8rev"})
            out.check(a.status_code == 200, f"reviewer approval returned HTTP {a.status_code}")
            expected_queries = len(qs) + 1          # every answered request writes one query.completed
            expected_commits = 1 if a.status_code == 200 else 0

            events = sb.state.export_audit(tenant)
            out.check(sb.state.verify_chain(tenant)[0], "store verify_chain failed on the validator tenant", severity="critical")
            out.check([e["seq"] for e in events] == list(range(1, len(events) + 1)), "audit sequence has gaps",
                      severity="critical")
            anchor = sb.state.anchor(tenant)
            out.check(anchor["seq"] == len(events) and (not events or anchor["head_hash"] == events[-1]["hash"]),
                      "anchor does not match the exported head", severity="critical")
            bad = _independent_verify(events)
            out.check(bad is None, f"independent verifier rejects the chain at seq {bad}", severity="critical")
            queries = sum(1 for e in events if e["event_type"] == "query.completed")
            commits = sum(1 for e in events if e["event_type"] == "action.committed")
            out.check(queries == expected_queries, f"expected {expected_queries} query.completed events, found {queries}",
                      severity="critical")
            out.check(commits == expected_commits, f"expected {expected_commits} action.committed events, found {commits}",
                      severity="critical")

            if events:
                tampered = [dict(e) for e in events]
                tampered[len(tampered) // 2]["payload_redacted"] = {"tampered": "yes"}
                out.check(_independent_verify(tampered) is not None, "tampered copy was accepted by the verifier",
                          severity="critical")
            return out
        finally:
            await sb.aclose()
