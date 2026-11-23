"""Transaction path (design.md FR-07, FR-09, KPI-19 analogue). Driven by data/insurance/golden/transactions.jsonl."""
import asyncio
import json
import pathlib
import uuid

import pytest
from fastapi.testclient import TestClient

from kestrel.approvals import ApprovalService
from kestrel.contracts import AgentQueryRequest, ErrorCode, PlatformError
from kestrel.execution import parse
from kestrel.forms import FORMS, redact_form, validate_form
from kestrel.gateway import create_app
from kestrel.triage import FixedTriage, RuleTriage
from kestrel.workflows import CLAIM_TRANSITIONS, POLICY_TRANSITIONS
from conftest import SETTINGS, ScriptedLLM, make_orch

G = pathlib.Path(__file__).resolve().parents[1] / "data/insurance/golden"
pytestmark = pytest.mark.skipif(not (G / "transactions.jsonl").exists(), reason="golden dataset not built")
TX = [json.loads(l) for l in open(G / "transactions.jsonl")] if (G / "transactions.jsonl").exists() else []
ENT = json.load(open(G / "entities.json")) if (G / "entities.json").exists() else {}
TABLES = json.load(open(G / "workflow_tables.json")) if (G / "workflow_tables.json").exists() else {}
T = "tenant_a"


def R(q, user="user_1", session=None, tenant=T):
    return AgentQueryRequest(tenant_id=tenant, user_id=user, query=q, session_id=session)


async def orch(**kw):
    return await make_orch(ScriptedLLM(), triage=FixedTriage("TRANSACTION_EXEC"), entities=ENT, **kw)


async def submit(o, q, user="user_1", session=None):
    h = await o.handle(R(q, user, session))
    return h, [(n, d) async for n, d in h.events]


def final(evs): return next(d for n, d in evs if n == "final")


# ---- the implementation's tables must equal the dataset's (which mirror the corpus workflow document)
def test_workflow_tables_match_the_dataset_and_corpus():
    assert {k: sorted(v) for k, v in CLAIM_TRANSITIONS.items()} == {k: sorted(v) for k, v in TABLES["claim"].items()}
    assert {k: sorted(v) for k, v in POLICY_TRANSITIONS.items()} == {k: sorted(v) for k, v in TABLES["policy"].items()}
    assert {k: list(v) for k, v in FORMS.items()} == TABLES["forms"] or {k: list(v) for k, v in FORMS.items()} == {k: list(v) for k, v in TABLES["forms"].items()}


# ---- every utterance in the dataset behaves as labelled
@pytest.mark.parametrize("case", TX, ids=[c["id"] for c in TX])
async def test_dataset_utterance(case):
    o = await orch()
    if case["valid"]:
        h, evs = await submit(o, case["utterance"])
        f = final(evs)
        assert h.path == "fast" and f["status"] == "PENDING_APPROVAL" and f["approval_id"] and f["route"] == "TRANSACTION_EXEC"
        a = await o.state.get_approval(T, f["approval_id"])
        assert a["status"] == "PENDING" and a["tool"] == case["tool"]
        if case["tool"] == "transition_workflow":
            assert a["args_redacted"] == case["args"]                                      # exact typed args incl. from_state
        else:
            assert a["args_redacted"]["form_id"] == case["args"]["form_id"]
        for k, (typed, fixed) in case["repaired"].items():
            assert [typed.lower(), fixed] == [a["repaired"][k][0].lower(), a["repaired"][k][1]]
        before = {k: o.state.get_entity(T, k)["state"] for k in ENT}
        assert before == ENT                                                                # nothing mutated by the request
    else:
        with pytest.raises(PlatformError) as e:
            await submit(o, case["utterance"])
        assert e.value.code.name == case["error_code"], (e.value.code.name, e.value.message)
        assert await o.state.list_pending(T) == []
        if case.get("missing"):
            assert set(case["missing"]) <= set(e.value.data["missing"])
        if case.get("bad"):
            assert set(case["bad"]) <= {b["field"] for b in e.value.data["bad"]}


# ---- parsing
def test_parse_transition_and_form():
    p = parse("Move claim 2 to under review")
    assert p.tool == "transition_workflow" and p.args == {"entity": "CLM-00002", "from_state": None, "to_state": "UNDER_REVIEW"}
    assert p.repaired == {"entity": ("claim 2", "CLM-00002")}
    p = parse("Submit an address change with policy_id=policy 100001, new_address=1 Main St Springfield, effective_date=2026-05-01")
    assert p.args["payload"]["policy_id"] == "POL-100001" and p.repaired["policy_id"] == ("policy 100001", "POL-100001")


@pytest.mark.parametrize("text", ["do something with it", "Move claim CLM-00002", "Update the thing", "submit it"])
def test_unparseable_requests_get_remediation(text):
    with pytest.raises(PlatformError) as e:
        parse(text)
    assert e.value.code == ErrorCode.INVALID_PARAMS and e.value.data["remediation"]


def test_form_validation_and_pii_redaction():
    ok = {"policy_id": "POL-100001", "new_address": "12 Elm Street Springfield", "effective_date": "2026-05-01"}
    assert validate_form("address_change", ok) == ok
    red = redact_form("address_change", ok)
    assert red["new_address"] == "[REDACTED:new_address]" and red["policy_id"] == "POL-100001"
    with pytest.raises(PlatformError) as e:
        validate_form("fnol", {"policy_id": "POL-100001", "loss_type": "meteor", "extra": "x"})
    d = e.value.data
    assert {"loss_date", "description", "estimated_amount"} <= set(d["missing"]) and {b["field"] for b in d["bad"]} >= {"loss_type", "extra"}


# ---- the Tier 3 invariant: nothing changes without an approved record (KPI-19 analogue)
async def test_request_stores_encrypted_payload_and_redacted_args_only():
    o = await orch()
    q = "Submit an address change form with policy_id=POL-100001, new_address=12 Elm Street Springfield, effective_date=2026-05-01"
    _, evs = await submit(o, q)
    aid = final(evs)["approval_id"]
    a = await o.state.get_approval(T, aid)
    assert "Elm Street" not in json.dumps(a) and "Elm Street" not in json.dumps(o.state.export_audit(T))
    assert o.state.vault_size(T) == 1
    assert (await o.state.peek_payload(T, aid))["payload"]["new_address"] == "12 Elm Street Springfield"


async def test_approve_commits_the_transition_exactly_once():
    o = await orch()
    _, evs = await submit(o, "Move claim CLM-00002 from UNDER_REVIEW to APPROVED")
    aid = final(evs)["approval_id"]
    assert o.state.get_entity(T, "CLM-00002")["state"] == "UNDER_REVIEW"
    r = await o.approvals.approve(T, aid, "reviewer_9")
    assert r["status"] == "APPROVED" and r["committed"] == {"kind": "transition", "entity": "CLM-00002", "from": "UNDER_REVIEW", "to": "APPROVED"}
    e = o.state.get_entity(T, "CLM-00002")
    assert e["state"] == "APPROVED" and e["history"][0]["approval_id"] == aid
    again = await o.approvals.approve(T, aid, "reviewer_9")                             # replay is idempotent
    assert again["replayed"] is True and len(o.state.get_entity(T, "CLM-00002")["history"]) == 1
    assert o.state.vault_size(T) == 0 and o.state.verify_chain(T) == (True, None)
    assert {"approval.requested", "approval.approved", "action.committed"} <= {ev["event_type"] for ev in o.state.export_audit(T)}


async def test_self_approval_is_blocked_and_nothing_changes():
    o = await orch()
    _, evs = await submit(o, "Move claim CLM-00001 to under review", user="user_1")
    aid = final(evs)["approval_id"]
    with pytest.raises(PlatformError) as e:
        await o.approvals.approve(T, aid, "user_1")
    assert e.value.code == ErrorCode.RBAC_DENIED and o.state.get_entity(T, "CLM-00001")["state"] == "SUBMITTED"
    assert (await o.state.get_approval(T, aid))["status"] == "PENDING"


async def test_reject_discards_payload_and_changes_nothing():
    o = await orch()
    _, evs = await submit(o, "Move claim CLM-00002 to DENIED")
    aid = final(evs)["approval_id"]
    await o.approvals.reject(T, aid, "reviewer_9", "not justified")
    assert o.state.get_entity(T, "CLM-00002")["state"] == "UNDER_REVIEW" and o.state.vault_size(T) == 0
    with pytest.raises(PlatformError):
        await o.approvals.approve(T, aid, "reviewer_9")                                 # cannot approve after reject


async def test_stale_request_fails_with_state_conflict_and_stays_pending():
    o = await orch()
    _, a = await submit(o, "Move claim CLM-00002 to APPROVED", user="user_1")
    _, b = await submit(o, "Move claim CLM-00002 to DENIED", user="user_2")
    aid_a, aid_b = final(a)["approval_id"], final(b)["approval_id"]
    await o.approvals.approve(T, aid_a, "reviewer_9")                                   # claim is now APPROVED
    with pytest.raises(PlatformError) as e:
        await o.approvals.approve(T, aid_b, "reviewer_9")
    assert e.value.code == ErrorCode.STATE_CONFLICT and e.value.data["current_state"] == "APPROVED"
    assert o.state.get_entity(T, "CLM-00002")["state"] == "APPROVED"
    assert (await o.state.get_approval(T, aid_b))["status"] == "PENDING"                # reviewer can still reject it
    await o.approvals.reject(T, aid_b, "reviewer_9", "superseded")


async def test_concurrent_conflicting_approvals_apply_exactly_one():
    o = await orch()
    ids = []
    for to, user in (("APPROVED", "user_1"), ("DENIED", "user_2"), ("INFO_REQUESTED", "user_3")):
        _, evs = await submit(o, f"Move claim CLM-00002 to {to}", user=user)
        ids.append(final(evs)["approval_id"])
    res = await asyncio.gather(*(o.approvals.approve(T, i, "reviewer_9") for i in ids), return_exceptions=True)
    ok = [r for r in res if not isinstance(r, Exception)]
    assert len(ok) == 1 and all(isinstance(r, PlatformError) and r.code == ErrorCode.STATE_CONFLICT for r in res if isinstance(r, Exception))
    assert len(o.state.get_entity(T, "CLM-00002")["history"]) == 1


async def test_fnol_approval_creates_a_new_claim_and_records_the_submission():
    o = await orch()
    _, evs = await submit(o, "File a claim for policy POL-100001 with loss_date=2026-03-02, loss_type=collision, description=Rear ended at a stoplight, estimated_amount=4200")
    aid = final(evs)["approval_id"]
    assert o.state.get_entity(T, "CLM-00007") is None
    r = await o.approvals.approve(T, aid, "reviewer_9")
    assert r["committed"]["claim_id"] == "CLM-00007" and o.state.get_entity(T, "CLM-00007")["state"] == "SUBMITTED"
    assert o.state.submissions(T)[0]["form_id"] == "fnol"


async def test_identical_pending_request_is_deduplicated_but_a_retry_after_rejection_is_new():
    o = await orch()
    _, a = await submit(o, "Move claim CLM-00001 to under review")
    _, b = await submit(o, "Move claim CLM-00001 to under review")
    assert final(a)["approval_id"] == final(b)["approval_id"] and len(await o.state.list_pending(T)) == 1
    await o.approvals.reject(T, final(a)["approval_id"], "reviewer_9")
    _, c = await submit(o, "Move claim CLM-00001 to under review")
    assert final(c)["approval_id"] != final(a)["approval_id"]


async def test_repeated_identical_invalid_requests_trip_the_breaker():
    o = await orch()
    seen = []
    for _ in range(4):
        with pytest.raises(PlatformError) as e:
            await o.execution.harness.call(caller="execution", tenant_id=T, session_id="s1", request_id="r1", tool="transition_workflow",
                                           args={"tenant_id": T, "entity": "CLM-00001", "from_state": "SUBMITTED", "to_state": "PAID"})
        seen.append(e.value.code)
    assert seen[:2] == [ErrorCode.INVALID_PARAMS] * 2 and seen[2:] == [ErrorCode.BREAKER_OPEN] * 2


async def test_tenant_isolation_of_entities_and_approvals():
    o = await make_orch(ScriptedLLM(), triage=FixedTriage("TRANSACTION_EXEC"), entities=ENT, tenants=("tenant_a", "tenant_b"))
    o.state._t["tenant_b"].entities.pop("CLM-00002")
    with pytest.raises(PlatformError):
        await o.handle(R("Move claim CLM-00002 to APPROVED", tenant="tenant_b"))
    _, evs = await submit(o, "Move claim CLM-00002 to APPROVED")
    aid = final(evs)["approval_id"]
    assert await o.state.get_approval("tenant_b", aid) is None
    with pytest.raises(PlatformError):
        await o.approvals.approve("tenant_b", aid, "reviewer_9")


# ---- triage on the labelled set (rules were adapted on these examples: dev-set numbers, not an unbiased KPI-04)
def test_rule_triage_on_labelled_set_reports_per_class_accuracy():
    rows = [json.loads(l) for l in open(G / "triage.jsonl")]
    tri, per = RuleTriage(), {}
    for r in rows:
        ok = tri.classify(r["text"]).route == r["label"]
        per.setdefault(r["label"], []).append(ok)
    acc = {k: sum(v) / len(v) for k, v in per.items()}
    print("rule triage per-class accuracy:", {k: round(v, 3) for k, v in acc.items()}, "overall", round(sum(sum(v) for v in per.values()) / len(rows), 3))
    assert acc["TRANSACTION_EXEC"] >= 0.95 and acc["HYBRID_RAG"] >= 0.95 and acc["DIRECT_ANSWER"] >= 0.9
    # REJECT_OOS accuracy is NOT asserted: the rules know only a few out-of-scope patterns, so KPI-04 (>= 97%) is NOT met
    # by this triage. A trained classifier is required. The user-visible safety net is tested below.


def test_greetings_are_chat_but_questions_with_a_greeting_are_not():
    t = RuleTriage()
    for s in ("Hi", "Thanks a lot", "Hello, can you help me?", "What can you help me with?", "Great, thank you"):
        assert t.classify(s).route == "DIRECT_ANSWER", s
    for s in ("Hi, what is the grace period?", "Thanks, how do I file a claim?", "Hello what is my deductible"):
        assert t.classify(s).route != "DIRECT_ANSWER", s


@pytest.mark.parametrize("path", ["unanswerable.jsonl"])
async def test_out_of_scope_and_unanswerable_never_reach_the_llm(path):
    """Safety net behind the weak OOS rules: with no support in the corpus the pipeline abstains/refuses deterministically
    (fake lexical reranker here; the real cross-encoder is checked in scripts/eval_golden.py)."""
    rows = [json.loads(l) for l in open(G / path)]
    llm = ScriptedLLM()
    o = await make_orch(llm)
    leaked = []
    for r in rows:
        h = await o.handle(R(r["question"], user=f"user_{r['id']}"))
        evs = [(n, d) async for n, d in h.events]
        f = final(evs)
        if f["status"] not in ("ABSTAINED", "REFUSED") or llm.calls:
            leaked.append((r["id"], r["question"], f["status"]))
            llm.calls.clear()
    rate = 1 - len(leaked) / len(rows)
    print(f"{path}: safely handled {rate:.1%}; leaked: {leaked[:5]}")
    assert rate >= 0.95, leaked


# ---- over HTTP: roles, headers, status codes
def _client(o):
    return TestClient(create_app(SETTINGS, ScriptedLLM(), o))


def test_http_end_to_end_request_review_approve():
    o = asyncio.run(orch())
    c = _client(o)
    h = {"X-Tenant-Id": T}
    r = c.post("/api/v1/agent/query", headers={**h, "X-User-Id": "user_1"},
               json={"tenant_id": T, "user_id": "user_1", "query": "Move claim 2 to APPROVED"})
    assert r.status_code == 202 and r.headers["x-path"] == "fast"
    f = r.json(); aid = f["approval_id"]
    assert f["status"] == "PENDING_APPROVAL" and "CLM-00002" in f["answer"] and f["remediation"].startswith("Please confirm")
    lst = c.get("/api/v1/approvals", headers=h).json()["approvals"]                               # tenant-scoped, no role gate
    assert [a["approval_id"] for a in lst] == [aid]
    assert c.get(f"/api/v1/approvals/{aid}", headers={**h, "X-User-Id": "user_2"}).status_code == 200   # no role gate on view
    rv = dict(h)
    assert c.post(f"/api/v1/approvals/{aid}/approve", headers=rv).status_code == 422                # X-User-Id missing
    assert c.post(f"/api/v1/approvals/{aid}/approve", headers={**rv, "X-User-Id": "user_1"}).status_code == 403  # self-approval
    ok = c.post(f"/api/v1/approvals/{aid}/approve", headers={**rv, "X-User-Id": "reviewer_9"})
    assert ok.status_code == 200 and ok.json()["committed"]["to"] == "APPROVED"
    assert c.post(f"/api/v1/approvals/{aid}/reject", headers={**rv, "X-User-Id": "reviewer_9"}).status_code == 409
    assert o.state.get_entity(T, "CLM-00002")["state"] == "APPROVED"


def test_http_validation_errors_are_422_and_conflicts_409():
    o = asyncio.run(orch())
    c = _client(o)
    body = lambda q: {"tenant_id": T, "user_id": "user_1", "query": q}
    h = {"X-Tenant-Id": T}
    r = c.post("/api/v1/agent/query", headers=h, json=body("Move claim CLM-00001 to APPROVED"))
    assert r.status_code == 422 and "allowed" in r.json()["error"]["data"]
    r = c.post("/api/v1/agent/query", headers=h, json=body("Move claim CLM-00002 from SUBMITTED to UNDER_REVIEW"))
    assert r.status_code == 409 and r.json()["error"]["code"] == ErrorCode.STATE_CONFLICT
