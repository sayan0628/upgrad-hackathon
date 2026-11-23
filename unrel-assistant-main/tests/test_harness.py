import asyncio
import uuid

import pytest

from kestrel.contracts import ErrorCode, PlatformError
from kestrel.harness import (ALLOWED, TOOLS, CircuitBreaker, IdempotencyStore, ToolExecutionHarness,
                               canonical_args_hash, repair_identifier)

T = "tenant_a"
S = "0b6f6a2e-1c3d-4a5b-8c7d-9e0f1a2b3c4d"


def make(dispatch=None, breaker=None):
    calls = []

    async def default(tool, args):
        calls.append((tool, args))
        return {"ok": True, "tool": tool}

    return ToolExecutionHarness(dispatch or default, breaker=breaker), calls


async def call(h, tool, args, caller="execution", req="r1", **kw):
    return await h.call(caller=caller, tenant_id=T, session_id=S, request_id=req, tool=tool, args=args, **kw)


@pytest.mark.parametrize("raw,fixed", [("SKU99", "SKU-099"), ("sku 7", "SKU-007"), ("lot 44", "LOT-00044"),
                                       ("LOT-00044", "LOT-00044"), ("SKU-1234", "SKU-1234"), ("widget", "widget")])
def test_repair_identifier(raw, fixed):
    assert repair_identifier(raw) == fixed


async def test_t3_repair_is_flagged_not_silent():
    h, calls = make()
    r = await call(h, "transition_workflow", {"tenant_id": T, "entity": "SKU99", "from_state": "A", "to_state": "B"})
    assert r.repaired == {"entity": ("SKU99", "SKU-099")}
    assert calls[0][1]["entity"] == "SKU-099"
    assert h.metrics.repaired_ok == 1 and h.metrics.first_pass_valid == 0


async def test_unrepairable_args_are_invalid_params():
    h, _ = make()
    with pytest.raises(PlatformError) as e:
        await call(h, "transition_workflow", {"tenant_id": T, "entity": "banana", "from_state": "A", "to_state": "B"})
    assert e.value.code == ErrorCode.INVALID_PARAMS


async def test_extra_fields_forbidden():
    h, _ = make()
    with pytest.raises(PlatformError) as e:
        await call(h, "submit_form", {"tenant_id": T, "form_id": "f1", "payload": {}, "idempotency_key": str(uuid.uuid4())})
    assert e.value.code == ErrorCode.INVALID_PARAMS      # keys are harness-minted, callers cannot supply them


async def test_unknown_tool():
    h, _ = make()
    with pytest.raises(PlatformError) as e:
        await call(h, "drop_database", {})
    assert e.value.code == ErrorCode.METHOD_NOT_FOUND


async def test_tenant_mismatch_between_tool_args_and_session():
    h, _ = make()
    with pytest.raises(PlatformError) as e:
        await call(h, "submit_form", {"tenant_id": "tenant_b", "form_id": "f", "payload": {}})
    assert e.value.code == ErrorCode.TENANT_MISMATCH


@pytest.mark.parametrize("tool", sorted(TOOLS))
async def test_every_tool_has_a_permitted_caller(tool):
    assert any(tool in tools for tools in ALLOWED.values()), f"{tool} has no legal caller"


async def test_rbac_matrix_denies_cross_caller():
    h, calls = make()
    for caller, tool, args in [
        ("retrieval", "submit_form", {"tenant_id": T, "form_id": "f", "payload": {}}),
        ("verification", "hybrid_retrieve", {"tenant_id": T, "query": "q"}),
        ("execution", "hybrid_retrieve", {"tenant_id": T, "query": "q"}),
        ("retrieval", "put_session_fact", {"tenant_id": T, "session_id": S, "fact": "x", "source_chunk_id": "a" * 16}),
    ]:
        with pytest.raises(PlatformError) as e:
            await call(h, tool, args, caller=caller)
        assert e.value.code == ErrorCode.RBAC_DENIED
    assert calls == []


async def test_t1_has_no_idempotency_key_t3_does_and_replays():
    h, calls = make()
    r1 = await call(h, "hybrid_retrieve", {"tenant_id": T, "query": "q"}, caller="retrieval")
    assert r1.idempotency_key is None
    args = {"tenant_id": T, "form_id": "f1", "payload": {"a": "b"}}
    a = await call(h, "submit_form", args)
    b = await call(h, "submit_form", args, req="r2")
    assert a.idempotency_key == b.idempotency_key and not a.replayed and b.replayed
    assert len([c for c in calls if c[0] == "submit_form"]) == 1     # executed once


async def test_same_key_different_args_conflicts():
    h, _ = make()
    a = await call(h, "submit_form", {"tenant_id": T, "form_id": "f1", "payload": {}})
    with pytest.raises(PlatformError) as e:
        await call(h, "submit_form", {"tenant_id": T, "form_id": "f2", "payload": {}}, idempotency_key=a.idempotency_key)
    assert e.value.code == ErrorCode.IDEMPOTENCY_CONFLICT


def test_args_hash_excludes_keys_and_includes_tenant():
    a = canonical_args_hash("t1", "x", {"q": 1})
    assert a == canonical_args_hash("t1", "x", {"q": 1})
    assert a != canonical_args_hash("t2", "x", {"q": 1})


async def test_breaker_opens_after_two_identical_failures_and_persists_in_session():
    async def boom(tool, args):
        raise RuntimeError("downstream")

    clock = [0.0]
    h, _ = make(boom, CircuitBreaker(clock=lambda: clock[0]))
    args = {"tenant_id": T, "query": "SKU-099"}
    for _ in range(2):
        with pytest.raises(PlatformError) as e:
            await call(h, "hybrid_retrieve", args, caller="retrieval")
        assert e.value.code == ErrorCode.INVALID_REQUEST
    with pytest.raises(PlatformError) as e:
        await call(h, "hybrid_retrieve", args, caller="retrieval")
    assert e.value.code == ErrorCode.BREAKER_OPEN
    h.breaker.end_request("r1")
    with pytest.raises(PlatformError) as e:                      # new request, same session, inside the 60 s TTL
        await call(h, "hybrid_retrieve", args, caller="retrieval", req="r2")
    assert e.value.code == ErrorCode.BREAKER_OPEN
    clock[0] = 61.0                                              # TTL lapsed: closes (no half-open in v1)
    with pytest.raises(PlatformError) as e:
        await call(h, "hybrid_retrieve", args, caller="retrieval", req="r3")
    assert e.value.code == ErrorCode.INVALID_REQUEST


async def test_breaker_is_per_tenant():
    async def boom(tool, args):
        raise RuntimeError

    h, _ = make(boom)
    for _ in range(2):
        with pytest.raises(PlatformError):
            await call(h, "hybrid_retrieve", {"tenant_id": T, "query": "x"}, caller="retrieval")
    with pytest.raises(PlatformError) as e:      # other tenant, same text: not blocked by tenant_a's breaker
        await h.call(caller="retrieval", tenant_id="tenant_b", session_id=S, request_id="rb", tool="hybrid_retrieve",
                     args={"tenant_id": "tenant_b", "query": "x"})
    assert e.value.code == ErrorCode.INVALID_REQUEST


async def test_dispatch_timeout_is_mcp_timeout_and_counts_as_failure():
    async def slow(tool, args):
        await asyncio.sleep(1)

    h, _ = make(slow)
    with pytest.raises(PlatformError) as e:
        await call(h, "resolve_anchor", {"tenant_id": T, "query": "q"}, caller="retrieval")   # 40 ms budget
    assert e.value.code == ErrorCode.MCP_TIMEOUT
