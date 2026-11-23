import asyncio
import random
import uuid

import numpy as np
import pytest

from kestrel.contracts import ErrorCode, PlatformError
from kestrel.stores import ApprovalStatus, InMemoryCache, InMemoryStateStore, normalise_query


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def vec(*xs): return np.array(xs, dtype=float)


# ---------------- cache
def test_normalise_query():
    assert normalise_query("  What IS   the policy?? ") == "what is the policy"


async def test_exact_hit_miss_and_user_scoping():
    c = InMemoryCache()
    await c.put("tenant_a", "user_1", "What is the policy?", {"a": 1})
    assert (await c.get("tenant_a", "user_1", "what is the policy")).tier == "exact"
    assert await c.get("tenant_a", "user_2", "what is the policy") is None      # per-user scope
    assert await c.get("tenant_b", "user_1", "what is the policy") is None      # per-tenant scope


def test_keys_are_tenant_hash_tagged_and_differ_by_tenant():
    k1, k2 = InMemoryCache.key("tenant_a", "u", "q"), InMemoryCache.key("tenant_b", "u", "q")
    assert k1.startswith("{tenant_a}:cache:") and k1 != k2


async def test_ttl_has_jitter_and_expires():
    clk = Clock()
    c = InMemoryCache(clock=clk, rng=random.Random(7))
    ttls = {round(await c.put("tenant_a", "u", f"q{i}", i), 3) for i in range(20)}
    assert all(255 <= t <= 345 for t in ttls) and len(ttls) > 10
    await c.put("tenant_a", "u", "x", 1)
    clk.t += 346
    assert await c.get("tenant_a", "u", "x") is None


async def test_semantic_tier_threshold_and_isolation():
    c = InMemoryCache()
    await c.put("tenant_a", "u1", "refund policy", "ans", embedding=vec(1, 0, 0))
    hit = await c.get("tenant_a", "u1", "how do refunds work", embedding=vec(0.99, 0.05, 0))
    assert hit and hit.tier == "semantic" and hit.score >= 0.92
    assert await c.get("tenant_a", "u1", "unrelated", embedding=vec(0, 1, 0)) is None            # below threshold
    assert await c.get("tenant_b", "u1", "how do refunds work", embedding=vec(0.99, 0.05, 0)) is None   # other tenant
    assert await c.get("tenant_a", "u2", "how do refunds work", embedding=vec(0.99, 0.05, 0)) is None   # other user


async def test_single_flight_coalesces_a_stampede():
    c, calls = InMemoryCache(), 0

    async def slow():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return "v"

    res = await asyncio.gather(*(c.single_flight("tenant_a", "u", "same q", slow) for _ in range(25)))
    assert res == ["v"] * 25 and calls == 1 and c.stats["coalesced"] == 24


async def test_single_flight_propagates_errors_and_recovers():
    c = InMemoryCache()

    async def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        await c.single_flight("tenant_a", "u", "q", boom)

    async def ok():
        return 1

    assert await c.single_flight("tenant_a", "u", "q", ok) == 1


async def test_flush_tenant_only_touches_that_tenant():
    c = InMemoryCache()
    await c.put("tenant_a", "u", "q", 1)
    await c.put("tenant_b", "u", "q", 2)
    assert c.flush_tenant("tenant_a") == 1
    assert await c.get("tenant_b", "u", "q") is not None


async def test_hot_window_keeps_last_three_turns_per_session():
    c = InMemoryCache()
    for i in range(5):
        await c.push_turn("tenant_a", "s1", {"i": i})
    assert [t["i"] for t in await c.window("tenant_a", "s1")] == [2, 3, 4]
    assert await c.window("tenant_b", "s1") == []


async def test_cross_tenant_leak_probe_under_concurrency():          # KPI-28 analogue
    c = InMemoryCache()

    async def worker(i):
        tenant = f"tenant_{i % 5}"
        await c.put(tenant, "user_1", "shared question", f"secret-of-{tenant}")
        hit = await c.get(tenant, "user_1", "shared question")
        return tenant, hit.payload

    for tenant, payload in await asyncio.gather(*(worker(i) for i in range(200))):
        assert payload == f"secret-of-{tenant}"


# ---------------- state store
async def test_only_verified_facts_and_tenant_scoping():
    s = InMemoryStateStore()
    with pytest.raises(PlatformError):
        await s.put_fact("tenant_a", "s1", "x", "a" * 16, verified=False)
    await s.put_fact("tenant_a", "s1", "SKU-099 is recalled", "a" * 16, verified=True)
    assert len(await s.get_facts("tenant_a", "s1")) == 1
    assert await s.get_facts("tenant_b", "s1") == []


async def test_audit_chain_is_per_tenant_and_detects_tampering():
    s = InMemoryStateStore()
    for i in range(5):
        await s.append_audit("tenant_a", "evt", {"i": str(i)})
    await s.append_audit("tenant_b", "evt", {"i": "0"})
    assert s.verify_chain("tenant_a") == (True, None) and s.anchor("tenant_a")["seq"] == 5
    assert s.anchor("tenant_b")["seq"] == 1                              # independent chain, seq per tenant
    s._t["tenant_a"].audit[2]["payload_redacted"]["i"] = "tampered"
    assert s.verify_chain("tenant_a") == (False, 3)
    assert s.verify_chain("tenant_b") == (True, None)


async def test_audit_concurrent_appends_keep_a_valid_chain():
    s = InMemoryStateStore()
    await asyncio.gather(*(s.append_audit("tenant_a", "evt", {"n": str(i)}) for i in range(100)))
    assert s.verify_chain("tenant_a") == (True, None) and s.anchor("tenant_a")["seq"] == 100


async def test_audit_refuses_raw_pii():
    s = InMemoryStateStore()
    with pytest.raises(PlatformError) as e:
        await s.append_audit("tenant_a", "evt", {"note": "contact jane@example.com"})
    assert e.value.code == ErrorCode.PII_BLOCKED and s.anchor("tenant_a")["seq"] == 0


async def _approval(s, tenant="tenant_a", key=None, by="user_1"):
    return await s.create_approval(tenant, "submit_form", {"form_id": "f1"}, {"form_id": "f1", "email": "jane@example.com"},
                                   key or uuid.uuid4(), by, {"entity": ("SKU99", "SKU-099")})


async def test_approval_lifecycle_vault_and_reviewer_rule():
    s = InMemoryStateStore()
    aid = await _approval(s)
    assert s.vault_size("tenant_a") == 1
    assert (await s.get_approval("tenant_a", aid))["repaired"] == {"entity": ("SKU99", "SKU-099")}
    with pytest.raises(PlatformError) as e:                                  # self-approval
        await s.decide("tenant_a", aid, "user_1", True)
    assert e.value.code == ErrorCode.RBAC_DENIED
    payload = await s.decide("tenant_a", aid, "reviewer_9", True, "ok")
    assert payload["email"] == "jane@example.com"                            # real payload available at commit
    assert s.vault_size("tenant_a") == 0                                     # ...and deleted afterwards
    with pytest.raises(PlatformError):                                       # cannot decide twice
        await s.decide("tenant_a", aid, "reviewer_9", True)
    assert (await s.get_approval("tenant_a", aid))["status"] == "APPROVED"


async def test_reject_deletes_payload_and_returns_none():
    s = InMemoryStateStore()
    aid = await _approval(s)
    assert await s.decide("tenant_a", aid, "reviewer_9", False, "no") is None
    assert s.vault_size("tenant_a") == 0 and (await s.get_approval("tenant_a", aid))["status"] == "REJECTED"


async def test_approval_idempotency_unique_per_tenant_and_key():
    s = InMemoryStateStore()
    k = uuid.uuid4()
    a1, a2 = await _approval(s, key=k), await _approval(s, key=k)
    assert a1 == a2 and len(await s.list_pending("tenant_a")) == 1
    assert await _approval(s, tenant="tenant_b", key=k) != a1                # same key, other tenant: distinct


async def test_approvals_expire_and_are_never_auto_approved():
    clk = Clock()
    s = InMemoryStateStore(clock=clk, approval_ttl_s=60)
    aid = await _approval(s)
    clk.t += 61
    assert (await s.get_approval("tenant_a", aid))["status"] == ApprovalStatus.EXPIRED.value
    assert s.vault_size("tenant_a") == 0
    with pytest.raises(PlatformError):
        await s.decide("tenant_a", aid, "reviewer_9", True)


async def test_approvals_are_tenant_scoped():
    s = InMemoryStateStore()
    aid = await _approval(s)
    assert await s.get_approval("tenant_b", aid) is None and await s.list_pending("tenant_b") == []
    with pytest.raises(PlatformError):
        await s.decide("tenant_b", aid, "reviewer_9", True)


async def test_chunks_are_tenant_scoped():
    s = InMemoryStateStore()
    await s.put_chunk("tenant_a", "c" * 16, "text")
    assert (await s.get_chunk("tenant_a", "c" * 16))["text"] == "text" and await s.get_chunk("tenant_b", "c" * 16) is None
