"""V2 Isolation & cache (FR-01, FR-03; KPI-28 and the cache-scope assertions of §2.7).

Three tenants get one canary string each, planted in their own document. Concurrent traffic from every tenant
probes for the others' canaries. Cache scope is checked through the SSE `meta` event, which reports `cache` only on
a hit.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..stores import InMemoryCache
from .base import Env, Fault, Outcome, Validator
from .fixtures import golden_set, sse_events

LETTERS = ("a", "b", "c")


class TenantlessKeyCache(InMemoryCache):
    """Seeded fault: the exact-tier key drops tenant_id (`cache key drops tenant_id`)."""

    @staticmethod
    def key(tenant: str, user: str, norm_query: str) -> str:
        return InMemoryCache.key("tenant_shared", user, norm_query)


class SharedBucketCache(InMemoryCache):
    """Seeded fault: every tenant reads and writes one bucket (`cache tier ignores tenant scope`)."""

    async def get(self, tenant, user, query, embedding=None):
        return await super().get("tenant_shared", user, query, embedding)

    async def put(self, tenant, user, query, payload, embedding=None):
        return await super().put("tenant_shared", user, query, payload, embedding)


def _walk_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _walk_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _walk_strings(v)]
    return []


class V2Isolation(Validator):
    id = "V2"
    title = "Isolation & cache"
    features = "FR-01, FR-03"
    kpis = "KPI-27, KPI-28"
    faults = (
        Fault("cache_key_drops_tenant", "exact cache key drops tenant_id", build={"cache": TenantlessKeyCache()}),
        Fault("cache_shared_across_tenants", "cache reads/writes a single shared bucket",
              build={"cache": SharedBucketCache()}),
    )
    skip_reasons = ("KPI-27 needs the §1.4.4 production traffic mix, which this build does not define",)

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:
        out = Outcome()
        tenants = [f"{tenant}_{x}" for x in LETTERS]
        canary = {t: f"ZQCANARY{x.upper()}7749" for t, x in zip(tenants, LETTERS)}
        extra = {t: {"internal/codeword.md": f"The internal code word for this tenant is {canary[t]}."} for t in tenants}
        sb = await env.sandbox(tenants, fault, extra)
        try:
            qs = [g["question"] for g in golden_set() if g["answerable"]][:5] + ["What is the internal code word?"]
            reqs = [(t, {"X-Tenant-Id": t}, {"tenant_id": t, "user_id": "user_v2_shared", "query": q})
                    for t in tenants for q in qs]
            resps = await asyncio.gather(*(sb.client.post("/api/v1/agent/query", headers=h, json=b) for _, h, b in reqs))

            leaks = 0
            for (t, _, _), r in zip(reqs, resps):
                body = r.json()
                text = " ".join(_walk_strings(body))
                if body.get("tenant_id") != t:
                    out.check(False, f"response for {t} echoed tenant {body.get('tenant_id')}", severity="critical")
                for other, code in canary.items():
                    if other != t and code in text:
                        leaks += 1
                        out.check(False, f"canary of {other} appeared in a response for {t}", severity="critical",
                                  trace_id=body.get("trace_id"))
            out.metric("KPI-28 cross-tenant canary leaks", leaks, 0, "<=", len(reqs))

            # tenant token != body tenant -> -32001 (403)
            mism = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenants[0]},
                                        json={"tenant_id": tenants[1], "user_id": "user_v2_shared", "query": "x"})
            out.check(mism.status_code == 403 and (mism.json().get("error") or {}).get("code") == -32001,
                      f"header/body tenant mismatch gave HTTP {mism.status_code}, expected 403 / -32001")

            # cache scope: warm A, then B (same user) must miss, A again must hit, A with another user must miss
            q = [g["question"] for g in golden_set() if g["answerable"]][6]
            a, b = tenants[0], tenants[1]
            await self._ask(sb, a, "user_v2_scope1", q)
            ev_b = await self._ask(sb, b, "user_v2_scope1", q)
            out.check("cache" not in ev_b.get("meta", {}), "tenant B was served from tenant A's cache entry", severity="critical")
            ev_a = await self._ask(sb, a, "user_v2_scope1", q)
            out.check(ev_a.get("meta", {}).get("cache") is not None, "repeat question for the same tenant and user missed the cache")
            ev_u2 = await self._ask(sb, a, "user_v2_scope2", q)
            out.check("cache" not in ev_u2.get("meta", {}), "another user in the same tenant was served from a cache entry")
            return out
        finally:
            await sb.aclose()

    @staticmethod
    async def _ask(sb, tenant: str, user: str, query: str) -> dict[str, Any]:
        r = await sb.client.post("/api/v1/agent/query", headers={"X-Tenant-Id": tenant, "Accept": "text/event-stream"},
                                 json={"tenant_id": tenant, "user_id": user, "query": query})
        return sse_events(r.text)
