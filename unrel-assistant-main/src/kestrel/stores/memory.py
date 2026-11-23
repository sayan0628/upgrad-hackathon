"""In-memory stand-ins for Redis (cache + hot window) and Postgres (facts, approvals, audit, chunks)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

import numpy as np
from cryptography.fernet import Fernet

from ..contracts import ErrorCode, PlatformError
from ..plugins import PiiRedactor, PreContext

Clock = Callable[[], float]
GENESIS = "0" * 64


def normalise_query(q: str) -> str:
    """Case-fold, collapse whitespace, strip trailing punctuation: the `norm_query` of the cache key."""
    return re.sub(r"[\s]+", " ", q.strip().lower()).rstrip("?.! ")


@dataclass
class CacheHit:
    payload: Any
    tier: str                    # "exact" | "semantic"
    score: float = 1.0


@dataclass
class _Entry:
    payload: Any
    expires: float
    user: str
    embedding: np.ndarray | None = None


class InMemoryCache:
    """Redis stand-in. Every key is `{tenant}:cache:<sha256(tenant:user:norm_query)>` (hash-tag style); the
    semantic tier only ever scans the caller's own (tenant, user) entries. TTL = 300 s ± 45 s jitter."""

    def __init__(self, clock: Clock = time.monotonic, rng: random.Random | None = None,
                 semantic_threshold: float = 0.92, ttl_s: float = 300.0, jitter_s: float = 45.0):
        self._clock, self._rng = clock, rng or random.Random()
        self.threshold, self.ttl_s, self.jitter_s = semantic_threshold, ttl_s, jitter_s
        self._kv: dict[str, _Entry] = {}
        self._hot: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=3))
        self._inflight: dict[str, asyncio.Future] = {}
        self.stats = {"hit_exact": 0, "hit_semantic": 0, "miss": 0, "coalesced": 0}

    @staticmethod
    def key(tenant: str, user: str, norm_query: str) -> str:
        digest = hashlib.sha256(f"{tenant}:{user}:{norm_query}".encode()).hexdigest()
        return f"{{{tenant}}}:cache:{digest}"

    def _live(self, k: str) -> _Entry | None:
        e = self._kv.get(k)
        if e and e.expires <= self._clock():
            del self._kv[k]
            return None
        return e

    async def get(self, tenant: str, user: str, query: str, embedding: np.ndarray | None = None) -> CacheHit | None:
        nq = normalise_query(query)
        if e := self._live(self.key(tenant, user, nq)):
            self.stats["hit_exact"] += 1
            return CacheHit(e.payload, "exact")
        if embedding is not None:
            prefix, best, best_s = f"{{{tenant}}}:cache:", None, 0.0
            q = embedding / (np.linalg.norm(embedding) or 1.0)
            for k in [k for k in self._kv if k.startswith(prefix)]:   # tenant namespace only
                e = self._live(k)
                if e is None or e.user != user or e.embedding is None:  # and only this user's entries
                    continue
                s = float(q @ (e.embedding / (np.linalg.norm(e.embedding) or 1.0)))
                if s > best_s:
                    best, best_s = e, s
            if best is not None and best_s >= self.threshold:
                self.stats["hit_semantic"] += 1
                return CacheHit(best.payload, "semantic", best_s)
        self.stats["miss"] += 1
        return None

    async def put(self, tenant: str, user: str, query: str, payload: Any, embedding: np.ndarray | None = None) -> float:
        ttl = self.ttl_s + self._rng.uniform(-self.jitter_s, self.jitter_s)
        self._kv[self.key(tenant, user, normalise_query(query))] = _Entry(payload, self._clock() + ttl, user, embedding)
        return ttl

    async def single_flight(self, tenant: str, user: str, query: str, factory: Callable[[], Awaitable[Any]]) -> Any:
        """Concurrent identical misses share one computation (stampede control)."""
        k = self.key(tenant, user, normalise_query(query))
        if k in self._inflight:
            self.stats["coalesced"] += 1
            return await self._inflight[k]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[k] = fut
        try:
            result = await factory()
            fut.set_result(result)
            return result
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()              # mark retrieved so un-awaited futures do not warn
            raise
        finally:
            del self._inflight[k]

    def flush_tenant(self, tenant: str) -> int:
        prefix = f"{{{tenant}}}:"
        n = 0
        for k in [k for k in self._kv if k.startswith(prefix)]:
            del self._kv[k]
            n += 1
        return n

    # hot window: last 3 turns per (tenant, session)
    async def push_turn(self, tenant: str, session: str, turn: dict) -> None:
        self._hot[(tenant, session)].append(turn)

    async def window(self, tenant: str, session: str) -> list[dict]:
        return list(self._hot[(tenant, session)])


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class _Approval:
    approval_id: str
    tenant_id: str
    tool: str
    args_redacted: dict
    idempotency_key: uuid.UUID
    requested_by: str
    created: float
    expires: float
    repaired: dict
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewed_by: str | None = None
    reason: str | None = None
    payload_ref: str | None = None
    result: dict | None = None


@dataclass
class _Tenant:
    facts: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    approvals: dict[str, _Approval] = field(default_factory=dict)
    by_key: dict[uuid.UUID, str] = field(default_factory=dict)
    payloads: dict[str, bytes] = field(default_factory=dict)
    chunks: dict[str, dict] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)
    entities: dict[str, dict] = field(default_factory=dict)
    submissions: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class InMemoryStateStore:
    """Postgres stand-in. All access is through methods that take tenant_id, and each tenant has its own
    table space, so a cross-tenant read is impossible through this API (the RLS analogue)."""

    def __init__(self, clock: Clock = time.time, approval_ttl_s: float = 24 * 3600):
        self._clock, self.approval_ttl_s = clock, approval_ttl_s
        self._t: dict[str, _Tenant] = defaultdict(_Tenant)
        self._vault = Fernet(Fernet.generate_key())     # KMS stand-in; key lives only in this process
        self._pii = PiiRedactor()

    # ---- session facts (verified only)
    async def put_fact(self, tenant: str, session: str, fact: str, source_chunk_id: str, verified: bool) -> str:
        if not verified:
            raise PlatformError(ErrorCode.INVALID_PARAMS, "only verified facts may be stored")
        fid = uuid.uuid4().hex
        self._t[tenant].facts[session].append({"fact_id": fid, "fact": fact, "source_chunk_id": source_chunk_id,
                                               "verified_at": self._clock()})
        return fid

    async def get_facts(self, tenant: str, session: str) -> list[dict]:
        return list(self._t[tenant].facts.get(session, []))

    # ---- chunks (source of truth for the indexes)
    async def put_chunk(self, tenant: str, chunk_id: str, text: str, meta: dict | None = None) -> None:
        self._t[tenant].chunks[chunk_id] = {"chunk_id": chunk_id, "text": text, "meta": meta or {}}

    async def get_chunk(self, tenant: str, chunk_id: str) -> dict | None:
        return self._t[tenant].chunks.get(chunk_id)

    # ---- audit: per-tenant SHA-256 hash chain (D-15)
    @staticmethod
    def _hash(prev: str, tenant: str, seq: int, event_type: str, payload: dict) -> str:
        blob = json.dumps({"tenant": tenant, "seq": seq, "type": event_type, "payload": payload},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev + blob).encode()).hexdigest()

    async def append_audit(self, tenant: str, event_type: str, payload_redacted: dict) -> dict:
        probe = PreContext(tenant, json.dumps(payload_redacted, sort_keys=True))
        self._pii.run(probe)
        if probe.flags.get("pii_redactions"):           # zero raw PII persistence: refuse, do not store
            raise PlatformError(ErrorCode.PII_BLOCKED, "audit payload contains unredacted PII")
        t = self._t[tenant]
        async with t.lock:
            seq = len(t.audit) + 1
            prev = t.audit[-1]["hash"] if t.audit else GENESIS
            ev = {"seq": seq, "tenant_id": tenant, "event_type": event_type, "payload_redacted": payload_redacted,
                  "prev_hash": prev, "hash": self._hash(prev, tenant, seq, event_type, payload_redacted),
                  "ts": self._clock()}
            t.audit.append(ev)
            return ev

    def verify_chain(self, tenant: str) -> tuple[bool, int | None]:
        prev = GENESIS
        for ev in self._t[tenant].audit:
            ok = ev["prev_hash"] == prev and ev["hash"] == self._hash(
                prev, tenant, ev["seq"], ev["event_type"], ev["payload_redacted"])
            if not ok:
                return False, ev["seq"]
            prev = ev["hash"]
        return True, None

    def anchor(self, tenant: str) -> dict:
        a = self._t[tenant].audit
        return {"tenant_id": tenant, "seq": len(a), "head_hash": a[-1]["hash"] if a else GENESIS}

    def export_audit(self, tenant: str) -> list[dict]:
        return [dict(e) for e in self._t[tenant].audit]

    # ---- Tier 3 approvals with encrypted payload vault (D-3)
    async def create_approval(self, tenant: str, tool: str, args_redacted: dict, payload: dict,
                              idempotency_key: uuid.UUID, requested_by: str, repaired: dict | None = None) -> str:
        t = self._t[tenant]
        if idempotency_key in t.by_key:                 # UNIQUE (tenant_id, idempotency_key): replay returns the same record
            return t.by_key[idempotency_key]
        aid = uuid.uuid4().hex
        ref = uuid.uuid4().hex
        t.payloads[ref] = self._vault.encrypt(json.dumps(payload).encode())
        now = self._clock()
        t.approvals[aid] = _Approval(aid, tenant, tool, args_redacted, idempotency_key, requested_by, now,
                                     now + self.approval_ttl_s, repaired or {}, payload_ref=ref)
        t.by_key[idempotency_key] = aid
        await self.append_audit(tenant, "approval.requested", {"approval_id": aid, "tool": tool, "by": requested_by})
        return aid

    def _sweep(self, tenant: str) -> None:
        t, now = self._t[tenant], self._clock()
        for a in t.approvals.values():
            if a.status is ApprovalStatus.PENDING and a.expires <= now:
                a.status = ApprovalStatus.EXPIRED
                t.payloads.pop(a.payload_ref or "", None)
                a.payload_ref = None

    async def list_pending(self, tenant: str) -> list[dict]:
        self._sweep(tenant)
        return [self._view(a) for a in self._t[tenant].approvals.values() if a.status is ApprovalStatus.PENDING]

    async def get_approval(self, tenant: str, approval_id: str) -> dict | None:
        self._sweep(tenant)
        a = self._t[tenant].approvals.get(approval_id)
        return self._view(a) if a else None

    @staticmethod
    def _view(a: _Approval) -> dict:
        return {"approval_id": a.approval_id, "tool": a.tool, "args_redacted": a.args_redacted,
                "status": a.status.value, "requested_by": a.requested_by, "reviewed_by": a.reviewed_by,
                "repaired": a.repaired, "idempotency_key": str(a.idempotency_key), "expires": a.expires,
                "created": a.created, "result": a.result}

    async def decide(self, tenant: str, approval_id: str, reviewer: str, approve: bool, reason: str = "") -> dict | None:
        """Returns the decrypted payload on approval (caller commits it with the stored idempotency key);
        None on rejection. The vault entry is deleted either way."""
        self._sweep(tenant)
        t = self._t[tenant]
        a = t.approvals.get(approval_id)
        if a is None:
            raise PlatformError(ErrorCode.INVALID_PARAMS, "unknown approval")
        if a.status is not ApprovalStatus.PENDING:
            raise PlatformError(ErrorCode.IDEMPOTENCY_CONFLICT, f"approval already {a.status.value}")
        if reviewer == a.requested_by:
            raise PlatformError(ErrorCode.RBAC_DENIED, "reviewer must differ from requester")
        payload = None
        if approve:
            payload = json.loads(self._vault.decrypt(t.payloads[a.payload_ref]))  # type: ignore[index]
        t.payloads.pop(a.payload_ref or "", None)
        a.payload_ref, a.reviewed_by, a.reason = None, reviewer, reason
        a.status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        await self.append_audit(tenant, f"approval.{a.status.value.lower()}", {"approval_id": a.approval_id, "by": reviewer})
        return payload

    # ---- workflow entities (claims, policies) and form submissions
    def seed_entities(self, tenant: str, states: dict[str, str]) -> None:
        for eid, st in states.items():
            self._t[tenant].entities[eid] = {"state": st, "owner": None, "history": []}

    def get_entity(self, tenant: str, entity_id: str) -> dict | None:
        e = self._t[tenant].entities.get(entity_id)
        return dict(e) if e else None

    def apply_transition(self, tenant: str, entity_id: str, from_state: str, to_state: str, approval_id: str) -> None:
        """Compare-and-set: succeeds only if the entity is still in from_state."""
        e = self._t[tenant].entities.get(entity_id)
        if e is None or e["state"] != from_state:
            raise PlatformError(ErrorCode.STATE_CONFLICT, f"{entity_id} is {e['state'] if e else 'unknown'}, not {from_state}",
                                {"current_state": e["state"] if e else None})
        e["history"].append({"from": from_state, "to": to_state, "approval_id": approval_id, "ts": self._clock()})
        e["state"] = to_state

    def new_claim_id(self, tenant: str) -> str:
        nums = [int(k[4:]) for k in self._t[tenant].entities if k.startswith("CLM-")]
        return f"CLM-{(max(nums) if nums else 0) + 1:05d}"

    def create_entity(self, tenant: str, entity_id: str, state: str, owner: str | None) -> None:
        self._t[tenant].entities[entity_id] = {"state": state, "owner": owner, "history": []}

    def record_submission(self, tenant: str, form_id: str, payload: dict, approval_id: str) -> str:
        sid = uuid.uuid4().hex[:12]
        self._t[tenant].submissions.append({"submission_id": sid, "form_id": form_id, "payload": payload, "approval_id": approval_id})
        return sid

    def submissions(self, tenant: str) -> list[dict]:
        return list(self._t[tenant].submissions)

    # ---- approval helpers used by the commit path
    async def get_approval_by_key(self, tenant: str, key: uuid.UUID) -> dict | None:
        aid = self._t[tenant].by_key.get(key)
        return await self.get_approval(tenant, aid) if aid else None

    async def peek_payload(self, tenant: str, approval_id: str) -> dict | None:
        """Decrypt the vault entry without consuming it (used to re-validate before approving)."""
        self._sweep(tenant)
        t = self._t[tenant]
        a = t.approvals.get(approval_id)
        if a is None or a.status is not ApprovalStatus.PENDING or a.payload_ref not in t.payloads:
            return None
        return json.loads(self._vault.decrypt(t.payloads[a.payload_ref]))

    def set_result(self, tenant: str, approval_id: str, result: dict) -> None:
        self._t[tenant].approvals[approval_id].result = result

    def get_result(self, tenant: str, approval_id: str) -> dict | None:
        a = self._t[tenant].approvals.get(approval_id)
        return a.result if a else None

    def vault_size(self, tenant: str) -> int:
        return len(self._t[tenant].payloads)
