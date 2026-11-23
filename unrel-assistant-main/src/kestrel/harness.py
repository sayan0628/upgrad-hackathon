"""ToolExecutionHarness: the mandatory path for every tool call (design.md §1.2.5, §1.2.9, D-7, D-17).

Order: strict validation -> deterministic repair (+ revalidate once) -> caller x tool RBAC ->
harness-minted idempotency (T2/T3 only) -> circuit breaker -> dispatch with per-tool timeout.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from .contracts import (AppendAuditIn, Base, ChunkId, ErrorCode, ExpandSubgraphIn, HybridRetrieveIn,
                        PlatformError, PutSessionFactIn, ResolveAnchorIn, SessionId, SubmitFormIn, TenantId,
                        Tier, TransitionWorkflowIn)

SKU_RE = re.compile(r"^SKU-\d{3,6}$")
LOT_RE = re.compile(r"^LOT-\d{5}$")
CLM_RE = re.compile(r"^CLM-\d{5}$")      # insurance claim ids
POL_RE = re.compile(r"^POL-\d{6}$")      # insurance policy ids
_SKU_LOOSE = re.compile(r"^\s*sku\s*[-_ ]?\s*(\d{1,6})\s*$", re.I)
_LOT_LOOSE = re.compile(r"^\s*lot\s*[-_ ]?\s*(\d{1,5})\s*$", re.I)
_CLM_LOOSE = re.compile(r"^\s*(?:clm|claim)\s*[-_ #]?\s*(\d{1,5})\s*$", re.I)
_POL_LOOSE = re.compile(r"^\s*(?:pol|policy)\s*[-_ #]?\s*(\d{1,6})\s*$", re.I)


class ReadChunkIn(Base):
    tenant_id: TenantId
    chunk_id: ChunkId


class GetSessionFactsIn(Base):
    tenant_id: TenantId
    session_id: SessionId


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tier: Tier
    model: type[BaseModel]
    timeout_s: float
    repair_fields: tuple[str, ...] = ()


TOOLS: dict[str, ToolSpec] = {s.name: s for s in (
    ToolSpec("hybrid_retrieve", Tier.T1_READ, HybridRetrieveIn, 0.110),
    ToolSpec("read_chunk", Tier.T1_READ, ReadChunkIn, 0.110),
    ToolSpec("resolve_anchor", Tier.T1_READ, ResolveAnchorIn, 0.040),
    ToolSpec("expand_subgraph", Tier.T1_READ, ExpandSubgraphIn, 0.040),
    ToolSpec("get_session_facts", Tier.T1_READ, GetSessionFactsIn, 0.100),
    ToolSpec("put_session_fact", Tier.T2_COMPUTE, PutSessionFactIn, 0.100),
    ToolSpec("append_audit", Tier.T2_COMPUTE, AppendAuditIn, 0.100),
    ToolSpec("submit_form", Tier.T3_MUTATE, SubmitFormIn, 0.100),
    ToolSpec("transition_workflow", Tier.T3_MUTATE, TransitionWorkflowIn, 0.100, repair_fields=("entity",)),
)}

# Caller x tool matrix (design.md §1.2.9). Anything not listed is RBAC_DENIED.
ALLOWED: dict[str, frozenset[str]] = {
    "retrieval": frozenset({"hybrid_retrieve", "resolve_anchor", "expand_subgraph", "read_chunk"}),
    "verification": frozenset({"read_chunk"}),
    "execution": frozenset({"submit_form", "transition_workflow"}),
    "orchestrator": frozenset({"get_session_facts", "put_session_fact", "append_audit"}),
    "reviewer_api": frozenset({"append_audit"}),
}


def repair_identifier(value: str) -> str:
    """SKU99 -> SKU-099, lot 44 -> LOT-00044. Pads to the minimum width only; no other guessing."""
    if SKU_RE.match(value) or LOT_RE.match(value) or CLM_RE.match(value) or POL_RE.match(value):
        return value
    if m := _SKU_LOOSE.match(value):
        return "SKU-" + m.group(1).zfill(3)
    if m := _LOT_LOOSE.match(value):
        return "LOT-" + m.group(1).zfill(5)
    if m := _CLM_LOOSE.match(value):
        return "CLM-" + m.group(1).zfill(5)
    if m := _POL_LOOSE.match(value):
        return "POL-" + m.group(1).zfill(6)
    return value


def canonical_args_hash(tenant_id: str, tool: str, args: dict[str, Any]) -> str:
    """Hash of tenant+tool+args. Idempotency keys are never part of args, so they are excluded by design."""
    blob = json.dumps({"t": tenant_id, "tool": tool, "a": args}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class IdempotencyStore:
    """In-memory reference implementation; Redis-backed in deployment (24 h TTL)."""

    def __init__(self) -> None:
        self._keys: dict[tuple[str, str, str, str], uuid.UUID] = {}
        self._results: dict[tuple[str, uuid.UUID], tuple[str, Any]] = {}

    def mint(self, tenant: str, session: str, tool: str, args_hash: str) -> uuid.UUID:
        return self._keys.setdefault((tenant, session, tool, args_hash), uuid.uuid4())

    def lookup(self, tenant: str, key: uuid.UUID, args_hash: str) -> tuple[bool, Any]:
        if (tenant, key) not in self._results:
            return False, None
        stored_hash, result = self._results[(tenant, key)]
        if stored_hash != args_hash:
            raise PlatformError(ErrorCode.IDEMPOTENCY_CONFLICT, "same idempotency key, different arguments")
        return True, result

    def store(self, tenant: str, key: uuid.UUID, args_hash: str, result: Any) -> None:
        self._results[(tenant, key)] = (args_hash, result)


class CircuitBreaker:
    """Counts failures of identical (tenant, tool, args) calls. Opens at max_failures within a request;
    the session-level state keeps it open for session_ttl_s so the user cannot re-trigger the same failing
    call. No half-open probing in v1. Never shared across tenants (tenant is part of the key)."""

    def __init__(self, max_failures: int = 2, session_ttl_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_failures, self.session_ttl_s, self._clock = max_failures, session_ttl_s, clock
        self._req: dict[tuple[str, str], int] = {}
        self._open_until: dict[tuple[str, str, str], float] = {}

    def check(self, request_id: str, tenant: str, session: str, args_hash: str) -> None:
        if self._open_until.get((tenant, session, args_hash), 0.0) > self._clock() or \
                self._req.get((request_id, args_hash), 0) >= self.max_failures:
            raise PlatformError(ErrorCode.BREAKER_OPEN, "circuit breaker open for identical call",
                                {"args_hash": args_hash[:12]})

    def record_failure(self, request_id: str, tenant: str, session: str, args_hash: str) -> None:
        n = self._req.get((request_id, args_hash), 0) + 1
        self._req[(request_id, args_hash)] = n
        if n >= self.max_failures:
            self._open_until[(tenant, session, args_hash)] = self._clock() + self.session_ttl_s

    def end_request(self, request_id: str) -> None:
        for k in [k for k in self._req if k[0] == request_id]:
            del self._req[k]


@dataclass
class ToolResult:
    value: Any
    tool: str
    tier: Tier
    repaired: dict[str, tuple[str, str]] = field(default_factory=dict)  # field -> (typed, will act on)
    idempotency_key: uuid.UUID | None = None
    replayed: bool = False


Dispatch = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class HarnessMetrics:
    first_pass_valid: int = 0
    repaired_ok: int = 0
    invalid: int = 0


class ToolExecutionHarness:
    def __init__(self, dispatch: Dispatch, idempotency: IdempotencyStore | None = None,
                 breaker: CircuitBreaker | None = None):
        self._dispatch = dispatch
        self.idempotency = idempotency or IdempotencyStore()
        self.breaker = breaker or CircuitBreaker()
        self.metrics = HarnessMetrics()  # feeds KPI-14 / KPI-15

    def _validate(self, spec: ToolSpec, args: dict[str, Any]) -> BaseModel:
        model = spec.model.model_validate(args)
        if spec.name == "transition_workflow":
            entity = model.entity  # type: ignore[attr-defined]
            if not any(r.match(entity) for r in (SKU_RE, LOT_RE, CLM_RE, POL_RE)):
                raise ValueError("entity must be SKU-nnn.., LOT-nnnnn, CLM-nnnnn or POL-nnnnnn")
        return model

    async def call(self, *, caller: str, tenant_id: str, session_id: str, request_id: str, tool: str,
                   args: dict[str, Any], idempotency_key: uuid.UUID | None = None) -> ToolResult:
        spec = TOOLS.get(tool)
        if spec is None:
            raise PlatformError(ErrorCode.METHOD_NOT_FOUND, f"unknown tool {tool!r}")

        # 1-2. strict validation, then deterministic repair and one revalidation
        repaired: dict[str, tuple[str, str]] = {}
        try:
            model = self._validate(spec, args)
            self.metrics.first_pass_valid += 1
        except (ValidationError, ValueError) as first_error:
            fixed = dict(args)
            for f in spec.repair_fields:
                if isinstance(fixed.get(f), str):
                    new = repair_identifier(fixed[f])
                    if new != fixed[f]:
                        repaired[f] = (fixed[f], new)
                        fixed[f] = new
            if not repaired:
                self.metrics.invalid += 1
                raise PlatformError(ErrorCode.INVALID_PARAMS, "arguments failed validation",
                                    {"errors": _errors(first_error)}) from first_error
            try:
                model = self._validate(spec, fixed)
            except (ValidationError, ValueError) as second_error:
                self.metrics.invalid += 1
                raise PlatformError(ErrorCode.INVALID_PARAMS, "arguments failed validation after repair",
                                    {"errors": _errors(second_error)}) from second_error
            self.metrics.repaired_ok += 1

        clean = model.model_dump()
        if clean.get("tenant_id", tenant_id) != tenant_id:
            raise PlatformError(ErrorCode.TENANT_MISMATCH, "tool tenant differs from session tenant")

        # 3. caller x tool RBAC
        if tool not in ALLOWED.get(caller, frozenset()):
            raise PlatformError(ErrorCode.RBAC_DENIED, f"caller {caller!r} may not call {tool!r}")

        args_hash = canonical_args_hash(tenant_id, tool, clean)

        # 4. idempotency (T2/T3 only; key minted here, never by a model)
        key: uuid.UUID | None = None
        if spec.tier >= Tier.T2_COMPUTE:
            key = idempotency_key or self.idempotency.mint(tenant_id, session_id, tool, args_hash)
            hit, stored = self.idempotency.lookup(tenant_id, key, args_hash)
            if hit:
                return ToolResult(stored, tool, spec.tier, repaired, key, replayed=True)

        # 5. breaker, then dispatch with the per-tool timeout (timeouts count as failures)
        self.breaker.check(request_id, tenant_id, session_id, args_hash)
        try:
            async with asyncio.timeout(spec.timeout_s):
                value = await self._dispatch(tool, clean)
        except TimeoutError as e:
            self.breaker.record_failure(request_id, tenant_id, session_id, args_hash)
            raise PlatformError(ErrorCode.MCP_TIMEOUT, f"{tool} exceeded {spec.timeout_s * 1000:.0f} ms") from e
        except PlatformError:
            self.breaker.record_failure(request_id, tenant_id, session_id, args_hash)
            raise
        except Exception as e:  # noqa: BLE001 - any dispatch failure counts toward the breaker
            self.breaker.record_failure(request_id, tenant_id, session_id, args_hash)
            raise PlatformError(ErrorCode.INVALID_REQUEST, f"{tool} failed: {type(e).__name__}") from e

        if key is not None:
            self.idempotency.store(tenant_id, key, args_hash, value)
        return ToolResult(value, tool, spec.tier, repaired, key)


def _errors(e: Exception) -> list[dict[str, str]]:
    if isinstance(e, ValidationError):
        return [{"loc": ".".join(map(str, x["loc"])), "msg": x["msg"]} for x in e.errors()]
    return [{"loc": "", "msg": str(e)}]
