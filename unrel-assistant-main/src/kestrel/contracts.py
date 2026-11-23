"""Shared types and contracts (design.md §1.2.1, §1.2.6). Pydantic v2, strict, extra=forbid."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import UUID4, BaseModel, ConfigDict, Field, StringConstraints

TenantId = Annotated[str, StringConstraints(pattern=r"^tenant_[A-Za-z0-9_-]{1,64}$")]
UserId = Annotated[str, StringConstraints(pattern=r"^user_[A-Za-z0-9_-]{1,64}$")]
SessionId = Annotated[str, StringConstraints(
    pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")]  # UUIDv4
SKU = Annotated[str, StringConstraints(pattern=r"^SKU-\d{3,6}$")]
LOT = Annotated[str, StringConstraints(pattern=r"^LOT-\d{5}$")]
ChunkId = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{16,64}$")]

Route = Literal["DIRECT_ANSWER", "HYBRID_RAG", "TRANSACTION_EXEC", "REJECT_OOS"]
Status = Literal["OK", "ABSTAINED", "PENDING_APPROVAL", "HALTED_BREAKER",
                 "HALTED_BUDGET", "DEGRADED", "REFUSED"]


class Tier(int, Enum):
    T1_READ = 1
    T2_COMPUTE = 2
    T3_MUTATE = 3


class Base(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class AgentQueryRequest(Base):
    tenant_id: TenantId
    user_id: UserId
    session_id: SessionId | None = None
    query: Annotated[str, StringConstraints(min_length=1, max_length=4000)]


class Citation(Base):
    chunk_id: ChunkId
    source_uri: str
    score: float = Field(ge=0, le=1)


class AgentQueryResponse(Base):
    tenant_id: TenantId  # echoed; used by the cross-tenant leak probe
    answer: str
    citations: list[Citation]
    route: Route
    status: Status
    grounding_score: float | None = Field(default=None, ge=0, le=1)  # inline verifier, not RAGAS
    approval_id: str | None = None
    remediation: str | None = None
    trace_id: str


class ErrorCode(int, Enum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    TENANT_MISMATCH = -32001
    RBAC_DENIED = -32002
    PII_BLOCKED = -32003
    INJECTION_BLOCKED = -32004
    MCP_TIMEOUT = -32005
    BREAKER_OPEN = -32006
    IDEMPOTENCY_CONFLICT = -32007
    PENDING_APPROVAL = -32008
    LOOP_BUDGET_EXCEEDED = -32009
    UNGROUNDED_DRAFT = -32010
    UPSTREAM_RATE_LIMITED = -32011
    STATE_CONFLICT = -32013      # entity state changed since the request (design.md §1.2.6, D-26)


HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.PARSE_ERROR: 400, ErrorCode.INVALID_REQUEST: 400, ErrorCode.METHOD_NOT_FOUND: 400,
    ErrorCode.INVALID_PARAMS: 422, ErrorCode.TENANT_MISMATCH: 403, ErrorCode.RBAC_DENIED: 403,
    ErrorCode.PII_BLOCKED: 422, ErrorCode.INJECTION_BLOCKED: 422, ErrorCode.MCP_TIMEOUT: 504,
    ErrorCode.BREAKER_OPEN: 200, ErrorCode.IDEMPOTENCY_CONFLICT: 409, ErrorCode.PENDING_APPROVAL: 202,
    ErrorCode.LOOP_BUDGET_EXCEEDED: 200, ErrorCode.UNGROUNDED_DRAFT: 200,
    ErrorCode.UPSTREAM_RATE_LIMITED: 429, ErrorCode.STATE_CONFLICT: 409,
}


class PlatformError(Exception):
    """Raised by any component; carries a JSON-RPC style code."""

    def __init__(self, code: ErrorCode, message: str, data: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data or {}

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.code]

    def to_dict(self) -> dict:
        return {"code": int(self.code), "name": self.code.name, "message": self.message, "data": self.data}


# --- tool argument contracts (idempotency keys are minted by the harness, never by callers)
class HybridRetrieveIn(Base):
    tenant_id: TenantId
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    bm25_top_k: int = Field(default=50, ge=1, le=100)
    dense_top_k: int = Field(default=50, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    pool: int = Field(default=40, ge=4, le=100)
    rerank_k: int = Field(default=4, ge=1, le=10)


class ResolveAnchorIn(Base):
    tenant_id: TenantId
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class ExpandSubgraphIn(Base):
    node_id: str
    depth: int = Field(default=2, ge=1, le=2)
    token_budget: int = Field(default=320, ge=32, le=1000)


class PutSessionFactIn(Base):
    tenant_id: TenantId
    session_id: SessionId
    fact: str
    source_chunk_id: ChunkId


class SubmitFormIn(Base):
    tenant_id: TenantId
    form_id: str
    payload: dict[str, str]


class TransitionWorkflowIn(Base):
    tenant_id: TenantId
    entity: str  # SKU-xxx or LOT-xxxxx, validated in the harness
    from_state: str
    to_state: str


class AppendAuditIn(Base):
    tenant_id: TenantId
    event_type: str
    payload_redacted: dict[str, str]
