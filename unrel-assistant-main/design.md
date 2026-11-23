# design.md — "Kestrel" → Enterprise-Grade Agentic Platform

**Status:** DRAFT v0.2 (second pass applied; Phase 0 measured; request pipeline, in-memory backends and validators-design in place; see §4.1) · **Gate:** 1 (pre-build) · **Source:** `Kestrel_MidCheck_Enterprise_Blueprint_v2.docx`
**Companion artifact:** `docs/workflow_orchestration.png` (generator: `docs/make_diagram.py`, matplotlib, 300 DPI)

---

## 0. Overview

### 0.1 Problem
The inherited prototype has five production-blocking defects:

| # | Defect | Root cause | Design answer |
|---|--------|-----------|---------------|
| P1 | Hallucinated facts | LLM answers without verified evidence | 3-stage hybrid RAG, cross-encoder confidence cut-off, deterministic abstention, NLI verification subagent, RAGAS gate |
| P2 | Runaway tool loops | Unbounded ReAct, no idempotency | Deterministic state machine, `max_iterations=5`, idempotent circuit breaker, UUIDv4 idempotency keys |
| P3 | Cross-user cache leaks | Shared / unscoped cache keys | `SHA256(tenant_id:user_id:norm_query)` keys, `tenant_id` in every query, concurrency probe (KPI-28) |
| P4 | Context drift | Tool output noise displaces intent | Immutable `GoalAnchorBlock`, observation compression ≤180 tokens, 3-turn window + durable `session_facts` |
| P5 | Latency collapse under load | Sync I/O, no batching, no isolation | Async stateless workers, bulkheads, dynamic micro-batching, semantic cache, vLLM continuous batching |

### 0.2 Solution in one paragraph
A multi-tenant, MCP-enabled agent platform. Requests enter through an API gateway, pass deterministic **pre-execution plugins** (PII redaction → injection shield), hit a **tenant-isolated semantic cache**, and are classified by a **triage agent** into four routes. Retrieval runs as a **3-stage hybrid pipeline** (BM25 ∥ dense → RRF → cross-encoder) behind an MCP server. A **bounded ReAct loop** drives tools through a **ToolExecutionHarness** (validation, auto-repair, RBAC, idempotency) and a **circuit breaker**. State-mutating (Tier 3) actions freeze at an **asynchronous HITL gate**. Every answer is verified by an independent **verification subagent** and scored by **RAGAS**; every hop is traced with **OpenTelemetry**. 36 KPIs gate release.

### 0.3 Technology stack

| Layer | Choice | Binding | Justification |
|-------|--------|---------|---------------|
| Agent runtime / API | Python 3.11, FastAPI, Uvicorn (async), `uvloop` | MCP host client (`mcp.ClientSession`) | Non-blocking I/O; stateless → horizontal scale to 50K connections |
| Edge | NGINX / Envoy | — | TLS termination, rate limiting, connection fan-in, traceparent injection |
| Sparse retrieval | Inverted-index engine with configurable BM25 `k1`/`b` and a custom tokeniser (Lucene-class/OpenSearch or Tantivy; chosen in the Phase 2 benchmark). Identifiers (`SKU-099`, `LOT-00044`, clause numbers) are kept as single tokens; Porter stemming only for prose. `rank-bm25` is kept for the RAG Lab only (scores every document in Python; will not hold the 12 ms p95) | `mcp-hybrid-rag` | Exact lexical match for SKU/lot/regulation codes; per-tenant indexes |
| Dense retrieval | `bge-large-en-v1.5` (ONNX) + FAISS **HNSW** | `mcp-hybrid-rag` | Semantic recall (see Open Issue O-1 on IVF-PQ vs HNSW) |
| Reranker | `bge-reranker-v2-m3` ONNX FP16, Triton / ONNX Runtime service | called by `mcp-hybrid-rag` | Cross-encoder precision, dynamic batching |
| State | PostgreSQL 16 + pgvector; Redis Cluster | `mcp-state-store` | ACID audit + sub-ms tenant-isolated cache |
| Token compression | Obsidian vault + Graphify graph | `mcp-graphify-vault` | 1–2 hop subgraph (~320 vs ~1,400 tokens) |
| Plugins | Pluggy entry-point registry | Pre / Post / PII / OTel / HITL | Hot-swappable deterministic interceptors outside LLM reasoning |
| Generation | **Hosted mid-tier model via OpenRouter** (D-21) behind a provider-agnostic `LLMClient` (streaming, provider prompt-caching of system + GoalAnchor prefix, per-tenant token bucket, global concurrency cap = `llm_pool`, fallback model on 429). Self-hosted vLLM/TRT-LLM kept as the Phase 0 fallback (see R-5) | `llm_pool` | Decision D-4. Model/vendor chosen by the Phase 0 latency benchmark |
| Validation | Pydantic v2 (strict) | ToolExecutionHarness | Typed contracts, first-pass validation KPI |
| PII | Microsoft Presidio + regex/homoglyph normaliser | `pii_redactor` | ≥99.8% recall incl. obfuscation |
| Injection shield | Rules/heuristics + small self-hosted prompt-injection classifier (DeBERTa-class), ≤ 8 ms | `injection_shield` | Deterministic, fail-closed, testable in isolation. Exact model chosen in Phase 1 |
| Triage | Distilled small classifier (MiniLM/DistilBERT-class) over `(intent, domain, risk_tier, complexity)`; no temperature, no LLM | `triage` | ≤ 25 ms; trained offline on the labelled intent set (KPI-04) |
| Inline verifier | Small NLI cross-encoder (DeBERTa-v3-class) + extractive overlap check | `verification` subagent | 5–15 ms veto before `final`; claim-level LLM audit is off-path (§1.2.11) |
| Semantic-cache embedder | Same `bge-large-en-v1.5` ONNX instance as dense retrieval | `gateway.embed` | One embedding per request, reused (D-2) |
| Eval | RAGAS, NLI claim verifier, 250-Q golden set | `ragas_auditor` | Release gating |
| Observability | OpenTelemetry → Collector → Prometheus + Tempo + Grafana | `otel_exporter` | Full-trace, W3C `traceparent` |
| Load test | Grafana k6 + Apache JMeter (1 controller + 10 injectors) | — | Dual-tool validation at 50K VUs |
| Dev workbench | Streamlit (`app.py`, 7 tabs as pages, §2.6–2.7) | — | Live mid-check verification; Tab 7 runs the feature validators on open |
| Packaging / deploy | `uv`, Docker, Kubernetes (HPA, PodDisruptionBudgets) | — | Reproducible, autoscaled |

### 0.4 Architecture diagram
![Workflow orchestration](docs/workflow_orchestration.png)

*Seven zones: ① Ingress & Security, ② Triage & Subagent Routing, ③ Control Plane (bounded ReAct), ④ MCP Servers & 3-Stage RAG, ⑤ Data & Knowledge Stores, ⑥ Verification, HITL Gate & Egress, ⑦ Observability (OTel).*

---

## 1.1 System Charter & Hard Boundaries

### Purpose
Answer enterprise questions with **verifiable, cited** responses, and perform controlled workflow actions, for multiple tenants, with **fast paths (cache hit, abstention, refusal, extractive answer) at ≈ 40–250 ms p95** and **LLM-generated answers at ≈ 1–2.2 s p95**, under 50,000 concurrent virtual users (per-path targets, D-23: the hosted-LLM time-to-first-token floor measured in Phase 0 is ≈ 0.6 s).

### In scope
- Question answering over tenant policies, SOPs and product catalogues (Obsidian atomic notes + indexed documents).
- Entity-aware lookups (SKU, LOT) and schema-validated form submissions / workflow transitions.
- Citation grounding, abstention on unanswerable queries, structured remediation on halted loops.
- Multi-tenant isolation, audit, observability, KPI scorecard, red-team harness, developer workbench.

### Non-goals (hard boundaries)
1. **No autonomous clinical, financial or legal execution.** Any Tier 3 (mutating) action requires an approved HITL record. This is enforced in code (KPI-19 = 100%), not by prompt.
2. No open-web browsing; the Execution subagent has no RAG or external web access.
3. No persistence of raw PII (only redacted forms and hashes).
4. No cross-tenant retrieval, caching or analytics.
5. No LLM fine-tuning or hosting of generation models (generation is a hosted mid-tier model, D-4). Small classifiers (triage, injection shield, inline NLI verifier) are trained/fine-tuned **offline** as Phase 1–2 deliverables with versioned datasets; there is no online training pipeline.
6. No LLM-authored security decisions: guardrails are deterministic plugins, `fail_closed`.

### Personas

| Persona | Needs | Key interactions |
|---------|-------|------------------|
| End user (tenant employee) | Fast, correct, cited answers; safe actions | `POST /api/v1/agent/query` |
| Tenant admin | Policy vault curation, tenant config, audit export | Vault ingest, audit API |
| HITL reviewer | Review/approve/reject Tier 3 mutations | Approval queue UI/API |
| SRE / on-call | Detect and mitigate latency, hallucination, MCP hangs | Grafana, runbooks, plugin toggles |
| ML / QA engineer | Evaluate RAG quality, tune hyper-parameters | RAG Lab, RAGAS suite |
| Red-teamer / security | Probe injection, PII smuggling, tenant leaks | Red-team runner |

### Functional requirements

| ID | Requirement |
|----|-------------|
| FR-01 | Accept query with `tenant_id`, `user_id`, `session_id`; reject if the `X-Tenant-Id` header ≠ body `tenant_id`. **No authentication (D-22):** identity is caller-declared |
| FR-02 | Redact PII and block prompt injection before any model sees the text |
| FR-03 | Serve semantic-cache hits (cosine ≥ 0.92 under aggressive mode, tenant-scoped) |
| FR-04 | Triage to `DIRECT_ANSWER`, `HYBRID_RAG`, `TRANSACTION_EXEC`, `REJECT_OOS` |
| FR-05 | Retrieve with BM25 + dense → RRF → cross-encoder; return ≤4 cited chunks |
| FR-06 | Abstain deterministically when top rerank score < threshold (0.42) |
| FR-07 | Validate and auto-repair tool arguments; enforce RBAC tier and idempotency |
| FR-08 | Bound every loop (5 iterations, ≤120 ms per tool step, request deadline 600 ms on fast paths / 3,500 ms on the LLM path, §1.4.3) and trip breaker on repeated identical failure |
| FR-09 | Freeze Tier 3 actions as `PENDING_APPROVAL` until reviewer decision |
| FR-10 | Verify every draft (claim→chunk NLI) and veto ungrounded output. Tokens stream immediately; a veto emits a `retract` event and the client replaces the text with the abstention (decision D-1, §4.1) |
| FR-11 | Emit hash-chained audit events and OTel spans for every hop |
| FR-12 | Expose the 36-KPI scorecard and run red-team / load suites from CI and Streamlit |
| FR-13 | Answer high-confidence factoid queries with a verbatim, cited extractive answer and no LLM call (§1.4.2); on the LLM path stream an `evidence` event (best cited sentence) before the first LLM token |

---

## 1.2 MCP Tool & Data Contracts

All tools are MCP tools over JSON-RPC 2.0. Local workers use **stdio**; distributed services use **Streamable HTTP/SSE**. Server manifests are **signed** and verified at startup.

### 1.2.1 Shared types (Pydantic v2, `strict=True`, `extra="forbid"`)

```python
from typing import Annotated, Literal
from pydantic import UUID4   # enforces version 4 (plain UUID does not)
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TenantId  = Annotated[str, StringConstraints(pattern=r"^tenant_[A-Za-z0-9_-]{1,64}$")]
UserId    = Annotated[str, StringConstraints(pattern=r"^user_[A-Za-z0-9_-]{1,64}$")]
SessionId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")]  # UUIDv4
SKU       = Annotated[str, StringConstraints(pattern=r"^SKU-\d{3,6}$")]
LOT       = Annotated[str, StringConstraints(pattern=r"^LOT-\d{5}$")]
ChunkId   = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{16,64}$")]   # chunk_hash
RiskTier  = Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"]

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
    tenant_id: TenantId                       # echoed; used by cross-tenant leak probe
    answer: str
    citations: list[Citation]
    route: Literal["DIRECT_ANSWER", "HYBRID_RAG", "TRANSACTION_EXEC", "REJECT_OOS"]
    status: Literal["OK", "ABSTAINED", "PENDING_APPROVAL", "HALTED_BREAKER",
                    "HALTED_BUDGET", "DEGRADED", "REFUSED"]
    # inline verifier score (self-hosted NLI, per response). NOT the RAGAS metric: RAGAS is async/sampled (KPI-20..22)
    grounding_score: float | None = Field(default=None, ge=0, le=1)
    approval_id: str | None = None            # set when status == PENDING_APPROVAL
    remediation: str | None = None            # set for HALTED_* / validation failures
    trace_id: str
```

### 1.2.2 `mcp-hybrid-rag` (stdio, timeout 110 ms, pool 64)

| Item | Contract |
|------|----------|
| Tool `hybrid_retrieve` | in: `HybridRetrieveIn` → out: `HybridRetrieveOut` |
| Resource `rag://chunks/{id}` | returns chunk text + metadata; `{id}` must match `ChunkId`; tenant enforced from session context |
| Tier | `TIER_1_READ` (read-only; subagent cannot call mutating tools) |

```python
class HybridRetrieveIn(Base):
    tenant_id: TenantId
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    bm25_top_k: int = Field(default=50, ge=1, le=100)
    dense_top_k: int = Field(default=50, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    pool: int = Field(default=40, ge=4, le=100)
    rerank_k: int = Field(default=4, ge=1, le=10)
    # no `threshold` field: the abstention threshold (0.42; 0.55 under the hallucination runbook) is server-side
    # config, never caller-settable. Abstain iff top_score < threshold (score == threshold passes).
    # no `idempotency_key`: T1 reads are naturally idempotent (keys apply to T2/T3 only, §1.2.5).

class ScoredChunk(Base):
    chunk_id: ChunkId; text: str
    bm25: float | None; cosine: float | None; rrf_rank: int | None
    rerank_score: float = Field(ge=0, le=1)

class HybridRetrieveOut(Base):
    chunks: list[ScoredChunk]                            # <= rerank_k
    top_score: float
    abstain: bool                                        # True iff top_score < threshold
    stage_ms: dict[Literal["bm25","dense","rrf","rerank"], float]
```

### 1.2.3 `mcp-state-store` (Streamable HTTP, pool 200)

| Tool | Tier | Input (key fields) | Output |
|------|------|--------------------|--------|
| `get_session_facts` | T1 | `tenant_id, session_id` | `list[Fact]` |
| `put_session_fact` | T2 | `tenant_id, session_id, fact (verified only), source_chunk_id` | `fact_id` |
| ~~`cache_lookup`~~ | — | **Moved out of MCP (D-2):** gateway-direct Redis client; see §1.3.2 step 4 | — |
| ~~`cache_store`~~ | — | **Moved out of MCP (D-2):** same client, `ttl_s = 300 ± 45` jitter, only after verification passes | — |
| `submit_form` | **T3** | `tenant_id, form_id, payload, idempotency_key` | `PENDING_APPROVAL` record id |
| `transition_workflow` | **T3** | `tenant_id, entity (SKU/LOT), from_state, to_state, idempotency_key` | `PENDING_APPROVAL` record id |
| `append_audit` | T2 (verification subagent only) | event payload | `event_hash` |

Tier 3 tools never mutate directly: they write `PENDING_APPROVAL`; a separate `approve_action` endpoint (reviewer role only) performs the commit.

### 1.2.4 `mcp-graphify-vault` (stdio)

| Tool | Input | Output |
|------|-------|--------|
| `resolve_anchor` | `tenant_id, query` | `node_id` or `null` |
| `expand_subgraph` | `node_id, depth ≤ 2, token_budget (default 320)` | pruned nodes + edges + text |

### 1.2.5 ToolExecutionHarness (mandatory path for every tool call)

1. **Pydantic v2 strict validation** → on failure go to step 2.
2. **Deterministic regex auto-repair** (no LLM), e.g. `SKU99 → SKU-099`, `lot 44 → LOT-00044` (digits are left-padded to the minimum width: SKU 3, LOT 5; longer inputs are kept as typed); re-validate once. Repair is logged for KPI-15. **Tier 3 rule:** any repaired argument on a T3 call is flagged `repaired=true` and shown to the reviewer as "typed X → will act on Y" (and echoed to the requester in the `PENDING_APPROVAL` response); it is never executed on a silent guess.
3. **RBAC tier check**: caller role vs. tool tier (`T1 < T2 < T3`); subagent allow-list enforced (Retrieval: T1 only; Execution: state-store only; Verification: audit only).
4. **Idempotency (T2/T3 only)**: the key is a UUIDv4 **minted by the harness, never by the model**; the mapping `(tenant_id, session_id, tool, canonical_args_hash) → key` is kept in Redis (24 h) so a retry reuses the same key. Duplicates return the stored result; same key with different args → -32007. T1 calls carry no key. The key is excluded from `canonical_args_hash`.
5. Dispatch through the **circuit breaker** keyed by `(tenant_id, tool_name, canonical_args_hash)`. Semantics: counts **failures** (error, timeout or schema-invalid result) of identical calls; opens at 2 within one request (→ `HALTED_BREAKER`), and a session-level counter (Redis, 60 s TTL) keeps it open across turns so the user cannot re-trigger the same failing call; it closes when the TTL lapses (no half-open probing in v1). Never shared across tenants. A separate, error-rate-based **server bulkhead breaker** (per MCP server) handles FMEA #3/#4.

### 1.2.6 Error codes (JSON-RPC `error.code`)

| Code | Name | Meaning | Retry | User-facing result |
|------|------|---------|-------|--------------------|
| -32700 | PARSE_ERROR | Invalid JSON | no | 400 |
| -32600 | INVALID_REQUEST | Malformed JSON-RPC | no | 400 |
| -32601 | METHOD_NOT_FOUND | Unknown tool | no | 400 |
| -32602 | INVALID_PARAMS | Failed Pydantic validation after repair | no | structured remediation |
| -32001 | TENANT_MISMATCH | Header/body/session tenant differ | no | 403 |
| -32002 | RBAC_DENIED | Tier/role violation | no | 403 |
| -32003 | PII_BLOCKED | PII plugin failed closed | no | 422 |
| -32004 | INJECTION_BLOCKED | Injection shield tripped | no | 422 |
| -32005 | MCP_TIMEOUT | > 110 ms (rag) / 100 ms (state-store) / 40 ms (graphify), always ≤ remaining step budget | yes (≤1, jittered, only if remaining ≥ 2× timeout) | `DEGRADED` answer |
| -32006 | BREAKER_OPEN | Circuit breaker open for `(tool,args)` | no | `HALTED_BREAKER` + remediation |
| -32007 | IDEMPOTENCY_CONFLICT | Same key, different args | no | 409 |
| -32008 | PENDING_APPROVAL | Tier 3 awaiting reviewer | n/a | 202 |
| -32009 | LOOP_BUDGET_EXCEEDED | 5 iterations or request deadline (600 ms fast path / 3,500 ms LLM path, §1.4.3) | no | `HALTED_BUDGET` + best verified partial answer |
| -32010 | UNGROUNDED_DRAFT | Verification vetoed | yes (1 re-ground) | abstain |
| -32011 | UPSTREAM_RATE_LIMITED | LLM 429 | yes (backoff, fallback model) | degraded / queue |

HTTP mapping: validation → 400/422, authz → 403, conflict → 409, HITL → 202, overload → 429 with `Retry-After`, server fault → 5xx (counted in KPI-34).

### 1.2.7 Config (`mcp_config.json`) — as specified in the blueprint
Servers `hybrid-rag` (stdio, **110 ms**, was 150 in the blueprint so it stays under the 120 ms step budget), `state-store` (streamable-http, 100 ms, pool 200), `graphify-vault` (stdio, 40 ms). Plugins: `pii_redactor` (PRE, p10, fail-closed), `injection_shield` (PRE, p20, fail-closed), `circuit_breaker` (TOOL_DISPATCH, `max_identical_failures=2`, scope request + 60 s session), `ragas_auditor` (POST, async, sampled, **alert-only**: `alert_faithfulness=0.92` on a rolling window, never blocks a response), `otel_exporter` (ALL_SPANS → `:4317`). Plugin order is by `priority`; fail-closed plugins reject the request on plugin error.

### 1.2.8 Prompt composition (decision D-5)
The LLM prompt is assembled by the orchestrator, in this order, with a hard cap of **1,500 input tokens** (target ≈ 1,300–1,400, consistent with the audit report's 1,320):

| Block | Budget (tok) | Notes |
|-------|--------------|-------|
| System + `GoalAnchorBlock` (immutable, index 0) | ≤ 250 | provider prompt-cache prefix |
| Top-4 reranked chunks, **extractively compressed** | ≤ 4 × 180 = 720 | compressor keeps sentences with highest query overlap; each keeps `chunk_id` |
| Graphify subgraph (entity-linked queries only) | ≤ 320 | omitted if no anchor resolves |
| Last 3 turns (hot window) + verified `session_facts` | ≤ 210 | |

**Verification and citations run against the full stored chunk text** (`rag://chunks/{id}`), not the compressed copy. KPI-26's baseline (~1,400 raw tokens) and KPI-29's token cost are measured on this assembled prompt. §1.4.1 prefill (row 7) is re-derived from the Phase 0 model benchmark for this prompt size, not from "~320 tokens".

### 1.2.9 Caller × tool matrix (decision D-7)
Subagents are LLM-driven and get only domain tools. Cache, hot window, session-fact writes and audit are owned by **platform code (gateway/orchestrator)**, which no model output can invoke directly.

| Tool | Gateway / orchestrator | Retrieval | Execution | Verification | Reviewer API |
|------|:---:|:---:|:---:|:---:|:---:|
| `hybrid_retrieve`, `rag://chunks/{id}` | | ✔ | | ✔ (read chunks for NLI) | |
| `resolve_anchor`, `expand_subgraph` | | ✔ | | | |
| `get_session_facts` | ✔ (injects into GoalAnchor) | | | | |
| `put_session_fact` | ✔ (only facts the Verification subagent marked verified, with `source_chunk_id`) | | | | |
| cache lookup / store (direct Redis, D-2) | ✔ | | | | |
| hot-window read/write (Redis) | ✔ | | | | |
| `submit_form`, `transition_workflow` (T3 → `PENDING_APPROVAL`) | | | ✔ | | |
| `append_audit` | ✔ (all hops, HITL events, verification verdicts) | | | | ✔ (decision events) |
| `approve_action` / reject | | | | | ✔ (role `reviewer`, reviewer ≠ requester) |

The harness enforces this table; any other pairing returns -32002 RBAC_DENIED. The Verification subagent *emits* verdicts; the orchestrator writes them to the audit log (it no longer holds `append_audit`).

### 1.2.10 Identity, roles and (no) authentication (decision D-22, supersedes D-8)
- **No login, no tokens.** Identity is caller-declared. The client sends `X-Tenant-Id` (required; must equal the body `tenant_id`, FR-01) and optionally `X-User-Id`. The header check catches misrouted or buggy clients; **it is not a security control.**
- **Consequence:** every isolation guarantee (cache keys, RLS-style scoping, KPI-28) holds for *honest* tenant declarations. A caller who lies about `tenant_id` is not stopped. Putting an authenticating proxy in front (OIDC/JWT, tenant from claims) is the first item for any real deployment and is left out of this build on purpose.
- **No role-based access control (D-25, supersedes the `X-Role` roles below).** The API and workbench carry no role header and perform no role checks. Access is tenant-scoped (`X-Tenant-Id` must match the body) and the reviewer-must-differ-from-requester rule (HITL, KPI-19) is kept. The harness tool-tier checks (§1.2.5, §1.2.9) remain as internal controls.
- **Internal:** the gateway derives the tenant for MCP sessions from the validated header/body pair, never from model output; mTLS between services remains a deployment item.
- **Streamlit workbench:** sets `X-Tenant-Id` and `X-User-Id` only. No persona or role selector (D-25).
- **Workbench ↔ service (D-9):** the seven tabs are a client of the real FastAPI service and of Prometheus/Tempo; they import no agent code. Load tests target the API directly, never Streamlit.

### 1.2.11 Model roles under the hosted-model decision (D-4)
| Role | Where | Why |
|------|-------|-----|
| Answer generation | Anthropic models via OpenRouter (the account's guardrails allow only these): `claude-haiku-5.5` with reasoning off (default), fallback `claude-haiku-4.5` | Quality and cost; tokens budgeted via KPI-29 |
| Triage | Self-hosted small classifier | ≤ 25 ms; no per-call token cost |
| Embedder, reranker | Self-hosted ONNX/Triton | Hot-path latency, §1.4.5 GPU sizing stays valid (rerank only) |
| Inline verifier (veto, 5–15 ms) | Self-hosted small NLI cross-encoder + extractive check | A hosted-LLM call cannot fit in 15 ms |
| KPI-24 claim-level audit, RAGAS judge, red-team scoring | Hosted mid-tier model, **off the critical path, sampled** | Needs strong judgement; no latency constraint. Sample rate configured per environment. Judge model + version pinned, temperature 0, with a 25-item human spot-check per release |

**LLM access via OpenRouter (decision D-21).** All generation, judge and claim-audit calls go through OpenRouter's OpenAI-compatible chat-completions API (streaming on), behind `LLMClient`; no other component calls a model provider directly.
- **Credential handling:** the API key is read from `OPENROUTER_API_KEY`, injected from the secret store (KMS/vault in deployed environments; a local, git-ignored `.env` in development). It is never committed, never written to `design.md`, config files, logs, traces or audit events, and the redaction plugin treats `sk-or-…` patterns as secrets. Separate keys per environment (dev, benchmark, CI validators, load/soak) with per-key spend limits set in OpenRouter, so a runaway validator or soak cannot exhaust the production budget.
- **Model selection:** model ids are config (`GENERATION_MODEL`, `FALLBACK_MODEL`, `JUDGE_MODEL`); defaults `anthropic/claude-haiku-5.5`, `anthropic/claude-haiku-4.5`, `anthropic/claude-sonnet-5.5` (judge ≠ generation model, §2.7). OpenRouter guardrails restrict the account to Anthropic endpoints and **cannot be loosened**, so the benchmark covers only these. Generation requests set `reasoning.enabled=false` (measured −0.5 s TTFT).
- **Latency:** OpenRouter adds a routing hop and may serve a model from different upstream providers, so TTFT and tokens/s vary by provider. The Phase 0 benchmark must therefore be run *through OpenRouter*, record which upstream provider served each request, and test provider-routing options (e.g. pinning or preferring lower-latency providers; verify the exact parameters in OpenRouter's docs). Fallback models use OpenRouter's multi-model fallback or the `LLMClient`'s own retry (-32011).
- **Caching and limits:** provider-side prompt caching of the system + GoalAnchor prefix depends on the upstream provider and is measured, not assumed. Per-key and per-account rate limits apply; the global concurrency cap (`llm_pool`) is set from them, and the 50K-VU SLA run still uses `llm-sim` (D-6).
- **Privacy:** only redacted text (after `pii_redactor`) is ever sent; per-tenant policy may restrict routing to providers with no-training/zero-retention terms.

### 1.2.12 External API surface (decision D-11)
No authentication (D-22) and no role-based access control (D-25). Errors use the table in §1.2.6.

| Endpoint | Role | Behaviour |
|----------|------|-----------|
| `POST /api/v1/agent/query` | user | `AgentQueryRequest`. `Accept: text/event-stream` → SSE stream; otherwise one JSON `AgentQueryResponse` (used by tests) |
| `GET /api/v1/approvals?status=PENDING` | reviewer | tenant-scoped queue: `approval_id, tool, args_redacted, repaired[], requested_by, age` |
| `GET /api/v1/approvals/{id}` | requester or reviewer | status and decision |
| `POST /api/v1/approvals/{id}/approve` · `/reject` | reviewer (≠ requester) | body `{reason}`; approve decrypts `payload_ref`, re-validates args and `from_state`, commits with the stored idempotency key, appends audit. Default expiry 24 h → `EXPIRED`, never auto-approved |
| `GET /api/v1/audit/export?from&to` | tenant_admin | NDJSON of that tenant's events with chain hashes and the latest anchor |
| `GET/PUT /api/v1/tenants/{id}/config` | tenant_admin | `cache_mode {exact, semantic}`, `require_verified_evidence` (default true), `shared_knowledge_tier` (v1.1) |
| `POST /api/v1/vault/ingest` → 202 `{job_id}` · `GET /api/v1/vault/jobs/{id}` | tenant_admin | async ingestion, §1.3.5 |
| `POST/GET/DELETE /api/v1/validators/runs[/{id}]` | ml_qa, sre, redteam | feature-validator runs for Tab 7 (§2.7) |
| `GET /healthz` · `GET /readyz` · `GET /metrics` | internal | liveness; readiness (Redis, PG, MCP servers, LLMClient); Prometheus scrape |

**SSE events** (`text/event-stream`): `meta {trace_id, route, path}` → optionally `evidence {chunk_id, quote}` (LLM path, before the first token) → `token {t}`* → optionally `retract {reason}` (the client discards streamed generated text and shows `final.answer`; evidence quotes are verbatim and stay) → `final {status, answer, citations, grounding_score, approval_id?, remediation?, trace_id}`; `error {code, message}` on failure before `final`. Extractive answers, cache hits, abstentions and refusals send `meta` then `final` only. Every response carries `X-Path: fast|llm`.

---

## 1.3 Distributed Topology & Data Flow

### 1.3.1 Components

| Component | Replicas (initial) | Scaling signal | State |
|-----------|--------------------|----------------|-------|
| Envoy / NGINX | 4–6 | connections, CPU | none |
| FastAPI agent workers | 40–80 pods (4 uvicorn workers each) | CPU, in-flight requests, p95 | none (stateless) |
| `mcp-hybrid-rag` workers | 16+ | queue depth, p95 stage latency | read-only per-tenant indexes **loaded in RAM** (FAISS HNSW is not mmap-friendly); tenants sharded across workers by consistent hash, not every index on every worker |
| ONNX/Triton bi-encoder + reranker | GPU pool, sized by §1.4.5 | queue time, GPU util | none |
| `mcp-state-store` | 12+ | pool saturation | none (state in PG/Redis) |
| `mcp-graphify-vault` | 4+ | CPU | `graph.json` in memory |
| Redis Cluster | 6 primaries + 6 replicas | CPU (alert 85%), memory | cache, hot window |
| PostgreSQL + pgvector | 1 primary + 2 sync/async replicas, PgBouncer | connections, replication lag | audit, `session_facts`, approvals |
| vLLM / TRT-LLM | GPU pool | queue time, KV-cache usage | none |
| OTel Collector → Prom / Tempo / Grafana | 3 collectors | span drop rate | metrics, traces |

### 1.3.2 Request data flow (happy path, `HYBRID_RAG`)
1. Client → Envoy (TLS, rate limit, `traceparent` injected).
2. FastAPI validates `AgentQueryRequest`; checks header tenant == body tenant.
3. `PRE_EXECUTION`: `pii_redactor` → `injection_shield` (both fail-closed).
4. `cache.lookup` via a **gateway-side direct Redis client (not MCP)**, two tiers (decision D-2, §4.1): (a) exact: key `SHA256(tenant_id:user_id:norm_query)`; (b) semantic: nearest neighbour in that tenant's namespace with cosine ≥ threshold, using the query embedding computed once here and reused by the dense retriever on a miss. **Hit → respond** (≈ 25 ms path). The embedding cost moves from row 5b to row 3 of §1.4.1; re-sum pending.
5. **Triage** (temperature 0, ≤25 ms) → route.
6. Retrieval subagent calls `hybrid_retrieve` over JSON-RPC: BM25 ∥ dense (parallel) → RRF → cross-encoder. In parallel it calls `graphify.expand_subgraph`.
7. If `top_score < 0.42`: **abstention short-circuit** — no LLM, deterministic response (< 90 ms total).
8. Else ReAct loop: `GoalAnchorBlock` at index 0 + compressed observations (≤180 tok) + last 3 turns → vLLM (streamed).
9. Inline verification (extractive claim→chunk check) vetoes ungrounded drafts; full RAGAS scoring runs `POST_EXECUTION` **off the critical path** and feeds KPI-20..22 and the hallucination alert.
10. Stream events: `token*` → (`retract` if vetoed) → `final{status, citations, trace_id}` (full schema to be added in v0.2). `cache_store` with jittered TTL only for answers that passed verification; hash-chained audit event appended; spans exported.

`TRANSACTION_EXEC`: Execution subagent → harness → state-store; Tier 3 writes `PENDING_APPROVAL` and returns 202; reviewer approves via UI/API; commit executes with the original idempotency key.

### 1.3.3 Isolation & resilience structure
- **Bulkheads:** separate pools — `rag_pool=64`, `db_pool=128`, `llm_pool=256`; health checks and triage use a reserved pool so a slow reranker cannot starve them.
- **Tenant isolation:** `tenant_id` in every Postgres `WHERE` (enforced by row-level security policies, not only app code), in every Redis key, in vector-index namespaces; MCP session carries tenant from the gateway, never from model output.
- **Statelessness:** workers hold no session state; hot window (last 3 turns) lives in Redis, durable facts in `session_facts`.
- **Audit:** per-tenant chain `hash_n = SHA256(hash_{n-1} ‖ canonical_event_payload)` (D-15, §1.3.4), append-only table, periodic anchor of each tenant's head hash.
- **Network:** mTLS between services; signed MCP manifests; egress deny-by-default.

### 1.3.4 Core data model (Postgres)

| Table | Key columns |
|-------|------------|
| `session_facts` | `tenant_id, session_id, fact_id, fact, source_chunk_id, verified_at` (RLS by tenant) |
| `pending_approvals` | `tenant_id, approval_id, tool, args_redacted, payload_ref, idempotency_key, status {PENDING,APPROVED,REJECTED,EXPIRED}, requested_by, reviewed_by, ts`; `UNIQUE (tenant_id, idempotency_key)`; reviewer ≠ requester |
| `approval_payloads` | `tenant_id, payload_ref, ciphertext (KMS envelope), expires_at` — full unredacted args needed at commit; deleted on commit/reject/expiry; never logged or audited (decision D-3) |
| `audit_log` | `seq, tenant_id, event_type, payload_redacted, prev_hash, hash` (append-only, no UPDATE/DELETE grants) |
| `chunks` (+pgvector) | `chunk_hash, tenant_id, text, meta, embedding, index_version` (source of truth; FAISS/BM25 rebuilt from it) |

**Audit chain (decision D-15):** one hash chain **per tenant** (`seq` is per tenant), so appends from different tenants never serialise on one head. Workers append through a per-tenant sequencer (Postgres advisory lock or a single-writer queue) in batches; during a PG failover, events are buffered per tenant in a durable local queue and chained in order on replay. The head hash of each tenant chain is anchored every 5 minutes; the audit export (§1.2.12) includes it.

### 1.3.5 Ingestion & index lifecycle (decision D-16)
1. `POST /api/v1/vault/ingest` (tenant_admin) queues a job → parse → **PII redaction** (no raw PII stored) → chunk (512 tokens, 64 overlap) → embed → write `chunks` rows with a new `index_version`.
2. Build per-tenant BM25 and FAISS HNSW indexes for that version, then run canary queries (recall and score checks against the tenant's golden items). Promotion is an atomic alias swap; the previous two versions are kept for rollback (FMEA #8).
3. Deletions and erasure requests tombstone the chunk immediately (excluded at query time) and are physically removed at the next rebuild (≤ 24 h).
4. Access control is **tenant-level only in v1**; document-level ACLs are a stated non-goal until a customer needs them.
5. Obsidian/Graphify: `graphify build` runs in the same job and publishes `graph.json` under the same version; a stale or missing graph falls back to plain RAG context.

### 1.3.6 Development and test backends (decision D-24)
Redis and Postgres are not available in the build environment (no Docker access), so storage sits behind two interfaces with in-memory reference implementations in `src/kestrel/stores/memory.py`. They are the default in development, CI and the validators' Quick profile.

| Real component | In-memory stand-in | Behaviour reproduced (and tested) | Not reproduced |
|----------------|--------------------|-----------------------------------|----------------|
| Redis cache | `InMemoryCache` | key `{tenant}:cache:sha256(tenant:user:norm_query)`; exact tier then per-(tenant,user) semantic tier at cosine ≥ 0.92; TTL 300 ± 45 s jitter; single-flight stampede control; per-tenant flush; concurrent cross-tenant leak probe (KPI-28 analogue) | latency, cluster slot/failover behaviour, memory eviction |
| Redis hot window | `InMemoryCache.push_turn/window` | last 3 turns per (tenant, session) | persistence, expiry policy |
| Postgres `session_facts` | `InMemoryStateStore.put_fact/get_facts` | verified-only writes; tenant scoping | RLS policies (scoping is by API design here, not by DB policy) |
| Postgres `pending_approvals` + payload vault | `create_approval/decide/list_pending/get_approval` | `UNIQUE (tenant, idempotency_key)`; reviewer ≠ requester; encrypted payload (Fernet) deleted on commit/reject/expiry; 24 h expiry with no auto-approve; repaired-args shown | KMS, durability |
| Postgres `audit_log` | `append_audit/verify_chain/anchor/export_audit` | per-tenant SHA-256 chains, per-tenant `seq`, concurrent appends, tamper detection, refusal of unredacted PII | append-only grants, replication/failover buffering |
| Postgres `chunks` | `put_chunk/get_chunk` | tenant scoping | pgvector search (retrieval uses FAISS/BM25 in memory in v1) |

The same contract tests (`tests/test_stores.py`) are meant to run against the real backends when they exist. Anything marked *not reproduced* (latency, RLS enforcement, memory growth for KPI-36, failover FMEA #1/#7) can only be validated against real Redis/Postgres and stays on the Phase 6–7 list.

---

## 1.4 End-to-End Latency Budget (per path, decision D-23)

**Definitions.** *Fast paths* = cache hit, deterministic abstention, refusal, extractive answer (no LLM call). *LLM path* = anything that calls the model, including `DIRECT_ANSWER`. *E2E latency* = request received at gateway → last byte of a streamed answer; for the LLM path, answers are capped at `max_output_tokens=96`. *TTFT* (LLM path) = request received → first LLM token. *Time-to-first-content (TTFC)* = request received → first cited content, i.e. the `evidence` event or an extractive answer. Responses carry `X-Path: fast|llm` so load tools can threshold each path separately. Load measured as in §1.4.4.

**Why per path.** Phase 0 (§4, R-5) measured a ≈ 600 ms server-side TTFT floor for hosted models through OpenRouter, even for a 1-token reply to a tiny prompt, with ≈ 300 tok/s streaming on the fastest allowed model. A single 200–600 ms target cannot hold for answers that need the LLM, so the design keeps the tight targets where no LLM is needed and sets measured, provisional targets for the LLM path.

### 1.4.1 Budget — full RAG path (cache miss, answerable)

| # | Hop | OTel span | p50 (ms) | p95 budget (ms) | Notes |
|---|-----|-----------|---------|-----------------|-------|
| 1 | Gateway + FastAPI parse | `gateway` | 2 | 4 | keep-alive, uvloop |
| 2 | Pre-hooks (PII + injection) | `hook.pre` | 7 | 12 | KPI-01 ≤ 15 |
| 3 | Cache: exact lookup ∥ query embedding, then per-tenant semantic NN | `cache.lookup`, `gateway.embed` | 12 | 21 | exact 1/3 ∥ embed 10/17 (incl. ≤8 ms micro-batch wait) + NN 2/4; embedding is reused by dense retrieval (D-2). Hits exit here |
| 4 | Triage | `triage.classify` | 10 | 20 | KPI-05 ≤ 25 |
| 5a | BM25 (parallel) | `mcp.rag.bm25` | 5 | 12 | |
| 5b | FAISS HNSW search (parallel with 5a; embedding reused from row 3) | `mcp.rag.faiss` | 4 | 8 | v0.1 had 14/25 including the embedding; KPI-08 = `gateway.embed` + `mcp.rag.faiss` = 17 + 8 = 25 ms |
| 5c | RRF | `mcp.rag.rrf` | 1 | 3 | |
| 5d | Cross-encoder rerank | `mcp.rag.rerank` | 22 | 40 | KPI-09 ≤ 45 |
| 5e | MCP JSON-RPC overhead (2 hops) | — | 2 | 4 | KPI-30 ≤ 4 |
| 5f | Graphify subgraph (parallel with 5a–5d) | `mcp.graphify` | 4 | 10 | hidden behind retrieval |
| 5g | Evidence-first event (best cited sentence of the top chunk) | `agent.evidence` | 2 | 5 | **TTFC ≈ 65 ms p50 / ≈ 125 ms p95** (rows 1–5g); emitted before any LLM token |
| 6 | Prompt assembly + compression | `agent.react.step` | 2 | 5 | |
| 7 | LLM first token (hosted via OpenRouter, reasoning off) | `llm.generate` | 690 | 1,100 | **Measured** (Phase 0, n = 6–8, ≈ 1.4K-token prompt): floor ≈ 600 ms regardless of prompt size; Haiku 5.5 p50 ≈ 650–690 ms, p95 ≈ 0.9–1.4 s; prefix caching and hedging showed no gain |
| | **Cumulative TTFT (LLM path)** | | **≈ 753** | **≈ 1,221 (budget 1,500)** | p50 = 63 + 690; p95 = 121 + 1,100 (non-LLM rows 1–6 are unchanged from the v0.2 re-sum) |
| 8 | LLM decode (≤96 tok, streamed) | `llm.generate` | 320 | 450 | ≈ 300 tok/s p50, ≈ 210 tok/s p95 (Haiku 5.5, measured) |
| 9 | Inline verification + audit append (overlapped with stream tail) | `hook.post` | 5 | 15 | RAGAS itself is async |
| | **Total E2E (LLM path)** | | **≈ 1,080** | **≈ 1,720 (≤ 2,200)** | margin ≈ 480 ms at p95; provisional until the benchmark runs with more samples |

Sum check (LLM path, p95): non-LLM rows 4+12+21+20+59+5 = 121; + first token 1,100 = **≈ 1,221 ms TTFT**; + decode 450 + verify 15 = 1,686 + ≈ 30 ms slack ≈ **1,720 ms**. p50: 63 + 690 = 753 TTFT; + 320 + 5 ≈ **1,080 ms**. p99: observed first-token maxima ≈ 1.3–1.4 s in 8 samples, so assume ≈ 1.6 s TTFT p99 + ≈ 0.6 s decode + ≈ 50 ms ≈ **2.3–2.6 s** vs the 3,000 ms target. Fast-path budgets are the rows 1–5 sums: TTFC/extractive ≈ 125–145 ms p95. Possible further savings if needed: start triage speculatively in parallel with the cache lookup, and start BM25 in parallel with the semantic NN (both cost extra work on cache hits).

### 1.4.2 Fast paths

| Path | Trigger | Expected share | p95 | Mechanism |
|------|---------|----------------|-----|-----------|
| Cache hit (exact or semantic) | per-user, per-tenant; cosine ≥ 0.92 (semantic) | ≥ 35% (KPI-27, under the §1.4.4 mix) | ≤ 40 ms | return stored verified answer |
| Deterministic abstention | rerank top score < 0.42 | ~15% of pipeline traffic (≈ 10% of all) | ≤ 90 ms | template response, 0 LLM tokens |
| **Extractive answer (new, FR-13)** | rerank top score ≥ θ_ext (provisional 0.80, calibrated on the golden set), triage complexity = low, an answer span found | planning assumption ≈ 20% of pipeline traffic (**measure on the golden set; set to 0 if precision is below target**) | ≤ 200 ms | verbatim cited sentence(s) from the top chunk; verification is a string-containment check; 0 LLM tokens; tenant flag `extractive_fast_path` (default on) |
| Reject OOS / blocked | pre-hook or triage | tiny | ≤ 40 ms | deterministic refusal |

`DIRECT_ANSWER` (no retrieval) is **not** a fast path: it calls the LLM and belongs to the LLM path. A blended p50 is no longer a useful gate (above ≈ 50% fast-path share it equals the fast-path latency; below it, it jumps to the LLM path), so KPIs are per path and the blend is reported only. Planning share: cache ≈ 35% + abstention ≈ 10% + extractive ≈ 13% + refusals ≈ 2% ≈ **60% fast**, ≈ 40% LLM.

**Evidence-first streaming (LLM path).** After retrieval and before the first LLM token, the gateway emits an `evidence {chunk_id, quote}` SSE event carrying the best cited sentence. The user sees verified, cited content at ≈ 65–125 ms instead of waiting ≈ 1 s for generated text. It does not make the LLM faster; it moves the first *useful* content earlier. The `final` event still carries the full answer; a veto sends `retract` as before (D-1), and the evidence quote is verbatim so it needs no retraction.

**Measured on the dev machine (CPU only, 4 cores, local MiniLM embedder + MiniLM cross-encoder, a 4-chunk demo corpus, pipeline time excluding HTTP; `src/kestrel/pipeline.py`):**

| Path | p50 | p95 | vs target | Notes |
|------|-----|-----|-----------|-------|
| Cache hit (exact) | 0.1 ms | 0.2 ms | ✓ (≤ 40 ms) | exact tier runs before any embedding |
| Abstain, no lexical overlap and low cosine | 31 ms | 37 ms | ✓ (≤ 90 ms) | pre-rerank gate skips the reranker (floor 0.20, **provisional**) |
| Abstain, with a lexical hit | 125 ms | 151 ms | ✗ (≤ 90 ms) | pays the cross-encoder |
| Extractive answer | 149 ms | 174 ms | ✓ (≤ 200 ms), thin | cross-encoder ≈ 120 ms of it |
| LLM path, time before the LLM call | 141 ms | 145 ms | ≈ row 1–6 budget (121 ms) | embedding ≈ 28 ms (budget 17) |
| **Re-measured on the insurance golden set, rerank pool 4, max_length 128** (`bench_results/latency_pool4_len128.json`, `tools/latency_bench.py`, n = 2 × 30 questions) | p50 ≈ 185 ms pre-LLM; fast OK p50 190 / p95 279 ms; abstain p50 184 / p95 223 ms | | ✗ abstain ≤ 90 ms; ✗ extractive p95 279 vs ≤ 200 ms; fast-path p95 ≈ 280 vs ≤ 250 ms | cross-encoder ≈ 161 ms of it (was ≈ 337 ms at pool 8, length 256). Retrieval quality unchanged: evidence doc in top-4 = 85.2% (pool 4) vs 84.8% (pool 8), `tools/retrieval_eval.py` |

Caveats: the corpus is tiny, so BM25/FAISS cost is negligible and the reranker sees at most the 4 chunks (a real corpus sends `rerank_pool` = 8 candidates: ≈ 150–350 ms on this CPU, measured separately). On CPU the cross-encoder, not retrieval, sets the fast-path latency (R-1). The ≤ 90 ms abstention and ≤ 200 ms extractive targets are therefore **only reliable with a GPU reranker or a smaller/quantised one**; they are not validated by this environment. Local models are the small ones already cached (all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2), chosen for CPU feasibility; whether `bge-*` or newer open models are better on our data is untested and should be benchmarked on the golden set (the interfaces are swappable).

### 1.4.3 Loop budget
**Request deadline model (decision D-12, revised by D-23).** `deadline_hard = t_gateway_receipt + 600 ms` on fast paths and `+ 3,500 ms` on the LLM path (≈ p99 target 3,000 ms + margin). Loop budget = `deadline_hard − elapsed_before_loop − 40 ms` (reserve for verification and egress), so it is always relative to what is left. The per-path p95 figures are measured targets, not cutoffs.
- **Tool steps:** timeout `min(120 ms, remaining)`; MCP transport timeouts are set below it (110 / 100 / 40 ms, §1.2.7). One jittered retry only if `remaining ≥ 2 × timeout`, counted against the same step.
- **Generation step:** governed by a first-token timeout of **2,000 ms** (≈ 1.5× the largest first-token latency seen in Phase 0) and by the deadline, **not** by the 120 ms tool-step budget. On first-token timeout: one attempt on the fallback model if `remaining ≥ 1,500 ms`; otherwise (or if that also fails) deliver the evidence/extractive answer with status `DEGRADED`. If the deadline hits mid-stream, stop and send `final` with status `HALTED_BUDGET` carrying the verified portion.
- **`max_iterations = 5`**, mean ≤ 2.2 steps (KPI-17). A step is one tool call or one generation call. Default `HYBRID_RAG`: orchestrator-issued retrieval (step 1) + generation (step 2). Extra retrievals only for multi-hop, LLM-chosen, max 3. **No LLM query rewrite on the hot path**: only deterministic normalisation (identifier extraction, case/Unicode folding, vault synonyms).
- On budget exhaustion the loop moves to `HALTED_BUDGET` and returns the best verified partial answer or an abstention.

### 1.4.4 Load model
- 50,000 concurrent VUs (k6, 10 injectors via k6 operator; JMeter 1 controller + 10 workers × 5,000 threads).
- Little's law: `VUs = RPS × (latency + think time)`. To land on the KPI-35 target of ≥ 8,500 RPS with a ≈ 0.48 s mean latency (≈ 60% fast paths at ≈ 0.07 s, ≈ 40% LLM path at ≈ 1.1 s), think time ≈ **5.4 s**. The blueprint script's `sleep(1)` would generate ≈ 34,000 RPS (see Open Issue O-3). Plan: sustain **8,500 RPS (SLA)**, with **headroom to ≥ 12,000 RPS** burst; a separate stress profile (`sleep(1)`) is run only to find the breaking point, not for SLA pass/fail. In-flight LLM streams at 8,500 RPS ≈ 3,400 × 1.1 s ≈ 3,700 concurrent; this is why the SLA run uses `llm-sim` (D-6).
- Test mix (decision D-13, v0.2): **40% repeats** (25% exact repeats + 15% paraphrases of earlier queries by the same user, only counted after the first occurrence has warmed the cache), 33% factoid/complex, 12% multi-hop, 10% negative/unanswerable (≈ 17% of the 60% uncached share, matching the ~15% abstention assumption), 5% transactions; 50 tenants. v0.1's 10% repeats could not produce the KPI-27 target of 35% hits. KPI-27 is gated under *this* mix; in production it is reported and alerted on, not gated, because per-user scoping makes it traffic-dependent (R-3). Sensitivity: at a 20% hit ratio the pipeline carries ~6,800 RPS, ~5,800 reach retrieval, and rerank demand at pool 18 is ~105K pairs/s (≈ 70 GPU-equivalents at 1.5K pairs/s, est.) — size the reranker pool for this case, not only for 35%.

### 1.4.5 Capacity sizing (first estimate, to be replaced by benchmarks)

| Resource | Demand at 8,500 RPS | Assumption | Initial sizing |
|----------|---------------------|------------|----------------|
| Cache absorbs | 35% → 3,000 RPS | KPI-27 | Redis Cluster 6 shards |
| Pipeline traffic | ~5,500 RPS (≈ 4,700 reach retrieval after OOS/blocks) | | |
| Abstentions skip LLM | ~15% of pipeline | | LLM RPS ≈ 4,000 |
| Reranker pairs/s | 4,700 × 40 = **~190K pairs/s** | ~1.5K pairs/s per L4/A10-class GPU @ 512 tok (unmeasured) | **~125 GPUs — too high** → see mitigation below |
| LLM (hosted, D-4) | ≈ 3,400 RPS (extractive path at 20% of pipeline) to 4,000 RPS (none) × (~1.4K prompt + ~90 output tok); ≈ 3,700 concurrent streams | Provider quota; far beyond typical limits (verify with vendor) | **Not sizable in GPUs.** Per-request cost uses token counts × a price table (KPI-29); the 8,500 RPS SLA run uses the calibrated simulator (D-6) |
| FastAPI | 8,500 RPS | ~300–400 RPS per 4-worker pod | 30–40 pods + 2× headroom |

**Mitigation for the reranker (top risk R-1):** reduce rerank pool 40 → **16–20** after RRF (recall impact to be measured against KPI-10/11), cap `max_len` at 384 for the first pass, enable INT8 for the pool-reduced path if nDCG holds, request-level dedup, and raise cache hit ratio. Target ≤ 60 GPU-equivalents. Decision recorded in ADR-002 pending benchmark.

### 1.4.6 Scaling levers & degradation ladder (in order)
1. HPA on in-flight requests / queue time; pre-warmed pools.
2. Aggressive semantic cache (cosine ≥ 0.92).
3. `top_k` 50 → 25 for BM25 and dense.
4. Pool 40 → 16 for the cross-encoder.
5. Shed load: 429 + `Retry-After` for lowest-priority tenants (bounded error budget).
6. Fail over to cached read-replica index when an MCP bulkhead trips.

---

## 1.5 KPI & SLA Gate Matrix (36 KPIs)

All KPIs are measured continuously, evaluated in CI/CD, and shown in Streamlit Tab 6 (blueprint text says Tab 5 in Gate 12 and Tab 6 in Gate 18; this design uses **Tab 6**). **Any FAIL blocks release.**

### Stage 1 — Ingress & Triage

| KPI | Metric | Threshold (pass) | Tool |
|-----|--------|------------------|------|
| KPI-01 | Pre-hook guardrail latency p95 | ≤ 15 ms | OTel `hook.pre` |
| KPI-02 | PII redaction recall (email/phone/SSN/obfuscated) | ≥ 99.8% | Presidio / regex audit |
| KPI-03 | Prompt-injection block rate (direct + indirect) | ≥ 99.5% | Red-team benchmark |
| KPI-04 | Triage routing accuracy | ≥ 97.0% | Labeled intent set |
| KPI-05 | Triage classification latency p95 | ≤ 25 ms | OTel `triage` |
| KPI-06 | False-positive refusal rate on benign queries | ≤ 1.0% | Benign control set |

### Stage 2 — Hybrid Retrieval

| KPI | Metric | Threshold | Tool |
|-----|--------|-----------|------|
| KPI-07 | Stage 1 BM25 latency p95 | ≤ 12 ms | OTel `rag.bm25` |
| KPI-08 | Stage 2 bi-encoder + FAISS latency p95 (embedding + search) | ≤ 25 ms | OTel `gateway.embed` + `mcp.rag.faiss` |
| KPI-09 | Stage 3 cross-encoder latency p95 | ≤ 45 ms | OTel `rag.rerank` |
| KPI-10 | Hybrid context recall@pool (candidates actually sent to the reranker: 40, or 16–20 after ADR-002); recall@50 on the fused list reported as a diagnostic | ≥ 0.92 | RAGAS / IR eval |
| KPI-11 | Post-rerank context precision@4 | ≥ 0.88 | RAGAS |
| KPI-12 | MRR@4 | ≥ 0.85 | IR benchmark script |
| KPI-13 | nDCG@4 | ≥ 0.86 | IR benchmark script |

### Stage 3 — Agent & Loop

| KPI | Metric | Threshold | Tool |
|-----|--------|-----------|------|
| KPI-14 | Tool-schema first-pass validation rate | ≥ 96.0% | Pydantic telemetry |
| KPI-15 | Auto-repair success on malformed args | ≥ 90.0% | Harness metrics |
| KPI-16 | Infinite-loop escape rate (breaker) | 100.0% | Chaos loop suite |
| KPI-17 | Mean ReAct steps per completed task | ≤ 2.2 | OTel trace count |
| KPI-18 | Context drift score (goal cosine alignment) | ≥ 0.85 | Embedding drift audit |
| KPI-19 | Tier 3 HITL gate enforcement rate | 100.0% | Audit-log integrity |

### Stage 4 — Generation & RAGAS

| KPI | Metric | Threshold | Tool |
|-----|--------|-----------|------|
| KPI-20 | RAGAS faithfulness | ≥ 0.92 | Post-hook RAGAS |
| KPI-21 | RAGAS answer relevancy | ≥ 0.90 | Post-hook RAGAS |
| KPI-22 | RAGAS context entity recall | ≥ 0.85 | Post-hook RAGAS |
| KPI-23 | Unanswerable-query abstention accuracy | ≥ 98.0% | Negative test set |
| KPI-24 | Unsupported claim / hallucination rate | ≤ 0.5% | NLI claim verifier |
| KPI-25 | TTFT p95 on the LLM path **and** time-to-first-content p95 (evidence event or extractive answer), measured during the 50K-VU run | LLM TTFT ≤ 1,500 ms; TTFC ≤ 250 ms | FastAPI stream metric |

### Stage 5 — Token & Graphify

| KPI | Metric | Threshold | Tool |
|-----|--------|-----------|------|
| KPI-26 | Prompt-token reduction via Graphify + Obsidian | ≥ 45.0% (target 55%) | Token counter hook |
| KPI-27 | Semantic cache hit ratio (steady state) | ≥ 35.0% | Redis hit/miss metric |
| KPI-28 | Cross-tenant cache isolation integrity | 100% (0 leaks) | Concurrency probe |
| KPI-29 | Cost per 1,000 enterprise queries | ≤ $0.45 | OTel token cost calc |
| KPI-30 | MCP JSON-RPC serialization overhead p95 | ≤ 4 ms | MCP client timer |

### Stage 6 — Scale & Load (50K VUs)

| KPI | Metric | Threshold | Tool |
|-----|--------|-----------|------|
| KPI-31 | E2E latency p50 @ 50K VUs, per path | fast paths ≤ 150 ms; LLM path ≤ 1,300 ms | k6 + JMeter (tag `path`) |
| KPI-32 | E2E latency p95 @ 50K VUs, per path | fast paths ≤ 250 ms; LLM path ≤ 2,200 ms | k6 + JMeter (tag `path`) |
| KPI-33 | E2E latency p99 @ 50K VUs, per path | fast paths ≤ 400 ms; LLM path ≤ 3,000 ms | k6 + JMeter (tag `path`) |
| KPI-34 | HTTP 5xx error rate @ 50K VUs | ≤ 0.05% | k6 / JMeter |
| KPI-35 | Sustained throughput | ≥ 8,500 RPS | k6 / JMeter |
| KPI-36 | RSS growth over 1-hr soak | ≤ 3.0% | Prometheus node exporter |

### Release gates beyond KPIs
RAGAS on the 250-question golden set (Factoid 25 / Complex 25 / Multi-hop 20 / Summary 15 / Negative 15 %): faithfulness ≥ 0.92, answer relevancy ≥ 0.90, context precision ≥ 0.88, context recall ≥ 0.85, entity recall ≥ 0.85. Red-team suite (100 prompts, 5 categories) has zero critical bypasses.

**Metric definitions and measurement sizing (decision D-14).** KPI-20 (faithfulness) is an *aggregate mean* over the golden set and sampled traffic; KPI-24 is a *claim-level* rate. They are different measures and the stricter one binds, so both must pass. A KPI measured as a rate passes at 95% confidence only if the Clopper–Pearson bound clears the threshold; with zero failures this needs n ≥ ln(0.05)/ln(p). Minimum sets:

| KPI | Threshold | Minimum n (0 failures) | Dataset |
|-----|-----------|-----------------------|---------|
| KPI-02 PII recall | 99.8% | ≈ 1,500 PII spans | generated + obfuscated (homoglyph, `[at]`) |
| KPI-03 injection block | 99.5% | ≈ 600 prompts | the 100 hand-written seeds × mutations; the 100-prompt suite stays the zero-critical-bypass smoke gate |
| KPI-23 abstention | 98% | ≈ 150 unanswerable queries | dedicated set (the golden set's 15% ≈ 38 is too small) |
| KPI-24 unsupported claims | 0.5% | ≈ 600 sampled claims | claim-level audit by the pinned judge |
| KPI-06 false refusals | 1% | ≈ 300 benign queries | benign control set |
| KPI-04 triage accuracy | 97% | ≈ 100 (≥ 300 recommended, so a few misses are tolerable) | labelled intent set |

With k > 0 failures, use the Clopper–Pearson upper bound and size n up accordingly.

---

## 1.6 Failure Mode & Effects Analysis (FMEA)

Severity (S), Occurrence (O), Detection (D) on 1–5 (5 = worst); RPN = S×O×D.

| # | Component / failure | Effect | S | O | D | RPN | Detection | Deterministic fallback | Recovery / owner |
|---|--------------------|--------|---|---|---|-----|-----------|------------------------|------------------|
| 1 | **Redis cluster down / partitioned** | No cache, no hot window; latency ↑ | 3 | 3 | 1 | 9 | Health probe, `cache.lookup` errors | Bypass cache (fail-open for cache only); hot window rebuilt from `session_facts`; enable load shedding; **never** fall back to a shared cache | Auto-failover to replicas; SRE |
| 2 | **Reranker timeout / GPU saturation** | p95 breach, no confidence score | 4 | 3 | 2 | 24 | `rag.rerank` > 45 ms, queue depth | Default (tenant `require_verified_evidence=true`): abstain deterministically. If the tenant sets it false: return RRF top-4 with `confidence=low` in *strict extractive quote mode* and status `DEGRADED` | Scale GPU pool; lower pool 40→16 |
| 3 | **`mcp-hybrid-rag` crash** | Retrieval unavailable | 5 | 2 | 1 | 10 | stdio EOF / health | Bulkhead trips; fail over to cached read-replica index worker; else abstain with "knowledge temporarily unavailable" | Supervisor restarts worker |
| 4 | **MCP hang (> 110 ms, > 1% rate)** | Tail latency, thread starvation | 4 | 3 | 2 | 24 | JSON-RPC timeout alert | Per-server bulkhead pool isolates it; request timeout → single jittered retry → degraded path | Restart worker; runbook |
| 5 | **LLM 429 / upstream overload** | Generation fails | 4 | 4 | 1 | 16 | HTTP 429, queue time | Exponential backoff w/ jitter (≤1 retry inside budget) → secondary model/replica → extractive answer from top chunk with citation | Capacity; quota |
| 6 | **LLM timeout / bad output (malformed JSON)** | Step wasted | 3 | 3 | 2 | 18 | Pydantic failure, step timeout 120 ms | Regex auto-repair; one re-prompt; else `HALTED` with partial answer | Prompt fix |
| 7 | **Postgres primary failover** | Writes/audit blocked | 5 | 2 | 1 | 10 | PG health, replication lag | Reads → replica; audit events buffered in durable local queue (hash chain preserved); Tier 3 mutations **refused** (fail-closed), not queued as executed | Promote replica; DBA |
| 8 | **Stale / corrupt FAISS or BM25 index** | Wrong or missing evidence | 4 | 2 | 3 | 24 | Index version/checksum mismatch, recall canary | Atomic blue/green index swap; roll back to last good snapshot; canary queries gate promotion | Rebuild from `chunks` |
| 9 | **PII plugin error / bypass** | Raw PII reaches LLM/logs | 5 | 2 | 3 | 30 | KPI-02 audits, plugin exception | `fail_closed`: reject request (-32003); post-hoc log scrub; alert | Hot-fix model/regex |
| 10 | **Injection shield false negative (incl. indirect via retrieved chunk)** | Goal hijack / data exfil | 5 | 2 | 4 | 40 | Red-team suite, anomalous tool calls | Defense in depth: retrieved text sanitised + tagged untrusted; Retrieval subagent read-only; harness RBAC; Tier 3 HITL; verification veto | Update classifier; incident |
| 11 | **Cache stampede / cross-tenant key collision** | Redis CPU > 85%, possible leak | 5 | 2 | 2 | 20 | Key-prefix audit, Redis CPU, leak probe | Enforce `SHA256(tenant:user:norm_query)` prefix, jittered TTL 300±45 s, request coalescing (single-flight) per key; flush affected namespace | Security review |
| 12 | **Tool retry loop (e.g., repeated bad SKU)** | Runaway cost/latency | 4 | 4 | 1 | 16 | Breaker counter | Breaker opens at 2 identical failures → `HALTED_BREAKER` + remediation prompt; global 5-iteration cap | None needed |

Additional (tracked, lower RPN): OTel collector down (spans buffered/dropped, **request path never blocks on telemetry**); HITL reviewer unavailable (approvals expire → `EXPIRED`, no auto-approve); Obsidian/Graphify graph stale (fall back to plain RAG context, KPI-26 alert).

---

## 1.7 Architecture Decision Records

### ADR-001 — Hybrid retrieval: BM25 + dense fused by RRF (k = 60)
- **Context:** Enterprise queries mix exact identifiers (SKU-099, LOT-00044, clause numbers) with paraphrased semantics. Dense-only misses exact tokens; BM25-only misses paraphrase.
- **Decision:** Retrieve top-50 from each (BM25 `k1=1.25, b=0.75`; dense `efSearch=64`, 512-token chunks, 64 overlap), fuse with **RRF, k=60, unweighted**, dedupe by `chunk_hash`, pass the top-40 (pool tunable, see ADR-002) to the cross-encoder, keep 4.
- **Why RRF:** rank-based, so no score-scale calibration between BM25 (unbounded) and cosine; robust default; ~3 ms. Weights are *not* introduced initially; a weighted RRF variant is evaluated in the RAG Lab only if KPI-10/12 miss.
- **Alternatives:** convex score fusion (needs per-tenant normalisation, brittle); dense-only (fails KPI-10 on identifiers).
- **Consequences:** two indices to maintain; parallel execution makes latency `max(S1,S2)` not the sum. Re-evaluate `k` on the golden set.

### ADR-002 — Reranker served as ONNX FP16 (INT8 optional), dynamic batching, tunable pool
- **Context:** The cross-encoder is the most expensive per-request stage (45 ms budget) and the biggest GPU cost driver (§1.4.5).
- **Decision:** Serve `bge-reranker-v2-m3` through ONNX Runtime/Triton as a **separate service**, FP16 by default, dynamic micro-batching (`batch=32, max_wait=8 ms`), `max_len=512`, sigmoid threshold 0.42. Allow INT8 and pool reduction (40 → 16–20) **only if** KPI-11/12/13 remain within threshold in A/B on the golden set.
- **Alternatives:** PyTorch eager (3–5× slower); in-process ONNX (couples GPU memory to API pods, no cross-request batching); skip rerank (fails precision KPIs).
- **Consequences:** network hop (≈ 2 ms) accepted for batching efficiency and bulkhead isolation; quantisation calibrated per release.

### ADR-003 — Redis key isolation & stampede control
- **Context:** The prototype leaked answers across users via shared keys.
- **Decision:** Every key is `{tenant_id}:…` hash-tagged (cluster slot locality) with digest `SHA256(tenant_id:user_id:norm_query)`; semantic index partitioned per tenant; TTL = 300 s ± 45 s jitter; single-flight lock per key; keys never derived from LLM output; ACL per service. Cross-tenant probe in CI (KPI-28).
- **Alternatives:** per-tenant Redis instances (clean but costly at 50+ tenants — kept as upgrade path for regulated tenants); app-level filtering only (rejected: one bug = leak).
- **Consequences:** lower cross-user hit rate (cache is per-user); mitigated by a tenant-scoped *shared-knowledge* tier for non-personalised answers, enabled per tenant policy.

### ADR-004 — Bounded ReAct as a deterministic state machine
- **Context:** Unbounded LLM-driven loops caused runaway tool calls.
- **Decision:** Control flow lives in code: `INIT → TRIAGE → REASONING → TOOL_CALL → OBSERVATION → VERIFY → DONE | HALTED`. Hard limits: 5 iterations, ≤120 ms per tool step, request deadline 600 / 3,500 ms by path (§1.4.3), no recursive self-call without state mutation; breaker on `(tool, args_hash)` ≥ 2 failures; `GoalAnchorBlock` pinned at index 0; observations compressed ≤180 tokens; only last 3 turns hot.
- **Alternatives:** free-form agent frameworks (non-deterministic, unbounded); single-shot RAG only (can't do transactions).
- **Consequences:** some complex tasks halt early; they return partial verified answers with explicit remediation. Metrics: KPI-16, 17, 18.

### ADR-005 — MCP as the single tool boundary (stdio local, Streamable HTTP distributed)
- **Decision:** All tools exposed via MCP; stdio for latency-critical local workers (`hybrid-rag`, `graphify-vault`), Streamable HTTP for scalable services (`state-store`). **Why:** uniform contracts, per-server bulkheads, independently deployable/signed servers. **Trade-off:** serialisation overhead (budget KPI-30 ≤ 4 ms).

### ADR-006 — Guardrails as Pluggy plugins, fail-closed, outside the LLM
- **Decision:** PII, injection, breaker, RAGAS and OTel are deterministic plugins registered by entry point with priority ordering and hot-toggle. **Why:** security controls must not depend on model behaviour and must be testable in isolation. **Trade-off:** plugin latency counts against the 15 ms pre-hook budget.

### ADR-007 — Graphify + Obsidian for context compression
- **Decision:** Atomic vault notes (YAML `id, domain, risk_tier, version` + `[[wikilinks]]`), `graphify build` → `graph.json`; at runtime resolve anchor node and traverse ≤ depth 2, token budget 320 (vs ~1,400 raw). **Why:** cuts input tokens 45–65%, reducing TTFT and cost (KPI-26, 29). **Trade-off:** vault curation cost; stale graph falls back to plain RAG.

### ADR-008 — Deterministic abstention short-circuit
- **Decision:** `top_score < 0.42` → template abstention, no LLM call (< 90 ms, ~80% token saving on those requests). Threshold raised to 0.55 by runbook during hallucination surges. **Trade-off:** may over-abstain; tracked by KPI-06/23.

---

## 2. Cross-Cutting Designs

### 2.1 Context retention & drift control (Gate 5)
- `GoalAnchorBlock{original_intent, normalized_entity_ids, active_policy_constraints}` immutable, index 0, prefix-cacheable.
- Extractive observation compressor: strip boilerplate JSON keys, keep salient fields, ≤ 180 tokens.
- Sliding window: last 3 turns in Redis; verified facts only → `session_facts`. Drift score = cosine(goal embedding, current step intent) ≥ 0.85 (KPI-18); below → re-anchor / halt.

### 2.2 Subagent topology (Gates 6–7)

| Subagent | Responsibility | MCP bindings | Guardrail |
|----------|---------------|--------------|-----------|
| Triage | Tag `(intent, domain, risk_tier, complexity)`, route | none | temp 0, ≤ 25 ms, distilled classifier |
| Retrieval | Deterministic query normalisation (no LLM rewrite on the hot path), 3-stage RAG, citations | `hybrid-rag`, `graphify-vault` | Read-only; no mutating tools |
| Execution | Schema validation, forms, workflow transitions | `state-store` | No RAG / web; Tier 3 → HITL freeze |
| Verification (a.k.a. "Audit" in the diagram brief) | Claim→chunk NLI, RAGAS triad, audit log | `state-store` (audit only) | Independent context; can veto |

### 2.3 Observability & incident response (Gates 10–11)
W3C `traceparent` propagated through every hop → OTel Collector → Prometheus/Tempo/Grafana. Spans: `hook.pre, cache.lookup, triage.classify, gateway.embed, mcp.rag.bm25, mcp.rag.faiss, mcp.rag.rrf, mcp.rag.rerank, agent.react.step, llm.generate, hook.post.ragas`. Telemetry export is asynchronous and non-blocking.

| Scenario | Alert trigger | Mitigation runbook |
|----------|---------------|--------------------|
| Latency breach | p95 > 550 ms for 30 s | Aggressive semantic cache (cosine ≥ 0.92); BM25/dense `top_k` 50 → 25 (then pool 40 → 16) |
| Hallucination surge | RAGAS faithfulness < 0.90 (5 m window) | Cross-encoder cutoff 0.42 → 0.55; strict extractive quote mode |
| MCP hang | JSON-RPC timeout > 110 ms at > 1% | Trip MCP bulkhead; fail over to cached read-replica index; restart worker |
| Cache stampede / leak | Tenant key collision or Redis CPU > 85% | Enforce key prefix; jittered TTL 300 ± 45 s; single-flight |

### 2.4 Governance, security-in-build, red teaming (Gates 13–14)
- SHA-256 hash-chained immutable audit log; `tenant_id` in every Postgres/Redis access (RLS + key prefix); zero raw PII persistence; signed MCP manifests; secrets via vault/KMS; least-privilege service accounts.
- 100-prompt red-team suite: direct jailbreak / prompt extraction; indirect injection hidden in Obsidian/PDF chunks; multi-turn crescendo; Unicode homoglyph & `name[at]domain` PII smuggling; tool-argument overflow / SQLi fuzzing. Runs in CI; failures block release.

### 2.5 Load-test strategy (Gates 15–16)
- k6: stages 2 m→10K, 3 m→50K, 5 m sustain, 2 m ramp-down; thresholds per `path` tag (from the `X-Path` response header): fast p50 < 150, p95 < 250, p99 < 400 ms; llm p50 < 1,300, p95 < 2,200, p99 < 3,000 ms (KPI-31..33); failure rate < 0.05%; check `tenant_id` echo for leaks. Think time per §1.4.4.
- JMeter: 1 controller + 10 workers × 5,000 threads, 180 s ramp, 600 s; duration assertion 600 ms (fast path) / 3,500 ms (LLM path); on the SSE `final` event, JSONPath `$.grounding_score ≥ 0.92` when `status = OK`; backend listener to Prometheus/InfluxDB.
- Also: 1-hour soak (KPI-36), chaos suite (Redis kill, MCP hang, LLM 429 injection, reranker slowdown) with SLO assertions.
- Test environment mirrors production topology. **LLM handling (D-6):** the 50K-VU SLA run uses `llm-sim`, a service behind the same `LLMClient` interface that replays the TTFT / tokens-per-second / 429 distribution *measured from the real model in the Phase 0 benchmark*. It must not be tuned to pass; the replayed distribution is published with the results. A separate, quota-bounded soak runs against the real model to validate the distribution. Every other component (gateway, Redis, Postgres, RAG, reranker, plugins) is real in the SLA run.

### 2.6 Developer workbench (Gate 18) — Streamlit tabs
1 Architecture & Workflow (this design checklist, diagram, triage console, ReAct trace, HITL queue) · 2 3-Stage RAG Lab (sliders `k1, b, RRF k, threshold`) · 3 MCP & Plugin Inspector (status, JSON-RPC payloads, hot-toggles) · 4 Memory & Graphify Tokens · 5 Red Team & Circuit Breaker · 6 36-KPI Scorecard & 50K Load · 7 Feature Validators (§2.7).

**Navigation requirement (D-19).** Streamlit's `st.tabs` executes the body of *every* tab on every rerun, so "run when the tab is opened" cannot be detected with it. The workbench uses page-style navigation (`st.navigation` / a sidebar selector) so that a page's code runs only while it is selected. "Tab" in this document means one such page.

### 2.7 Feature validators (decision D-19)

**Purpose.** Every functional requirement (FR-01..FR-12) and its KPIs gets one independent validator that exercises the *running system* and reports a verdict, so a regression in one feature is visible on its own and not only as a movement in an aggregate KPI.

**Principles**
1. **Agents write, run and triage; deterministic assertions decide.** A validator agent may generate test inputs, vary prompts, and summarise failures. PASS/FAIL comes from code-level assertions and thresholds (for example "0 cross-tenant canary strings", "KPI-19 = 100%"), never from an LLM's opinion. The only LLM-judged measures are KPI-20..24, and those use the pinned judge of §1.2.11 with a human spot-check. This keeps non-goal 6 (no LLM-authored security decisions) intact.
2. **Independence.** Validators live in their own package (`validators/`), with their own prompts and fixtures, import no agent code, and talk to the system only through the public API (§1.2.12), read-only Prometheus/Tempo queries and read-only audit views. The judge model differs from the generation model.
3. **Validators are themselves validated.** Each ships *seeded faults* (a flag-controlled fault on a sandbox tenant or in `llm-sim`, e.g. "cache key drops `tenant_id`", "breaker threshold = 99"). A run includes at least two per validator; if a seeded fault is not caught the validator's verdict is `UNTRUSTED`, whatever else it reports.
4. **Test traffic is isolated.** Validators use dedicated tenants (`tenant_validator_*`, plus a sandbox tenant for Tier 3), tagged in traces and excluded from production KPI statistics. They never touch a customer tenant.

**Validator matrix**

| ID | Validator | Features | KPIs | Deterministic assertions (gate) | Quick profile (tab open) | Delivered in |
|----|-----------|----------|------|--------------------------------|--------------------------|--------------|
| V1 | Contract & harness | FR-01, FR-07 | 14, 15 | every error code and HTTP mapping of §1.2.6 reachable; strict validation; T3 repairs flagged; KPI-14 ≥ 96%, KPI-15 ≥ 90% | ~40 generated valid/malformed payloads | Phase 1 |
| V2 | Isolation & cache | FR-01, FR-03 | 27, 28 | **0** cross-tenant canary leaks; token≠body tenant → -32001; exact and semantic tiers hit only within tenant/user scope; KPI-27 under the §1.4.4 mix | 5 tenants × concurrent requests with planted canary strings | Phase 1 (isolation), 4 (cache) |
| V3 | Guardrails | FR-02 | 1, 2, 3, 6 | KPI-02 ≥ 99.8%, KPI-03 ≥ 99.5%, KPI-06 ≤ 1%, `hook.pre` p95 ≤ 15 ms; fail-closed on plugin error | ~60 sampled prompts (indicative only) | Phase 1 |
| V4 | Routing & retrieval | FR-04, FR-05, FR-06 | 4, 5, 7–13, 23 | triage accuracy/latency; stage latencies; recall@pool, precision@4, MRR, nDCG; abstention accuracy; ≤ 4 cited chunks | ~25 golden + 10 unanswerable queries | Phase 2 |
| V5 | Loop & breaker | FR-08 | 16, 17, 18 | breaker opens at 2 identical failures; `HALTED_BUDGET` before the path's request deadline; ≤ 5 iterations; mean steps ≤ 2.2; **KPI-16 = 100%** | ~10 fault-injection scenarios (repeat bad SKU, MCP timeout, malformed LLM output via `llm-sim`) | Phase 3 |
| V6 | HITL gate | FR-09 | 19 | **no Tier 3 commit without an approved record**; self-approval rejected; expiry never auto-approves; replay with the same key is idempotent; repaired args shown; **KPI-19 = 100%** | ~8 scenarios on the sandbox tenant | Phase 3 |
| V7 | Grounding | FR-10 | 20, 21, 22, 24 | veto triggers `retract`; KPI-20..22 on the golden set; KPI-24 claim-level rate with the §1.5 sample size | ≤ 20 claims judged (budget-capped; `SKIPPED` if the judge quota is exhausted) | Phase 2 (RAGAS), 3 (veto) |
| V8 | Audit chain | FR-11 | 19 | every tenant chain verifies end to end, no gaps; anchors match; a tampered copy is detected | last N events per validator tenant + a tamper canary | Phase 3 |
| V9 | Load & chaos | FR-12 | 31–36 | latency, throughput, 5xx and RSS thresholds; `llm-sim` distribution vs real-model soak within tolerance | **smoke only**: ~45 s at low VUs to prove the harness and the percentile plumbing. The 50K-VU and soak results are shown from the last Full run | Phase 6 |
| V10 | Telemetry | FR-11 (OTel) | 1, 5, 25, 30 | every hop in §2.3 emits a span; `traceparent` continuity; the request path never blocks when the collector is down | 20 requests, then a collector-down request | Phase 5 |

**Run profiles**

| | Quick | Full |
|--|-------|------|
| Trigger | opening Tab 7 (or its "Re-run" button) | CI, pre-release, or a manual "Run full" button |
| Wall-clock budget | ≤ 90 s total, validators in parallel | hours (includes the 50K-VU run and soak) |
| Sample sizes | small, fixed (see matrix) | the §1.5 minimums |
| Generation model | `llm-sim` | real model for the functional suites, `llm-sim` for the SLA load run (D-6) |
| Can change a release gate? | **No.** Results are indicative: with small n a PASS carries a `WARN` when the confidence interval crosses the threshold | **Yes.** Only Full runs write the KPI gate results |

**Tab 7 behaviour (run on open, not continuously)**
- On opening the page the workbench calls `POST /api/v1/validators/runs {profile: "quick"}`; the job runs in a worker, **not in the Streamlit process**, and results appear per validator as each finishes. The page polls only while its own run is in progress; when the run completes (or the user leaves the page) polling stops. There is no scheduler and no background refresh.
- A guard in `st.session_state` ensures one run per page open, not one per rerun. Leaving the page cancels an unfinished run. If a Quick run for the same tenant finished less than 60 s ago, it is shown instead of starting a new one. Limits: one active Quick run per tenant, two globally; extra requests show "queued".
- "Re-run" starts a new Quick run. "Run full" is shown only to `sre`/`ml_qa`, asks for confirmation, and shows the estimated duration and model spend.
- Heavy suites are never started by opening the page. V9 shows its smoke result plus the **last Full result with its date and age**.

**What the tab shows for each validator ("agent performance")**
- **Verdict** `PASS` / `FAIL` / `WARN` / `SKIPPED` (budget, quota or disabled) / `UNTRUSTED` (seeded fault missed), and checks passed / total.
- **Metrics vs thresholds** (KPI value, threshold, n, 95% interval).
- **Detection quality:** seeded faults caught / total. This measures the validator itself.
- **Cost and speed:** its own duration, and for LLM-using validators input/output tokens and estimated spend.
- **Findings:** up to 20, each with a `trace_id` linking to Tempo; payload text is visible only to the `redteam` role.
- **History:** verdict and key metric of the last 10 Full runs (sparkline), plus the age of the last Full run.
- **Header:** run id, profile, started/finished, wall time against the 90 s budget, total tokens and cost.

```python
Verdict = Literal["PASS", "FAIL", "WARN", "SKIPPED", "UNTRUSTED"]

class MetricResult(Base):
    name: str                                   # e.g. "KPI-28 cross-tenant leaks"
    value: float; threshold: float
    comparator: Literal["<=", ">=", "=="]
    n: int
    ci95: tuple[float, float] | None = None

class Finding(Base):
    summary: str
    trace_id: str | None = None
    severity: Literal["critical", "major", "minor"]

class ValidatorResult(Base):
    validator_id: Annotated[str, StringConstraints(pattern=r"^V([1-9]|10)$")]
    verdict: Verdict
    checks_passed: int; checks_total: int
    metrics: list[MetricResult]
    seeded_faults_caught: int; seeded_faults_total: int
    duration_ms: int
    llm_tokens_in: int = 0; llm_tokens_out: int = 0; est_cost_usd: float = 0.0
    findings: list[Finding] = Field(default_factory=list, max_length=20)

class ValidatorRun(Base):
    run_id: UUID4
    profile: Literal["quick", "full"]
    status: Literal["QUEUED", "RUNNING", "DONE", "CANCELLED"]
    results: list[ValidatorResult]              # partial while RUNNING
```

**API (added to §1.2.12).** `POST /api/v1/validators/runs` → 202 `{run_id}` (quick: `ml_qa`/`sre`/`redteam`; full: `sre`/`ml_qa`); `GET /api/v1/validators/runs/{id}` → `ValidatorRun`; `DELETE /api/v1/validators/runs/{id}` cancels; `GET /api/v1/validators/runs?profile=full&limit=10` returns history.

**Open questions for this section:** the exact seeded-fault mechanism per validator (flag vs sandbox build), the per-run model-spend cap, and whether the history store is Postgres or Prometheus.

---

## 3. Delivery Plan

| Phase | Output | Exit criterion |
|-------|--------|----------------|
| 0 | `design.md` locked, diagram; **model latency benchmark** (hosted candidates at 1.5K-token prompts) and calibrated `llm-sim` distribution | Reviewer sign-off; R-5 gate passed or an escape option chosen |
| 1 | Skeleton: gateway, FastAPI, plugins (PII, injection), contracts, harness | Contract tests green; KPI-01/02/14 |
| 2 | `mcp-hybrid-rag` 3-stage + golden set + RAGAS | KPI-07..13, 20..23 |
| 3 | State machine, breaker, HITL, subagents, state-store | KPI-16..19, FMEA #12 chaos tests |
| 4 | Graphify/Obsidian, cache, token KPIs | KPI-26..29 |
| 5 | OTel + dashboards + workbench | All spans visible; Tab 6 live; Tab 7 live with V1–V8 and V10 passing a Quick run and catching their seeded faults (V9 joins in Phase 6; each validator is delivered with its feature, see §2.7) |
| 6 | Red team + load/soak at 50K | KPI-31..36, red-team clean |
| 7 | Hardening, runbook drill, sign-off | 36/36 KPI PASS |

---

## 4. Risks, Assumptions & Open Issues

**Risks**
- **R-1 Reranker GPU cost/throughput** at 40-pair pools (§1.4.5). Mitigation: pool reduction, INT8, caching; benchmark in Phase 2.
- **R-2 Fast-path share drives the user-visible latency.** With a ≈ 600 ms LLM first-token floor (R-5), the mean experience depends on how many requests avoid the LLM (cache ≈ 35%, abstention ≈ 10%, extractive ≈ 13%: all planning assumptions). Mitigation: measure the extractive path's precision on the golden set before enabling it; report the fast-path share as a first-class metric; evidence-first streaming for the rest.
- **R-3 Single-tenant cache hit ratio** (35%) depends on query repetition per user; may need the shared-knowledge tier (ADR-003).
- **R-4 Cost KPI-29 (≤ $0.45/1K)** is now dominated by hosted-model token cost (≈ 1.4K in + ≤ 96 out per generated answer, minus cache hits and abstentions). Compute from a price table in config once the model is chosen; encoder GPUs are the smaller term.
- **R-5 Hosted-model latency (resolved by D-23, option 3; residual risk below).** Phase 0 (2026-11-18, one client location, n = 6–8 per cell, through OpenRouter, ≈ 1.4K-token prompt, ≤ 96 output tokens) measured:
  - Baseline: `claude-sonnet-5.5` TTFT p50 ≈ 914 ms / p95 ≈ 1,315 ms at ≈ 193 tok/s; `claude-haiku-4.5` TTFT p50 ≈ 956 ms / p95 ≈ 1,343 ms at ≈ 201 tok/s (≈ 100 tok/s with latency-sorted routing). Against the old gate (TTFT ≲ 59 ms, ≳ 420 tok/s) the hosted path missed by ≈ 15× and ≈ 2×.
  - **What helped (within the account's guardrails):** disabling reasoning on `claude-haiku-5.5` cut TTFT p50 from ≈ 1.2 s to ≈ 0.65–0.69 s and gave ≈ 300 tok/s (now the default, `generation_extra_body`).
  - **What did not:** the ≈ 600 ms floor is server-side (client connect 4 ms, TTFB 35 ms; 1-token reply to a tiny prompt: p50 605 ms); prompt size from 350 to 1,400 tokens moved TTFT only ≈ 580 → 650 ms; Anthropic `cache_control` on the prefix showed no gain; hedging Haiku 5.5 ∥ 4.5 moved p50 685 → 658 ms and p95 1,108 → 1,078 ms (not worth doubling cost); latency-sorted provider routing gave no consistent gain.
  - **Guardrails:** the OpenRouter account excludes non-Anthropic endpoints (Gemini, DeepSeek, Mistral, Llama returned 404), and they cannot be loosened (owner decision). The candidate set is therefore `claude-haiku-5.5` (generation), `claude-haiku-4.5` (fallback), `claude-sonnet-5.5` (judge).
  - **Outcome (D-23):** keep the tight targets only for fast paths; LLM-path targets are re-derived from these measurements (§1.4, KPI-25, KPI-31..33). **Residual risk:** samples are small and from one location; re-run with ≥ 100 samples per cell, at different times of day and under concurrency, before the Phase 0 sign-off. The targets are provisional until then.

**Inconsistencies found in the source blueprint & how this draft resolves them (need owner confirmation)**

| ID | Inconsistency | Draft decision |
|----|---------------|----------------|
| O-1 | Gate 0 says FAISS **IVF-PQ**; Gate 3 says FAISS **HNSW (efSearch=64)** | Use HNSW (latency); IVF-PQ only if memory-bound |
| O-2 | Reranker described as FP16 and INT8; "in-process" vs microservice | Separate ONNX/Triton service, FP16 default, INT8 optional (ADR-002) |
| O-3 | k6 uses `sleep(1)` at 50K VUs (≈ 35K RPS) but KPI-35 expects ≥ 8,500 RPS | Think time ≈ 5.4 s for SLA run; `sleep(1)` as stress profile |
| O-4 | k6 `p(50)<250` vs KPI-31 ≤ 240 ms | Use 240 |
| O-5 | KPI-33 ≤ 590 vs k6 `p(99)<600` | v0.1: gate on 590, 600 outer SLA. **Superseded by D-23** (per-path p99: 400 ms fast / 3,000 ms LLM) |
| O-6 | Token reduction 55% (Gate 0) vs 45–65% (Gate 17) vs KPI-26 ≥ 45% | Gate ≥ 45%, target 55% |
| O-7 | Triage budget ≤ 30 ms (Gate 2) vs ≤ 25 ms (KPI-05, Gate 6) | 25 ms |
| O-8 | "Audit Subagent" (Gate 2) vs "Verification Subagent" (Gate 7) | Single Verification subagent |
| O-9 | Scorecard in Tab 5 (Gate 12) vs Tab 6 (Gate 18) | Tab 6 |
| O-10 | Stage latency sum (triage 25 + RAG ≈ 85 + LLM) vs 450 ms p95 not shown in blueprint | Explicit per-path budget in §1.4.1 (LLM path: TTFT ≈ 1.2 s, E2E ≈ 1.7 s p95; fast paths ≤ 250 ms) after D-23 |
| O-11 | Gate 3 pool = 40 for rerank, KPI-10 measures recall@50 | v0.1: recall@50 on the fused set. **v0.2:** KPI-10 is recall@pool (what the reranker actually sees); recall@50 is a diagnostic |
| O-12 | E2E latency definition (full stream vs first token) unspecified | Defined in §1.4 (last byte, ≤ 96 tokens) |
| O-13 | `Post_Hardening_Engineering_Audit_Report.docx` states measured results (185/390/540 ms, 9,240 RPS, 52.4% token reduction, "all 36 KPIs PASSED") but this design is pre-build | Treat the report's numbers as targets/narrative, not measurements; design.md is authoritative until the Phase 6–7 runs produce real data |
| O-14 | Audit report: "4-stage sequential" pipeline | 3 retrieval stages with BM25 ∥ dense (RRF is a fusion step, not a stage) |
| O-15 | Span names differ (`rag.dense`/`triage`/`ragas` vs `mcp.rag.*`/`triage.classify`/`hook.post.ragas`) | design.md §2.3 names are canonical (now `gateway.embed`, `mcp.rag.faiss`) |
| O-16 | Audit report lists TensorRT-LLM as roadmap; v0.1 offered vLLM/TRT-LLM | Generation is a hosted mid-tier model (D-4); self-hosted serving is only the R-5 fallback |

### 4.1 Decisions log (second-pass review, v0.2 in progress)

| ID | Decision | Status |
|----|----------|--------|
| D-1 | Verification veto: stream tokens, then `retract` + `final` trailer event | Decided; stream event schema to write |
| D-2 | Cache: gateway-direct Redis, exact + per-tenant semantic tier; shared-knowledge tier deferred to v1.1; reconcile KPI-27 (35%) with the 10% repeat-query load mix | Decided; budget re-sum and KPI-27 reconciliation open |
| D-3 | Tier 3 approvals: encrypted payload vault with TTL, `UNIQUE (tenant_id, idempotency_key)` | Decided |
| Fixed | v0.1 TTFT arithmetic corrected (149 → 166 ms), then superseded by the per-path budget of D-23; `Citation` ordering; `UUID4`; diagram caption (7 zones); response fields (`grounding_score`, statuses, `approval_id`, `remediation`) | Done |
| D-4 | Generation (and off-path judge / claim audit) uses a hosted mid-tier model behind `LLMClient`; encoders, triage and the inline verifier stay self-hosted (§1.2.11) | Decided; model choice by Phase 0 |
| D-5 | Prompt = anchor + 4 extractively compressed chunks + optional Graphify + hot window, cap 1,500 tokens; verification on full chunks (§1.2.8) | Decided |
| D-6 | 50K-VU SLA run uses a calibrated `llm-sim` replaying measured model latency/429s; quota-bounded real-model soak validates it; no tuning to pass (§2.5) | Decided |
| D-7 | Caller × tool matrix; cache/hot-window/session-fact/audit owned by platform code (§1.2.9) | Decided |
| D-8 | ~~OIDC/JWT at the edge, tenant from claims~~ | **Superseded by D-22** |
| D-9 | Streamlit (7 tabs as pages) is a client of the real FastAPI service; load test hits the API | Decided |
| D-10 | ~~Keep 200–600 ms E2E / p95 ≤ 450 ms as written~~ | **Superseded by D-23** |
| D-11 | External API surface and SSE event schema (§1.2.12) | Decided |
| D-12 | Request deadline model: 600 ms (fast paths) / 3,500 ms (LLM path); MCP timeouts 110/100/40 ms below the 120 ms tool-step budget; generation governed by a 2,000 ms first-token timeout + the deadline; no LLM query rewrite on the hot path (§1.4.3) | Decided (revised by D-23) |
| D-13 | Load mix has 40% repeats so KPI-27 is reachable; KPI-27 gated under that mix, informational in production; sizing includes a 20%-hit sensitivity case (§1.4.4) | Decided |
| D-14 | KPI-20 aggregate vs KPI-24 claim-level; minimum test-set sizes by Clopper–Pearson (§1.5) | Decided |
| D-15 | Per-tenant audit hash chains with sequencer, batching, 5-min anchors (§1.3.4) | Decided |
| D-16 | Ingestion/index lifecycle; tenant-level ACLs only in v1 (§1.3.5) | Decided |
| D-17 | Breaker counts failures, scope tenant + request + 60 s session, no half-open in v1; harness-minted idempotency keys; T3 repairs shown to reviewer (§1.2.5) | Decided |
| D-18 | Abstention threshold server-side; on reranker outage default to abstain (`require_verified_evidence=true`); `rank-bm25` demoted to RAG Lab, production uses a configurable inverted-index engine (§0.3, FMEA #2) | Decided; BM25 engine finalised in Phase 2 |
| D-21 | All LLM calls go through OpenRouter behind `LLMClient`; key from the secret store, per-environment keys with spend limits, never in the repo or logs; benchmark run through OpenRouter (§1.2.11) | Decided; key must be rotated (see below) |
| D-22 | **No authentication.** No login, JWT or OIDC; identity is caller-declared via `X-Tenant-Id` (must equal body) and `X-Role`; isolation guarantees assume honest tenant declarations; add an authenticating proxy before any real deployment (§1.2.10) | Decided (owner) |
| D-23 | **Per-path latency targets (option 3).** Fast paths (cache hit, abstention, refusal, extractive) keep tight targets (p95 ≤ 250 ms); LLM path targets are re-derived from Phase 0: TTFT p95 ≤ 1,500 ms, E2E p50/p95/p99 ≤ 1,300 / 2,200 / 3,000 ms. New within-guardrail features: reasoning off, extractive fast path (FR-13), evidence-first streaming; hedging and prefix caching rejected on evidence (R-5) | Decided (owner); provisional until a larger benchmark |
| D-24 | In-memory backends behind the same interfaces for Redis (cache, hot window) and Postgres (facts, approvals, audit, chunks) in development and CI; contract tests are shared with the future real backends (§1.3.6) | Decided (owner); implemented in `src/kestrel/stores/` |
| D-19 | One independent validator per feature (V1–V10, §2.7): agents generate/run/triage, deterministic assertions decide; seeded faults make validators self-checking; workbench uses page navigation so Tab 7 can detect "opened" | Decided |
| D-20 | Tab 7 runs a Quick profile (≤ 90 s, indicative, cannot change a release gate) once per page open, no background refresh; only Full runs (CI/manual) write KPI gate results | Decided; seeded-fault mechanism, spend cap and history store open |
| Open | Phase 0 model benchmark (R-5); 8,500 RPS FastAPI/Redis/PG sizing re-check; remaining housekeeping (section numbering) | To do |

**Assumptions:** GPU class L4/A10 for encoders and the inline NLI verifier; generation uses a hosted mid-tier model (D-4, replaces the v0.1 "~8B FP8 self-hosted, no external LLM API" assumption); 50 tenants, English corpus, per-user cache scoping; embedding, reranker, triage and inline verifier self-hosted.

---
*End of design.md v0.1 — open for review before lock.*
