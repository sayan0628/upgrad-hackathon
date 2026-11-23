"""Gateway (design.md §1.2.10 no-auth, §1.2.12, §1.3.2): header/body tenant check -> Orchestrator -> JSON or SSE.

SSE events: meta -> [evidence] -> token* -> [retract] -> final | error. Every response carries `X-Path: fast|llm`.
No authentication (D-22): identity is caller-declared; X-Tenant-Id must equal the body tenant.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from ..config import Settings
from ..contracts import AgentQueryRequest, ErrorCode, PlatformError
from ..embed import Embedder, HashEmbedder
from ..llm.client import LLMClient
from ..pipeline import Orchestrator
from ..plugins import InjectionShield, PiiRedactor, PrePipeline
from ..retrieval import HybridRetriever, IndexRegistry, LexicalReranker
from ..stores import InMemoryCache, InMemoryStateStore
from ..triage import Triage


def _identity(request: Request) -> dict[str, Any]:
    """No authentication (D-22) and no role-based access control: identity is caller-declared. X-Tenant-Id must be
    present and equal the body tenant (FR-01; catches misrouted clients, not a security control)."""
    tenant = request.headers.get("x-tenant-id")
    if not tenant:
        raise PlatformError(ErrorCode.TENANT_MISMATCH, "X-Tenant-Id header is required")
    return {"tenant_id": tenant, "user_id": request.headers.get("x-user-id")}


def _need_user(ident: dict[str, Any]) -> None:
    """Approve and reject are the only calls that need a user id: the reviewer must differ from the requester."""
    if not ident["user_id"]:
        raise PlatformError(ErrorCode.INVALID_PARAMS, "X-User-Id header is required")


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def build_orchestrator(settings: Settings, llm: LLMClient | None, *, embedder: Embedder | None = None,
                       registry: IndexRegistry | None = None, triage: Triage | None = None,
                       reranker=None, cache: InMemoryCache | None = None,
                       state: InMemoryStateStore | None = None) -> Orchestrator:
    embedder = embedder or HashEmbedder()
    registry = registry or IndexRegistry()
    retriever = HybridRetriever(registry, reranker or LexicalReranker(), pool=settings.rerank_pool,
                                threshold=settings.abstain_threshold)
    return Orchestrator(settings, llm, embedder, retriever, cache or InMemoryCache(), state or InMemoryStateStore(),
                        PrePipeline([PiiRedactor(), InjectionShield()]), triage)


def create_app(settings: Settings | None = None, llm: LLMClient | None = None,
               orchestrator: Orchestrator | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    orch = orchestrator or build_orchestrator(settings, llm)
    app = FastAPI(title="kestrel gateway", version="0.2.0")
    app.state.orchestrator = orch

    @app.exception_handler(PlatformError)
    async def _platform_error(_: Request, e: PlatformError) -> JSONResponse:
        return JSONResponse({"error": e.to_dict()}, status_code=e.http_status)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        checks = {"llm_client": orch.llm is not None, "generation_model": bool(settings.generation_model)}
        return JSONResponse({"ready": all(checks.values()), "checks": checks, "paths": orch.stats},
                            status_code=200 if all(checks.values()) else 503)

    @app.post("/api/v1/agent/query")
    async def query(request: Request):
        ident = _identity(request)
        try:
            body = AgentQueryRequest.model_validate(await request.json())
        except (ValidationError, ValueError) as e:
            raise PlatformError(ErrorCode.INVALID_PARAMS, "invalid request body",
                                {"errors": [{"loc": ".".join(map(str, x["loc"])), "msg": x["msg"]}
                                            for x in (e.errors() if isinstance(e, ValidationError) else [])]}) from e
        if body.tenant_id != ident["tenant_id"]:                                    # FR-01
            raise PlatformError(ErrorCode.TENANT_MISMATCH, "header and body tenant differ")
        tp = request.headers.get("traceparent", "").split("-")
        trace_id = tp[1] if len(tp) >= 4 else uuid.uuid4().hex

        h = await orch.handle(body, trace_id)                  # all pre-LLM work happens here; may raise -32003/-32004
        headers = {"X-Path": h.path}

        if "text/event-stream" in request.headers.get("accept", ""):
            async def gen() -> AsyncIterator[bytes]:
                async for name, data in h.events:
                    yield _sse(name, data)
            return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

        final: dict | None = None
        async for name, data in h.events:
            if name == "error":
                return JSONResponse({"error": data}, status_code=429 if data["code"] == ErrorCode.UPSTREAM_RATE_LIMITED else 502, headers=headers)
            if name == "final":
                final = data
        return JSONResponse(final, headers=headers, status_code=202 if final and final["status"] == "PENDING_APPROVAL" else 200)

    # ---- approval API (design.md §1.2.12). Tenant-scoped; reviewer != requester is enforced in ApprovalService.
    @app.get("/api/v1/approvals")
    async def list_approvals(request: Request, status: str = "PENDING"):
        ident = _identity(request)
        if status != "PENDING":
            raise PlatformError(ErrorCode.INVALID_PARAMS, "only status=PENDING is listed")
        return {"approvals": await orch.approvals.list_pending(ident["tenant_id"])}

    @app.get("/api/v1/approvals/{approval_id}")
    async def get_approval(approval_id: str, request: Request):
        ident = _identity(request)
        a = await orch.approvals.get(ident["tenant_id"], approval_id)
        if a is None:
            raise PlatformError(ErrorCode.INVALID_PARAMS, "unknown approval")
        return a

    @app.post("/api/v1/approvals/{approval_id}/approve")
    async def approve(approval_id: str, request: Request):
        ident = _identity(request)
        _need_user(ident)
        return await orch.approvals.approve(ident["tenant_id"], approval_id, ident["user_id"])

    @app.post("/api/v1/approvals/{approval_id}/reject")
    async def reject(approval_id: str, request: Request):
        ident = _identity(request)
        _need_user(ident)
        try:
            reason = (await request.json()).get("reason", "")
        except Exception:                                                        # noqa: BLE001
            reason = ""
        return await orch.approvals.reject(ident["tenant_id"], approval_id, ident["user_id"], str(reason)[:300])

    from ..validators.api import mount_validators          # lazy: validators import this module's helpers
    mount_validators(app, orch)
    return app
