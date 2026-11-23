"""Orchestrator: the request pipeline of design.md §1.3.2.

pre-hooks -> embed once -> cache (exact | per-tenant semantic) -> triage -> route
  REJECT_OOS  -> deterministic refusal                           (fast path)
  DIRECT_ANSWER -> LLM                                            (LLM path)
  HYBRID_RAG  -> retrieve -> abstain (< threshold)               (fast path)
                          -> extractive answer, no LLM (>= θ_ext) (fast path)
                          -> evidence event, LLM, inline verify   (LLM path)
  TRANSACTION_EXEC -> not implemented in this build (explicit REFUSED with remediation)

`handle()` does all pre-LLM work and returns the *path* before any byte is streamed, so the gateway can set
`X-Path`. The returned event iterator covers the remainder.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .answering import SYSTEM_PROMPT, best_sentence, build_messages, extractive_answer, grounding_score
from .config import Settings
from .approvals import ApprovalService
from .contracts import AgentQueryRequest, AgentQueryResponse, Citation, ErrorCode, PlatformError
from .execution import ExecutionAgent, ExecutionBackend
from .harness import ToolExecutionHarness
from .embed import Embedder
from .llm.client import LLMClient
from .plugins import PreContext, PrePipeline
from .retrieval import HybridRetriever
from .retrieval.retriever import Retrieval
from .stores import InMemoryCache, InMemoryStateStore
from .triage import RuleTriage, Triage

log = logging.getLogger("kestrel.pipeline")
Event = tuple[str, dict[str, Any]]

ABSTAIN_TEXT = "I could not find this in the available knowledge base, so I will not guess."
REFUSAL_TEXT = "This request is outside what this assistant can help with."
PERSONAL_TEXT = ("I cannot see your personal account details (policy, claim or payment data). "
                 "I can answer general questions about Acme Mutual products and policies, or file a request for review.")
TXN_TEXT = ("Workflow and form actions need the human-approval path, which is not available in this build yet. "
            "No action was taken.")
RETRACT_TEXT = "I could not verify that answer against the sources, so I am withholding it."


@dataclass
class Handled:
    path: str                       # "fast" | "llm"
    route: str
    events: AsyncIterator[Event]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Orchestrator:
    def __init__(self, settings: Settings, llm: LLMClient | None, embedder: Embedder, retriever: HybridRetriever,
                 cache: InMemoryCache, state: InMemoryStateStore, pre: PrePipeline, triage: Triage | None = None,
                 execution: ExecutionAgent | None = None, approvals: ApprovalService | None = None):
        self.s, self.llm, self.embedder, self.retriever = settings, llm, embedder, retriever
        self.cache, self.state, self.pre = cache, state, pre
        self.triage = triage or RuleTriage()
        self.harness = ToolExecutionHarness(ExecutionBackend(state))
        self.execution = execution or ExecutionAgent(self.harness, state)
        self.approvals = approvals or ApprovalService(state)
        self.stats = {"fast": 0, "llm": 0}

    # ------------------------------------------------------------------ public
    async def handle(self, req: AgentQueryRequest, trace_id: str | None = None) -> Handled:
        trace_id = trace_id or uuid.uuid4().hex
        T: dict[str, float] = {}
        t_all = time.perf_counter()
        ctx = self.pre.run(PreContext(req.tenant_id, req.query))        # may raise -32003 / -32004
        T["hook.pre"] = ctx.timings_ms.get("hook.pre", 0.0)
        query = ctx.text
        qhash = hashlib.sha256(query.encode()).hexdigest()[:16]

        t = time.perf_counter()
        hit = await self.cache.get(req.tenant_id, req.user_id, query)       # exact tier first: needs no embedding
        T["cache.lookup"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        emb = None
        if not hit:                                                         # embed once; reused by dense retrieval
            emb = (await self.embedder.embed([query]))[0]
            T["gateway.embed"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            hit = await self.cache.get(req.tenant_id, req.user_id, query, emb)   # semantic tier (per tenant+user)
            T["cache.lookup"] += (time.perf_counter() - t) * 1000
        if hit:
            payload = dict(hit.payload, trace_id=trace_id)
            return self._fast(req, "cache", payload["route"], payload, trace_id, qhash, T, t_all,
                              meta_extra={"cache": hit.tier})

        t = time.perf_counter()
        tri = self.triage.classify(query)
        T["triage.classify"] = (time.perf_counter() - t) * 1000

        if tri.route == "REJECT_OOS":
            text = PERSONAL_TEXT if tri.reason == "personal_account" else REFUSAL_TEXT
            return self._fast(req, "refusal", "REJECT_OOS", self._final(req, text, [], "REJECT_OOS", "REFUSED", trace_id), trace_id, qhash, T, t_all)
        if tri.route == "TRANSACTION_EXEC":
            t = time.perf_counter()
            try:
                x = await self.execution.request(req, query, trace_id)          # parse -> harness -> PENDING_APPROVAL
            finally:
                self.harness.breaker.end_request(trace_id)
            T["execution.request"] = (time.perf_counter() - t) * 1000
            note = "".join(f" I read '{a}' as {b}." for a, b in x["repaired"].values())
            f = self._final(req, f"Your request to {x['summary']} is pending reviewer approval (reference {x['approval_id']}). "
                                 f"No action has been taken yet.{note}", [], "TRANSACTION_EXEC", "PENDING_APPROVAL", trace_id)
            f["approval_id"] = x["approval_id"]
            if x["repaired"]:
                f["remediation"] = "Please confirm the interpreted value(s): " + "; ".join(f"{k}: '{a}' -> {b}" for k, (a, b) in x["repaired"].items())
            return self._fast(req, "pending_approval", "TRANSACTION_EXEC", f, trace_id, qhash, T, t_all)
        if tri.route == "DIRECT_ANSWER":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT.split(" Cite")[0] + ". Answer briefly."},
                    {"role": "user", "content": query}]
            return self._llm(req, "DIRECT_ANSWER", msgs, [], None, emb, query, trace_id, qhash, T, t_all)

        # HYBRID_RAG
        r = await self.retriever.retrieve(req.tenant_id, query, emb)
        T.update({f"mcp.rag.{k}": v for k, v in r.stage_ms.items()})
        if r.abstain:
            return self._fast(req, "abstain", "HYBRID_RAG", self._final(req, ABSTAIN_TEXT, [], "HYBRID_RAG", "ABSTAINED", trace_id), trace_id, qhash, T, t_all)

        cites = [Citation(chunk_id=c.chunk.chunk_id, source_uri=c.chunk.source, score=round(min(max(c.rerank_score, 0.0), 1.0), 4)) for c in r.chunks]
        if self.s.extractive_fast_path and tri.complexity == "low" and r.top_score >= self.s.extractive_threshold:
            ex = extractive_answer(query, r.chunks)
            if ex:
                f = self._final(req, ex.quote, cites[:1], "HYBRID_RAG", "OK", trace_id, grounding=1.0)
                f["mode"] = "extractive"
                await self.cache.put(req.tenant_id, req.user_id, query, f, emb)    # verbatim answers are verified by construction
                return self._fast(req, "extractive", "HYBRID_RAG", f, trace_id, qhash, T, t_all)
        if self.llm is None or not self.s.generation_model:
            raise PlatformError(ErrorCode.INVALID_REQUEST, "generation is not configured")
        msgs = build_messages(query, r.chunks)
        return self._llm(req, "HYBRID_RAG", msgs, r.chunks, r, emb, query, trace_id, qhash, T, t_all, cites=cites)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _final(req: AgentQueryRequest, answer: str, cites: list[Citation], route: str, status: str, trace_id: str,
               grounding: float | None = None) -> dict[str, Any]:
        return AgentQueryResponse(tenant_id=req.tenant_id, answer=answer, citations=cites, route=route,  # type: ignore[arg-type]
                                  status=status, grounding_score=grounding, trace_id=trace_id).model_dump()

    def _fast(self, req, kind, route, final, trace_id, qhash, T, t_all, meta_extra=None) -> Handled:
        self.stats["fast"] += 1
        T["total_pre_llm"] = (time.perf_counter() - t_all) * 1000
        meta = {"trace_id": trace_id, "route": route, "path": "fast", "kind": kind, "timings_ms": {k: round(v, 2) for k, v in T.items()}, **(meta_extra or {})}

        async def events() -> AsyncIterator[Event]:
            yield "meta", meta
            yield "final", final
            await self._after(req, final, qhash, "fast", kind, cache_tier=(meta_extra or {}).get("cache"))

        return Handled("fast", route, events(), T)

    def _llm(self, req, route, msgs, chunks, retrieval: Retrieval | None, emb, query, trace_id, qhash, T, t_all,
             cites: list[Citation] | None = None) -> Handled:
        self.stats["llm"] += 1
        T["total_pre_llm"] = (time.perf_counter() - t_all) * 1000
        meta = {"trace_id": trace_id, "route": route, "path": "llm", "timings_ms": {k: round(v, 2) for k, v in T.items()}}
        cites = cites or []

        async def events() -> AsyncIterator[Event]:
            yield "meta", meta
            if chunks:                                     # evidence-first: verbatim best sentence, before any token
                b = best_sentence(query, chunks[:2])
                if b:
                    yield "evidence", {"chunk_id": b[1].chunk.chunk_id, "quote": b[0]}
            parts: list[str] = []
            t0 = time.perf_counter()
            try:
                async for ev in self.llm.stream(msgs, model=self.s.generation_model, max_tokens=self.s.max_output_tokens,
                                                extra_body=self.s.generation_extra_body):
                    if ev.kind == "token":
                        if not parts:
                            T["llm.first_token"] = (time.perf_counter() - t0) * 1000
                        parts.append(ev.text)
                        yield "token", {"t": ev.text}
            except PlatformError as e:
                yield "error", e.to_dict()
                return
            answer = "".join(parts).strip()
            texts = [c.chunk.text for c in chunks]
            g = grounding_score(answer, texts) if chunks else None
            if chunks and (not answer or g < self.s.grounding_min):        # veto: retract, then abstain deterministically
                yield "retract", {"reason": "ungrounded", "grounding_score": round(g, 3)}
                final = self._final(req, RETRACT_TEXT, [], route, "ABSTAINED", meta["trace_id"], grounding=g)
            else:
                final = self._final(req, answer, cites, route, "OK", meta["trace_id"], grounding=g)
                await self.cache.put(req.tenant_id, req.user_id, query, final, emb)   # only verified answers are cached
            yield "final", final
            await self._after(req, final, qhash, "llm", "llm", T=T)

        return Handled("llm", route, events(), T)

    async def _after(self, req, final, qhash, path, kind, cache_tier=None, T=None) -> None:
        """Post-response bookkeeping that must never fail the request: hot window + audit."""
        try:
            await self.cache.push_turn(req.tenant_id, req.session_id or "none", {"q": qhash, "status": final["status"]})
            await self.state.append_audit(req.tenant_id, "query.completed", {
                "trace_id": final["trace_id"], "route": final["route"], "status": final["status"], "path": path,
                "kind": kind, "query_sha256": qhash, "cache": cache_tier or "none"})
        except Exception:                                  # noqa: BLE001
            log.exception("post-response bookkeeping failed")
