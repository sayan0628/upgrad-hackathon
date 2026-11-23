"""LLMClient: the only component that talks to a model provider (design.md D-4, D-21).

OpenRouter exposes an OpenAI-compatible streaming chat-completions API. The key comes from the
environment (Settings) and is only ever placed in the Authorization header: it is not logged, not put in
exceptions, and not returned in any event.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

import httpx

from ..config import Settings
from ..contracts import ErrorCode, PlatformError


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None      # as reported by OpenRouter usage accounting, if present


@dataclass
class StreamEvent:
    kind: str                           # "token" | "reasoning" | "done"
    text: str = ""
    t_ms: float = 0.0                   # ms since request start
    provider: str | None = None         # upstream provider that served the request (OpenRouter reports it)
    model: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    ttft_ms: float | None = None
    total_ms: float | None = None


class LLMClient(Protocol):
    def stream(self, messages: list[dict[str, Any]], *, model: str, max_tokens: int,
               temperature: float = 0.0, extra_body: dict[str, Any] | None = None) -> AsyncIterator[StreamEvent]: ...


class OpenRouterClient:
    def __init__(self, settings: Settings, *, http: httpx.AsyncClient | None = None, timeout_s: float = 30.0):
        if not settings.openrouter_api_key:
            raise PlatformError(ErrorCode.INVALID_REQUEST, "OPENROUTER_API_KEY is not set")
        self._key = settings.openrouter_api_key
        self._url = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._owns_http = http is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def stream(self, messages: list[dict[str, Any]], *, model: str, max_tokens: int,
                     temperature: float = 0.0, extra_body: dict[str, Any] | None = None) -> AsyncIterator[StreamEvent]:
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens,
                "temperature": temperature, "usage": {"include": True}, **(extra_body or {})}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                   "X-Title": "kestrel"}
        t0 = time.perf_counter()
        ttft: float | None = None
        provider = served_model = None
        usage = LLMUsage()
        async with self._http.stream("POST", self._url, json=body, headers=headers) as r:
            if r.status_code == 429:
                raise PlatformError(ErrorCode.UPSTREAM_RATE_LIMITED, "LLM provider rate limited",
                                    {"retry_after": r.headers.get("retry-after")})
            if r.status_code >= 400:
                detail = (await r.aread())[:300].decode("utf-8", "replace")
                raise PlatformError(ErrorCode.INVALID_REQUEST, f"LLM provider error {r.status_code}",
                                    {"detail": detail.replace(self._key, "***")})
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue                       # SSE comments / keep-alives (": OPENROUTER PROCESSING")
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise PlatformError(ErrorCode.INVALID_REQUEST, "LLM stream error",
                                        {"detail": str(chunk["error"])[:300].replace(self._key, "***")})
                provider = chunk.get("provider") or provider
                served_model = chunk.get("model") or served_model
                if u := chunk.get("usage"):
                    usage = LLMUsage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost"))
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if think := (delta.get("reasoning") or delta.get("reasoning_content")):
                        yield StreamEvent("reasoning", think, (time.perf_counter() - t0) * 1000, provider, served_model)
                    piece = delta.get("content")
                    if piece:
                        now = (time.perf_counter() - t0) * 1000
                        ttft = ttft if ttft is not None else now
                        yield StreamEvent("token", piece, now, provider, served_model)
        total = (time.perf_counter() - t0) * 1000
        yield StreamEvent("done", "", total, provider, served_model, usage, ttft, total)
