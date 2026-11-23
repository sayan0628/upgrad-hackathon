"""HTTP client helpers for the workbench (design.md D-9: a client of the real API, no kestrel imports).

Pure helpers (header building, SSE parsing, error formatting, percentiles) live here so they can be
unit-tested without a network. Streamlit pages import from this module.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TENANT = "tenant_acme"
DEFAULT_USER = "user_demo"
TENANT_RE = re.compile(r"^tenant_[A-Za-z0-9_-]{1,64}$")
USER_RE = re.compile(r"^user_[A-Za-z0-9_-]{1,64}$")
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 60.0          # streaming read timeout (design: ~60 s)


# ---------------------------------------------------------------- validation and headers
def validate_ids(tenant: str, user: str) -> list[str]:
    """Return a list of human-readable problems with the tenant / user id fields (empty if valid)."""
    problems = []
    if not TENANT_RE.match(tenant or ""):
        problems.append("Tenant must look like tenant_acme (letters, digits, _ or -, max 64 chars).")
    if not USER_RE.match(user or ""):
        problems.append("User id must match ^user_[A-Za-z0-9_-]{1,64}$, e.g. user_demo.")
    return problems


def api_headers(tenant: str, user: str | None = None) -> dict[str, str]:
    """Headers for an API call: X-Tenant-Id always; X-User-Id only when given (approve/reject need it).
    No role header: the platform has no role-based access control. No secrets are ever added."""
    headers = {"X-Tenant-Id": tenant}
    if user:
        headers["X-User-Id"] = user
    return headers



# ---------------------------------------------------------------- SSE parsing
@dataclass
class SSEEvent:
    event: str
    data: Any
    arrived_ms: float = 0.0          # ms since the request was sent


class SSEParser:
    """Incremental parser for text/event-stream bytes. Feed chunks; get complete events back.

    Handles events split across chunks, CRLF or LF line endings, comment lines (leading ':'),
    and multi-line data fields. Events without a name default to 'message'.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._event = "message"
        self._data: list[str] = []

    def feed(self, chunk: bytes | str, now_ms: float = 0.0) -> list[SSEEvent]:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        self._buf += text.replace("\r\n", "\n").replace("\r", "\n")
        out: list[SSEEvent] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line == "":
                if self._data:
                    out.append(self._make(now_ms))
                self._event, self._data = "message", []
                continue
            if line.startswith(":"):
                continue
            field_name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field_name == "event":
                self._event = value
            elif field_name == "data":
                self._data.append(value)
        return out

    def _make(self, now_ms: float) -> SSEEvent:
        raw = "\n".join(self._data)
        try:
            data = json.loads(raw)
        except ValueError:
            data = {"raw": raw}
        return SSEEvent(self._event, data, now_ms)


def parse_sse_bytes(payload: bytes) -> list[SSEEvent]:
    """Parse a complete SSE body in one go (used by tests and by non-streaming helpers)."""
    parser = SSEParser()
    events = parser.feed(payload, 0.0)
    return events


# ---------------------------------------------------------------- errors
def error_message(status: int | None, body: Any = None, exc: Exception | None = None) -> str:
    """Turn an HTTP status / gateway error body / exception into one readable sentence."""
    if exc is not None:
        if isinstance(exc, httpx.ConnectError):
            return "Server unreachable. Is the gateway running at the URL in the sidebar?"
        if isinstance(exc, httpx.TimeoutException):
            return "Timed out waiting for the server (no data within the read timeout)."
        return f"Request failed: {type(exc).__name__}."
    detail = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("name") or "")
        elif isinstance(body.get("detail"), str):
            detail = body["detail"]
    if status == 403:
        return f"403 Forbidden: {detail or 'tenant mismatch (check X-Tenant-Id against the body tenant).'}"
    if status == 404:
        return f"404 Not found: {detail or 'endpoint or resource does not exist on this server.'}"
    if status == 422:
        return f"422 Invalid request: {detail or 'the server rejected the input.'}"
    if status == 429:
        return "429 Rate limited by the upstream model. Wait a moment and retry."
    if status is not None and status >= 500:
        return f"{status} Server error: {detail or 'see the gateway log.'}"
    if status is not None:
        return f"HTTP {status}: {detail or 'unexpected response.'}"
    return "Unknown error."


def safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text[:300]}


# ---------------------------------------------------------------- streaming query
@dataclass
class StreamResult:
    path: str | None = None
    events: list[SSEEvent] = field(default_factory=list)
    answer_text: str = ""            # what the user sees: tokens, replaced by final.answer after a retract
    final: dict | None = None
    error: dict | None = None
    ttfb_ms: float | None = None     # time to first token event
    total_ms: float = 0.0
    status_code: int | None = None
    problem: str | None = None       # readable connection / HTTP problem, if any


def run_streaming_query(client: httpx.Client, base_url: str, body: dict, tenant: str,
                        user: str | None = None, on_event=None) -> StreamResult:
    """POST /api/v1/agent/query with Accept: text/event-stream and consume events as they arrive.

    on_event(SSEEvent, StreamResult) is called for each event so a page can render live.
    Never raises for network or HTTP problems: they are reported in result.problem.
    """
    result = StreamResult()
    headers = api_headers(tenant, user)
    headers["Accept"] = "text/event-stream"
    start = time.perf_counter()
    try:
        with client.stream("POST", f"{base_url}/api/v1/agent/query", json=body, headers=headers,
                           timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)) as resp:
            result.status_code = resp.status_code
            result.path = resp.headers.get("X-Path")
            if resp.status_code != 200:
                resp.read()
                result.problem = error_message(resp.status_code, safe_json(resp))
                result.total_ms = (time.perf_counter() - start) * 1000
                return result
            parser = SSEParser()
            for chunk in resp.iter_bytes():
                now = (time.perf_counter() - start) * 1000
                for ev in parser.feed(chunk, now):
                    _apply_event(result, ev)
                    if on_event:
                        on_event(ev, result)
                if result.final is not None or result.error is not None:
                    break
    except (httpx.HTTPError, OSError) as exc:
        result.problem = error_message(None, exc=exc)
    result.total_ms = (time.perf_counter() - start) * 1000
    return result


def _apply_event(result: StreamResult, ev: SSEEvent) -> None:
    result.events.append(ev)
    if ev.event == "token":
        if result.ttfb_ms is None:
            result.ttfb_ms = ev.arrived_ms
        result.answer_text += str(ev.data.get("t", "")) if isinstance(ev.data, dict) else ""
    elif ev.event == "retract":
        result.answer_text = ""          # discard streamed generated text; final.answer follows
    elif ev.event == "final":
        result.final = ev.data if isinstance(ev.data, dict) else {"raw": ev.data}
        result.answer_text = str(result.final.get("answer", result.answer_text))
    elif ev.event == "error":
        result.error = ev.data if isinstance(ev.data, dict) else {"message": str(ev.data)}


# ---------------------------------------------------------------- stats
def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0..100). None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(-(-pct * len(ordered) // 100)))   # ceil(pct/100 * n)
    return ordered[min(rank, len(ordered)) - 1]
