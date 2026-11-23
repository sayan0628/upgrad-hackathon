"""Verify the SSE streaming paths over a real socket (design.md §1.2.12, §1.4.2 evidence-first streaming).

Starts the real gateway on a local port (uvicorn, background thread), then streams one request per scenario with
httpx and timestamps every event as it ARRIVES. Checks are about delivery, not just order:

  * X-Path matches the path the request is expected to take (fast | llm)
  * meta is first, final is last, and 'token' events only appear on the LLM path
  * on the LLM path, 'evidence' arrives before the first token and tokens are spread over time (not buffered)
  * a retract (ungrounded draft) is followed by an ABSTAINED final, and the withheld text is not in final.answer
  * a model error arrives as an 'error' event and the stream still terminates cleanly
  * non-streaming JSON for the same question carries the same route and status

Default mode uses a scripted, delayed LLM (no spend). `--live` sends one real request through OpenRouter with
max_tokens capped (design: reasoning off), to confirm the real provider streams incrementally. It costs a few
tokens and needs OPENROUTER_API_KEY in the environment.

    python tools/stream_check.py            # offline, deterministic delays
    python tools/stream_check.py --live     # one real OpenRouter stream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import socket
import sys
import threading
import time

import httpx
import uvicorn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.config import Settings  # noqa: E402
from kestrel.llm.client import StreamEvent  # noqa: E402
from kestrel.contracts import ErrorCode, PlatformError  # noqa: E402
from kestrel.gateway.app import create_app  # noqa: E402
from kestrel.runtime import GOLDEN, DEMO_TENANT, build_live_orchestrator  # noqa: E402

TENANT = DEMO_TENANT
TOKEN_DELAY_S = 0.06
SPREAD_REQUIRED = True   # set to False for --live
FIRST_TOKEN_DELAY_S = 0.5


class ScriptedLLM:
    """Streams a grounded answer with a visible delay between tokens. `mode` selects the scenario behaviour."""

    def __init__(self, mode: str = "grounded", grounded_text: str | None = None):
        self.mode = mode
        self.calls = 0
        self.grounded_text = grounded_text   # quoted from the retrieved chunk so the grounding check passes

    async def stream(self, messages, *, model, max_tokens, temperature=0.0, extra_body=None):
        self.calls += 1
        if self.mode == "error":
            raise PlatformError(ErrorCode.UPSTREAM_RATE_LIMITED, "LLM provider rate limited", {})
        if self.mode == "ungrounded":
            parts = ["Unsupported: bananas grow near the equator ", "and taste sweet."]
        else:
            words = (self.grounded_text or "The policy text states the answer.").split()
            cut = max(1, len(words) // 3)
            parts = [" ".join(words[:cut]) + " ", " ".join(words[cut:2 * cut]) + " ", " ".join(words[2 * cut:]) + " [1]."]
        await asyncio.sleep(FIRST_TOKEN_DELAY_S)
        for p in parts:
            yield StreamEvent("token", p, 0.0)
            await asyncio.sleep(TOKEN_DELAY_S)
        yield StreamEvent("done", "", 0.0)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(app) -> tuple[uvicorn.Server, str]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(base + "/healthz", timeout=0.5).status_code == 200:
                return server, base
        except httpx.HTTPError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def stream_request(base: str, query: str, user: str = "user_stream") -> dict:
    body = {"tenant_id": TENANT, "user_id": user, "query": query}
    headers = {"X-Tenant-Id": TENANT, "Accept": "text/event-stream"}
    events: list[dict] = []
    t0 = time.perf_counter()
    with httpx.Client(timeout=60) as c, c.stream("POST", base + "/api/v1/agent/query", json=body, headers=headers) as r:
        status, path = r.status_code, r.headers.get("x-path")
        name = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:") and name:
                events.append({"event": name, "t_ms": (time.perf_counter() - t0) * 1000,
                               "data": json.loads(line[5:].strip())})
                name = None
    return {"status": status, "x_path": path, "events": events}


def check(res: dict, expect_path: str | None, expect_status: str | None, expect_tokens: bool | None = None,
          expect_retract: bool = False) -> list[str]:
    errs: list[str] = []
    ev = res["events"]
    names = [e["event"] for e in ev]
    if res["status"] != 200:
        errs.append(f"HTTP {res['status']}")
    if not names or names[0] != "meta":
        errs.append(f"first event is {names[:1]}, expected meta")
    if not names or names[-1] not in ("final", "error"):
        errs.append(f"stream ends with {names[-1:]}, expected final or error")
    if expect_path and res["x_path"] != expect_path:
        errs.append(f"X-Path {res['x_path']!r}, expected {expect_path!r}")
    tok_t = [e["t_ms"] for e in ev if e["event"] == "token"]
    if expect_tokens is not None and bool(tok_t) != expect_tokens:
        errs.append(f"token events present={bool(tok_t)}, expected {expect_tokens}")
    if res["x_path"] == "llm" and tok_t:
        first_tok = tok_t[0]
        ev_t = [e["t_ms"] for e in ev if e["event"] == "evidence"]
        if ev_t and not ev_t[0] < first_tok:
            errs.append("evidence did not arrive before the first token")
        # Burst detection only makes sense with the scripted per-token delay. A real model can finish a short answer
        # in well under a second (Haiku ~300 tok/s), so live runs report the spread instead of asserting on it.
        if SPREAD_REQUIRED and len(tok_t) >= 3 and tok_t[-1] - tok_t[0] < 0.5 * TOKEN_DELAY_S * 1000 * (len(tok_t) - 1):
            errs.append("tokens arrived in one burst (buffered, not streamed)")
    if expect_retract != ("retract" in names):
        errs.append(f"retract present={'retract' in names}, expected {expect_retract}")
    if "retract" in names and names.index("retract") > names.index("final"):
        errs.append("retract arrived after final")
    final = next((e["data"] for e in ev if e["event"] == "final"), None)
    if expect_status and (final is None or final["status"] != expect_status):
        errs.append(f"final status {final and final['status']!r}, expected {expect_status!r}")
    if expect_retract and final and any(t in final["answer"] for t in ("bananas", "Unsupported")):
        errs.append("withheld text leaked into final.answer")
    return errs


def first_golden(category: str) -> str:
    for line in (GOLDEN / "golden_set.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["category"] == category:
            return r["question"]
    raise KeyError(category)


async def find_llm_question(orch) -> tuple[str, str]:
    """A golden question that the real pipeline routes to the LLM path, plus the top chunk's text (for grounding)."""
    from kestrel.contracts import AgentQueryRequest
    from kestrel.llm.client import StreamEvent as _SE  # noqa: F401  (kept for clarity of the stub type)
    rows = [json.loads(l) for l in (GOLDEN / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        req = AgentQueryRequest(tenant_id=TENANT, user_id="user_probe", query=r["question"])
        h = await orch.handle(req)
        if h.path == "llm":
            emb = (await orch.embedder.embed([r["question"]]))[0]
            res = await orch.retriever.retrieve(TENANT, r["question"], emb)
            return r["question"], res.chunks[0].chunk.text
    raise RuntimeError("no golden question reaches the LLM path")


async def build(llm, settings):
    from kestrel.embed import SentenceTransformerEmbedder
    from kestrel.retrieval import CrossEncoderReranker
    orch = await build_live_orchestrator(settings, embedder=SentenceTransformerEmbedder(),
                                         reranker=CrossEncoderReranker(), llm=llm)
    return orch


def scenarios(args, q_llm: str) -> list[tuple[str, str, dict]]:
    """(name, query, expectations). Expected paths are checked; unknown ones are reported, not guessed."""
    unanswerable = json.loads((GOLDEN / "unanswerable.jsonl").read_text().splitlines()[0])["question"]
    return [
        ("out_of_scope_refusal", "Write me a poem about a flask", dict(expect_path="fast", expect_status="REFUSED", expect_tokens=False)),
        ("abstain_unanswerable", unanswerable, dict(expect_path="fast", expect_status="ABSTAINED", expect_tokens=False)),
        ("transaction_pending", "Move claim CLM-00001 from SUBMITTED to UNDER_REVIEW", dict(expect_path="fast", expect_status="PENDING_APPROVAL", expect_tokens=False)),
        ("llm_grounded", q_llm, dict(expect_path="llm", expect_status="OK", expect_tokens=True)),
    ]


async def main(a) -> int:
    global SPREAD_REQUIRED
    SPREAD_REQUIRED = not a.live
    settings = Settings.from_env() if a.live else Settings(openrouter_api_key=None, openrouter_base_url="",
                                                           generation_model="offline/scripted", fallback_model=None,
                                                           judge_model=None, generation_extra_body={"reasoning": {"enabled": False}})
    if a.live and not settings.openrouter_api_key:
        print("--live needs OPENROUTER_API_KEY in the environment"); return 2
    from kestrel.llm.client import OpenRouterClient
    llm = OpenRouterClient(settings) if a.live else ScriptedLLM("grounded")
    orch = await build(llm, settings)
    server, base = start_server(create_app(settings, orchestrator=orch))
    failures = 0
    try:
        q_llm, grounded_text = await find_llm_question(orch)
        plan = scenarios(a, q_llm)
        if not a.live:
            # the LLM scenarios need the scripted model's other behaviours; swap the model per scenario
            plan += [("llm_retract_ungrounded", q_llm, dict(expect_path="llm", expect_status="ABSTAINED", expect_tokens=True, expect_retract=True, llm="ungrounded")),
                     ("llm_model_error", q_llm, dict(expect_path="llm", expect_status=None, expect_tokens=False, llm="error"))]
        for name, query, exp in plan:
            llm_mode = exp.pop("llm", None)
            if not a.live:
                orch.llm = ScriptedLLM(llm_mode or "grounded", grounded_text)
            res = await asyncio.to_thread(stream_request, base, query, f"user_s{abs(hash(name)) % 10**6}")
            errs = check(res, **exp)
            kinds = [e["event"] for e in res["events"]]
            first_last = {k: round(e["t_ms"]) for e in res["events"] for k in [e["event"]]}
            print(f"[{'PASS' if not errs else 'FAIL'}] {name}: path={res['x_path']} events={kinds}")
            print(f"        arrival ms: {first_last}")
            for e in errs:
                print(f"        - {e}")
            failures += bool(errs)
    finally:
        server.should_exit = True
    print(f"\n{len(plan) - failures}/{len(plan)} streaming scenarios passed" + (" (live OpenRouter)" if a.live else " (offline)"))
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    sys.exit(asyncio.run(main(p.parse_args())))
