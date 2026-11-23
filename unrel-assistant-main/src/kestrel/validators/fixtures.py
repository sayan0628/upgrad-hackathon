"""Deterministic inputs for validators: the stub generation model, the fake embedder/reranker factories, and the
golden data loader. Quick runs never reach a real model API; the stub below is the only `generation model`.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from ..config import Settings
from ..embed import HashEmbedder
from ..llm.client import LLMUsage, StreamEvent
from ..retrieval import LexicalReranker
from ..runtime import GOLDEN, load_corpus, load_entities
from .base import Env

_CTX_LINE = re.compile(r"^\[(\d+)\] (.*)$", re.M)
_SENT = re.compile(r"(?<=[.!?])\s+")


class StubLLM:
    """Deterministic stand-in for the generation model (design.md §2.7 quick profile: `llm-sim`).

    With evidence in the prompt it answers with the first sentence of the top context chunk and cites it [1], so a
    healthy answer is grounded by construction. Without evidence it returns a fixed refusal-style line.
    Its token usage is reported as the word counts, so token budgets can be checked without a tokenizer.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages: list[dict[str, Any]], *, model: str, max_tokens: int,
                     temperature: float = 0.0, extra_body: dict[str, Any] | None = None) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        user = messages[-1]["content"]
        m = _CTX_LINE.search(user)
        if m:
            sentence = _SENT.split(m.group(2).strip())[0]
            answer = f"{sentence} [1]"
        else:
            answer = "I can only answer from the knowledge base."
        words = answer.split()
        prompt_words = sum(len(x["content"].split()) for x in messages)
        for i, w in enumerate(words):
            yield StreamEvent("token", (" " if i else "") + w, float(i + 1))
        yield StreamEvent("done", usage=LLMUsage(prompt_words, len(words)))


def sse_events(text: str) -> dict[str, Any]:
    """Return the first `meta` and the `final` payload of an SSE body."""
    out: dict[str, Any] = {}
    event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event in ("meta", "final") and event not in out:
            out[event] = json.loads(line[6:])
    return out


def golden_set() -> list[dict]:
    return [json.loads(line) for line in (GOLDEN / "golden_set.jsonl").read_text(encoding="utf-8").splitlines() if line]


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (GOLDEN / name).read_text(encoding="utf-8").splitlines() if line]


def make_env(settings: Settings | None = None) -> Env:
    """Quick-profile environment: HashEmbedder + LexicalReranker + StubLLM + RuleTriage (the orchestrator default)."""
    settings = settings or Settings.from_env()
    return Env(settings, load_corpus(), load_entities(),
               embedder_factory=HashEmbedder, reranker_factory=LexicalReranker, llm_factory=StubLLM)
