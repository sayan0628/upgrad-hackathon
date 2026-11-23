import pytest

from kestrel.config import Settings
from kestrel.embed import HashEmbedder
from kestrel.gateway.app import build_orchestrator
from kestrel.llm.client import LLMUsage, StreamEvent
from kestrel.retrieval import IndexRegistry, LexicalReranker, ingest_texts
from kestrel.sample_data import DEMO_DOCS
from kestrel.stores import InMemoryStateStore

SETTINGS = Settings(openrouter_api_key="sk-or-v1-FAKEKEYFORTESTS000000", openrouter_base_url="https://or.test/api/v1",
                    generation_model="test/model", fallback_model=None, judge_model=None,
                    generation_extra_body={"reasoning": {"enabled": False}})


class ScriptedLLM:
    """Streams a fixed answer; records every call."""

    def __init__(self, parts=("Supplier certificates ", "are retained for seven years [1].")):
        self.parts, self.calls = parts, []

    async def stream(self, messages, *, model, max_tokens, temperature=0.0, extra_body=None):
        self.calls.append({"messages": messages, "model": model, "extra_body": extra_body})
        for i, p in enumerate(self.parts):
            yield StreamEvent("token", p, 5.0 * (i + 1))
        yield StreamEvent("done", usage=LLMUsage(10, 8))


async def make_orch(llm=None, triage=None, tenants=("tenant_a",), settings=SETTINGS, docs=DEMO_DOCS, embedder=None, entities=None):
    """Orchestrator over the demo corpus with deterministic fakes (HashEmbedder + LexicalReranker)."""
    emb, reg, state = embedder or HashEmbedder(), IndexRegistry(), InMemoryStateStore()
    for t in tenants:
        await ingest_texts(t, docs, emb, reg, state, words=40, overlap=5)
    for t in tenants:
        if entities:
            state.seed_entities(t, entities)
    return build_orchestrator(settings, llm, embedder=emb, registry=reg, triage=triage, state=state)


@pytest.fixture
def llm():
    return ScriptedLLM()
