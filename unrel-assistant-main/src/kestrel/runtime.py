"""Live runtime: the real app with real local models, the insurance corpus and (if configured) the OpenRouter client.

Used by the workbench, the validators and `uvicorn kestrel.runtime:app`. Tests keep using the deterministic fakes
in tests/conftest.py.

Tenants: `tenant_acme` is seeded with the insurance corpus and golden entities; `tenant_validator_*` are created
on demand by the validators (design.md §2.7 principle 4) and never touch `tenant_acme`.
"""
from __future__ import annotations

import json
import os
import pathlib
import warnings

from .config import Settings
from .embed import SentenceTransformerEmbedder
from .gateway.app import build_orchestrator, create_app
from .llm.client import OpenRouterClient
from .retrieval import IndexRegistry, OnnxCrossEncoderReranker, ingest_texts
from .stores import InMemoryStateStore
from .triage import RuleTriage

ROOT = pathlib.Path(__file__).resolve().parents[2]
INSURANCE_DOCS = ROOT / "data" / "insurance" / "docs"
GOLDEN = ROOT / "data" / "insurance" / "golden"
DEMO_TENANT = "tenant_acme"


def configure_torch() -> None:
    """Performance and warning hygiene for the CPU-only dev box (see design.md §1.4.2 measurements).

    * torch defaults to 2 intra-op threads here even though 4 logical CPUs are visible (2 physical cores with SMT).
      The explicit setting makes that a deliberate choice instead of an accident of the environment.
    * The INT8 path uses torch.ao.quantization, which torch deprecates in favour of torchao. We keep it working
      for now and silence only that DeprecationWarning, so real warnings still surface.
    """
    import torch

    threads = int(os.environ.get("KESTREL_TORCH_THREADS", "2"))
    torch.set_num_threads(threads)
    warnings.filterwarnings("ignore", message="torch.ao.quantization is deprecated", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="torch.quantize_per_tensor", category=UserWarning)


def load_corpus() -> dict[str, str]:
    return {str(p.relative_to(INSURANCE_DOCS)): p.read_text(encoding="utf-8")
            for p in sorted(INSURANCE_DOCS.rglob("*.md"))}


def load_entities() -> dict[str, str]:
    return json.loads((GOLDEN / "entities.json").read_text(encoding="utf-8"))


async def build_live_orchestrator(settings: Settings | None = None, *, embedder=None, reranker=None,
                                  llm=None, tenants: tuple[str, ...] = (DEMO_TENANT,)):
    """Real components. Pass fakes in for tests; the defaults are the production-shaped local stack."""
    settings = settings or Settings.from_env()
    configure_torch()
    embedder = embedder or SentenceTransformerEmbedder()
    reranker = reranker or OnnxCrossEncoderReranker()   # fp32 ONNX: same rankings as torch, faster (bench_results/reranker_onnx_bench.json)
    state, registry = InMemoryStateStore(), IndexRegistry()
    corpus = load_corpus()
    for t in tenants:
        await ingest_texts(t, corpus, embedder, registry, state, words=70, overlap=10)
        state.seed_entities(t, load_entities())
    if llm is None and settings.openrouter_api_key:
        llm = OpenRouterClient(settings)
    return build_orchestrator(settings, llm, embedder=embedder, registry=registry, triage=RuleTriage(),
                              reranker=reranker, state=state)


def create_live_app(settings: Settings | None = None):
    """Synchronous factory for uvicorn. Ingestion runs once at startup."""
    import asyncio
    settings = settings or Settings.from_env()
    orch = asyncio.run(build_live_orchestrator(settings))
    return create_app(settings, orchestrator=orch)


def __getattr__(name: str):  # lazy: `uvicorn kestrel.runtime:app`
    if name == "app":
        globals()["app"] = create_live_app()
        return globals()["app"]
    raise AttributeError(name)
