"""Validator protocol and the seeded-fault harness (design.md §2.7 principles 2-4).

A validator's `check()` drives a sandbox app over HTTP and returns an `Outcome`. The runner calls it once on the
healthy build and once per seeded fault. A fault is a sandbox mutation applied through public constructor arguments
or public attributes (see `Sandbox.build`); it is caught when the check reports a failed check or metric.

Validators import no agent internals: the only agent-side names used are `build_orchestrator`/`create_app`
(public constructors) and the `Orchestrator` public attributes that faults touch.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ..config import Settings
from ..gateway.app import build_orchestrator, create_app
from ..pipeline import Orchestrator
from ..retrieval import IndexRegistry, ingest_texts
from ..stores import InMemoryStateStore
from .models import Finding, MetricResult

TENANT_PREFIX = "tenant_validator_"
# Pinned for validators: `build_orchestrator` kwargs that a fault may override (embedder, reranker, cache, triage,
# llm, state) and the orchestrator-level mutations (`post`).
BuildOverrides = dict[str, Any]


@dataclass(frozen=True)
class Fault:
    name: str
    description: str
    build: BuildOverrides = field(default_factory=dict)       # kwargs passed to build_orchestrator
    settings: dict = field(default_factory=dict)              # Settings fields to change (e.g. grounding_min)
    post: Callable[[Orchestrator], None] | None = None         # applied to the built orchestrator


@dataclass
class Outcome:
    checks_passed: int = 0
    checks_total: int = 0
    metrics: list[MetricResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    skipped: list[str] = field(default_factory=list)            # human-readable reasons for SKIPPED parts

    def check(self, ok: bool, summary: str | None = None, severity: str = "major", trace_id: str | None = None) -> bool:
        self.checks_total += 1
        if ok:
            self.checks_passed += 1
        elif summary:
            self.findings.append(Finding(summary=summary[:500], severity=severity, trace_id=trace_id))  # type: ignore[arg-type]
        return ok

    def metric(self, name: str, value: float, threshold: float, comparator: str, n: int,
               ci95: tuple[float, float] | None = None) -> MetricResult:
        m = MetricResult(name=name, value=float(value), threshold=float(threshold), comparator=comparator,  # type: ignore[arg-type]
                         n=n, ci95=ci95)
        self.metrics.append(m)
        self.check(m.passed, f"{name} = {value:.4g} violates {comparator} {threshold:.4g}")
        return m

    @property
    def clean(self) -> bool:
        return self.checks_passed == self.checks_total and all(m.passed for m in self.metrics)


class Sandbox:
    """One isolated app instance: fresh state, fresh index, the given fault applied."""

    def __init__(self, app, orch: Orchestrator, state: InMemoryStateStore):
        self.app, self.orch, self.state = app, orch, state
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://validator.local")

    async def aclose(self) -> None:
        await self.client.aclose()


class Env:
    """Shared, read-only inputs for validators: settings, corpus, golden data, and the component factories."""

    def __init__(self, settings: Settings, corpus: dict[str, str], entities: dict[str, str], *,
                 embedder_factory: Callable[[], Any], reranker_factory: Callable[[], Any],
                 llm_factory: Callable[[], Any], triage_factory: Callable[[], Any] | None = None):
        self.settings, self.corpus, self.entities = settings, corpus, entities
        self.embedder_factory, self.reranker_factory = embedder_factory, reranker_factory
        self.llm_factory, self.triage_factory = llm_factory, triage_factory

    async def sandbox(self, tenants: str | list[str], fault: Fault | None = None,
                      extra_docs: dict[str, dict[str, str]] | None = None, llm: Any = None,
                      state: InMemoryStateStore | None = None) -> Sandbox:
        """Fresh app with the shared corpus ingested for each tenant (plus `extra_docs[tenant]`, e.g. canaries)."""
        tenants = [tenants] if isinstance(tenants, str) else tenants
        for t in tenants:
            assert t.startswith(TENANT_PREFIX), "validators may only use tenant_validator_* tenants"
        overrides = dict(fault.build) if fault else {}
        settings = with_settings(self.settings, **fault.settings) if fault and fault.settings else self.settings
        embedder = overrides.pop("embedder", None) or self.embedder_factory()
        reranker = overrides.pop("reranker", None) or self.reranker_factory()
        llm = overrides.pop("llm", None) or llm or self.llm_factory()
        triage = overrides.pop("triage", None) or (self.triage_factory() if self.triage_factory else None)
        state = overrides.pop("state", None) or state or InMemoryStateStore()
        registry = IndexRegistry()
        for t in tenants:
            docs = {**self.corpus, **(extra_docs or {}).get(t, {})}
            await ingest_texts(t, docs, embedder, registry, state, words=70, overlap=10)
            state.seed_entities(t, self.entities)
        orch = build_orchestrator(settings, llm, embedder=embedder, registry=registry, triage=triage,
                                  reranker=reranker, state=state, **overrides)
        if fault and fault.post:
            fault.post(orch)
        app = create_app(settings, orchestrator=orch)
        return Sandbox(app, orch, state)


class Validator:
    """Subclasses set the class attributes and implement `check`."""

    id: str = ""
    title: str = ""
    features: str = ""
    kpis: str = ""
    faults: tuple[Fault, ...] = ()
    # Full-profile-only parts that this build cannot run (external infrastructure, judge quota). Reported as SKIPPED.
    skip_reasons: tuple[str, ...] = ()

    async def check(self, env: Env, tenant: str, fault: Fault | None = None) -> Outcome:  # pragma: no cover
        raise NotImplementedError

    async def run(self, env: Env, profile: str) -> Outcome:
        """Quick profile: small fixed samples. Full profile: subclasses may override for larger samples."""
        return await self.check(env, f"{TENANT_PREFIX}{self.id.lower()}_main")


def tenant_for(validator_id: str, idx: int | str = "main") -> str:
    return f"{TENANT_PREFIX}{validator_id.lower()}_{idx}"


def with_settings(settings: Settings, **changes: Any) -> Settings:
    return dataclasses.replace(settings, **changes)
