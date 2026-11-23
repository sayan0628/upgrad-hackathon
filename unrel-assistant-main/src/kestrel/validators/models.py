"""Result models for the feature validators (design.md §2.7). Strict pydantic v2, extra fields forbidden."""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field, StringConstraints

Verdict = Literal["PASS", "FAIL", "WARN", "SKIPPED", "UNTRUSTED"]
Profile = Literal["quick", "full"]
RunStatus = Literal["QUEUED", "RUNNING", "DONE", "CANCELLED"]


class Base(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class MetricResult(Base):
    name: str                                   # e.g. "KPI-28 cross-tenant leaks"
    value: float
    threshold: float
    comparator: Literal["<=", ">=", "=="]
    n: int
    ci95: tuple[float, float] | None = None

    @property
    def passed(self) -> bool:
        if self.comparator == "<=":
            return self.value <= self.threshold
        if self.comparator == ">=":
            return self.value >= self.threshold
        return self.value == self.threshold


class Finding(Base):
    summary: str
    trace_id: str | None = None
    severity: Literal["critical", "major", "minor"]


class ValidatorResult(Base):
    validator_id: Annotated[str, StringConstraints(pattern=r"^V([1-9]|10)$")]
    verdict: Verdict
    checks_passed: int
    checks_total: int
    metrics: list[MetricResult] = Field(default_factory=list)
    seeded_faults_caught: int
    seeded_faults_total: int
    duration_ms: int
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    est_cost_usd: float = 0.0
    findings: list[Finding] = Field(default_factory=list, max_length=20)
    skip_reason: str | None = None


class ValidatorRun(Base):
    run_id: UUID4
    profile: Profile
    status: RunStatus
    tenant_id: str
    results: list[ValidatorResult] = Field(default_factory=list)


def new_run_id() -> UUID:
    return uuid4()
