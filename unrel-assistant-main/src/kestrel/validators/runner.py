"""Runs a profile: validators in parallel, a wall-clock budget per validator, seeded faults, and the verdict rules.

Verdict rules (design.md §2.7):
  * fewer than 2 seeded faults, or any seeded fault not caught -> UNTRUSTED (whatever the healthy result says)
  * nothing could be checked (every part skipped)               -> SKIPPED
  * a failed check or metric on the healthy build               -> FAIL
  * quick profile: a metric whose 95% CI crosses its threshold  -> WARN (small n is indicative only)
  * otherwise                                                   -> PASS
"""
from __future__ import annotations

import asyncio
import logging
import time

from typing import Callable

from .base import Env, Outcome, Validator, tenant_for
from .models import Finding, ValidatorResult

log = logging.getLogger("kestrel.validators")
MIN_SEEDED_FAULTS = 2
QUICK_BUDGET_S = 80.0          # the whole quick run must finish within 90 s; leave headroom for start-up


def _ci_crosses(metric) -> bool:
    if metric.ci95 is None:
        return False
    lo, hi = metric.ci95
    return lo <= metric.threshold <= hi


async def run_validator(v: Validator, env: Env, profile: str) -> ValidatorResult:
    t0 = time.perf_counter()
    findings: list[Finding] = []
    try:
        healthy: Outcome = await v.run(env, profile)
    except Exception as e:  # noqa: BLE001 - a crashing validator must not take the run down
        log.exception("validator %s crashed on the healthy build", v.id)
        healthy = Outcome(checks_passed=0, checks_total=1)
        findings.append(Finding(summary=f"validator crashed: {type(e).__name__}", severity="critical"))

    caught = 0
    faults_total = len(v.faults)
    for i, fault in enumerate(v.faults):
        try:
            out = await v.check(env, tenant_for(v.id, f"f{i}"), fault)
            if not out.clean:
                caught += 1
        except Exception as e:  # noqa: BLE001 - no outcome under a fault means the fault was not shown to be caught
            log.warning("validator %s: seeded fault %s raised %s (not counted as caught)", v.id, fault.name, type(e).__name__)
            findings.append(Finding(summary=f"seeded fault {fault.name} raised {type(e).__name__}", severity="major"))
    if faults_total < MIN_SEEDED_FAULTS:
        findings.append(Finding(summary=f"only {faults_total} seeded faults defined (need {MIN_SEEDED_FAULTS})",
                                severity="critical"))
    missed = faults_total - caught
    if missed or faults_total < MIN_SEEDED_FAULTS:
        verdict = "UNTRUSTED"
        findings.append(Finding(summary=f"{missed} of {faults_total} seeded faults not caught: "
                                        f"{', '.join(f.name for f in v.faults)}", severity="critical"))
    elif healthy.checks_total == 0:
        verdict = "SKIPPED"
    elif not healthy.clean:
        verdict = "FAIL"
    elif profile == "quick" and any(_ci_crosses(m) for m in healthy.metrics):
        verdict = "WARN"
    else:
        verdict = "PASS"
    findings.extend(healthy.findings)
    skip_reason = "; ".join(healthy.skipped + list(v.skip_reasons)) or None
    if verdict == "SKIPPED" and skip_reason is None:
        skip_reason = "no checks could run"
    return ValidatorResult(
        validator_id=v.id, verdict=verdict,
        checks_passed=healthy.checks_passed, checks_total=healthy.checks_total,
        metrics=healthy.metrics, seeded_faults_caught=caught, seeded_faults_total=faults_total,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        llm_tokens_in=healthy.tokens_in, llm_tokens_out=healthy.tokens_out,
        findings=findings[:20], skip_reason=skip_reason,
    )


async def run_profile(validators: list[Validator], env: Env, profile: str, results: list[ValidatorResult],
                      budget_s: float = QUICK_BUDGET_S, cancelled: Callable[[], bool] = lambda: False) -> None:
    """Appends each finished ValidatorResult to `results` as it completes (the API reads this while RUNNING).
    `cancelled()` is polled; when it turns true the running validators are cancelled and nothing else starts."""
    async def one(v: Validator) -> None:
        try:
            res = await asyncio.wait_for(run_validator(v, env, profile), timeout=budget_s)
        except asyncio.TimeoutError:
            res = ValidatorResult(validator_id=v.id, verdict="FAIL", checks_passed=0, checks_total=1,
                                  seeded_faults_caught=0, seeded_faults_total=len(v.faults), duration_ms=int(budget_s * 1000),
                                  findings=[Finding(summary=f"exceeded the {budget_s:.0f} s budget", severity="major")])
        results.append(res)

    tasks = [asyncio.ensure_future(one(v)) for v in validators]

    async def watch() -> None:
        while not all(t.done() for t in tasks):
            if cancelled():
                for t in tasks:
                    t.cancel()
                return
            await asyncio.sleep(0.2)

    watcher = asyncio.ensure_future(watch())
    await asyncio.gather(*tasks, return_exceptions=True)
    watcher.cancel()
    results.sort(key=lambda r: int(r.validator_id[1:]))
