"""Feature validators (design.md §2.7): runner verdicts, seeded-fault detection, and the validator API lifecycle.

The healthy-build test runs the real validators (V9 is left out: its smoke takes ~45 s on its own). The API tests use
a fresh app with fast fake validators so the state machine is exercised without real runs.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel.contracts import PlatformError
from kestrel.gateway.app import build_orchestrator
from kestrel.config import Settings
from kestrel.validators.api import mount_validators
from kestrel.validators.base import Env, Fault, Outcome, Validator, tenant_for
from kestrel.validators.fixtures import make_env
from kestrel.validators.registry import all_validators
from kestrel.validators.runner import run_profile, run_validator

S = Settings(openrouter_api_key=None, openrouter_base_url="https://or.test/api/v1", generation_model="stub/model",
             fallback_model=None, judge_model=None)

REAL = [v for v in all_validators() if v.id != "V9"]
EXPECTED_HEALTHY = {"V1": "PASS", "V2": "PASS", "V3": "FAIL", "V4": "FAIL", "V5": "PASS", "V6": "PASS",
                    "V7": "PASS", "V8": "PASS", "V10": "PASS"}   # V3/V4 fail on real gaps, see the report


@pytest.fixture(scope="module")
def env() -> Env:
    return make_env(S)


# ------------------------------------------------------------------ runner and verdicts
async def test_runner_healthy_build_verdicts(env):
    results: list = []
    await run_profile(REAL, env, "quick", results)
    by_id = {r.validator_id: r for r in results}
    assert set(by_id) == set(EXPECTED_HEALTHY)
    for vid, verdict in EXPECTED_HEALTHY.items():
        assert by_id[vid].verdict == verdict, (vid, by_id[vid].findings)
    # the validators themselves are sound: every seeded fault is caught on every validator
    for r in results:
        assert r.seeded_faults_caught == r.seeded_faults_total >= 2, r.validator_id
    assert [f.summary for f in by_id["V3"].findings if "injection" in f.summary]
    assert [f.summary for f in by_id["V4"].findings if "KPI-04" in f.summary]


@pytest.mark.parametrize("v", REAL, ids=lambda v: v.id)
async def test_each_validator_catches_seeded_faults(v, env):
    assert len(v.faults) >= 2
    for i, fault in enumerate(v.faults):
        out = await v.check(env, tenant_for(v.id, f"t{i}"), fault)
        assert not out.clean, f"{v.id} missed seeded fault {fault.name}"


async def test_v9_smoke_plumbing_only(env):
    """V9's percentile and 5xx plumbing on the healthy build (the full smoke is run by the quick profile)."""
    from kestrel.validators.v9_load import V9Load
    import kestrel.validators.v9_load as m
    old = m.SMOKE_S
    m.SMOKE_S = 2.0
    try:
        out = await V9Load().check(env, tenant_for("V9", "smoke"))
    finally:
        m.SMOKE_S = old
    assert out.clean, [f.summary for f in out.findings]


class _Blind(Validator):
    """Deliberately blind: its check passes whatever the sandbox looks like, so it cannot tell the faults apart."""
    id = "V1"
    title = "blind"
    faults = (Fault("blind_a", "not detected"), Fault("blind_b", "not detected"))

    async def check(self, env, tenant, fault=None):
        out = Outcome()
        out.check(True)
        return out


class _HalfBlind(_Blind):
    """Catches one seeded fault and misses the other: still UNTRUSTED."""
    faults = (Fault("seen", "detected"), Fault("missed", "not detected"))

    async def check(self, env, tenant, fault=None):
        out = Outcome()
        out.check(not (fault and fault.name == "seen"), "seeded fault seen")
        return out


async def test_blind_validator_is_untrusted(env):
    res = await run_validator(_Blind(), env, "quick")
    assert res.verdict == "UNTRUSTED"
    assert res.seeded_faults_caught == 0 and res.seeded_faults_total == 2


async def test_partly_blind_validator_is_untrusted(env):
    res = await run_validator(_HalfBlind(), env, "quick")
    assert res.verdict == "UNTRUSTED"
    assert res.seeded_faults_caught == 1 and res.seeded_faults_total == 2


# ------------------------------------------------------------------ API lifecycle
class _Slow(Validator):
    """Fake validator for the API tests: sleeps, passes on the healthy build, fails under a fault."""
    id = "V1"
    title = "slow"
    faults = (Fault("f1", "x"), Fault("f2", "y"))
    delay = 0.6

    async def check(self, env, tenant, fault=None):
        await asyncio.sleep(self.delay)
        out = Outcome()
        out.check(fault is None, "fault present")
        return out


def _app(validators, delay=0.6):
    _Slow.delay = delay
    app = FastAPI()

    @app.exception_handler(PlatformError)
    async def _pe(_, e: PlatformError):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": e.to_dict()}, status_code=e.http_status)

    orch = build_orchestrator(S, None)
    mount_validators(app, orch, make_env_fn=lambda: None, validators_fn=lambda: validators, budget_s=10.0)
    return app


def _wait(client, run_id, tenant, want, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/v1/validators/runs/{run_id}", headers={"X-Tenant-Id": tenant}).json()
        if body["status"] in want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run did not reach {want}")


def _post(client, tenant, profile):
    return client.post("/api/v1/validators/runs", headers={"X-Tenant-Id": tenant},
                       json={"profile": profile})


def test_lifecycle_202_running_done():
    client = TestClient(_app([_Slow()]))
    r = _post(client, "tenant_validator_api_a", "quick")
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    first = client.get(f"/api/v1/validators/runs/{run_id}", headers={"X-Tenant-Id": "tenant_validator_api_a"}).json()
    assert first["status"] in ("QUEUED", "RUNNING")
    running = _wait(client, run_id, "tenant_validator_api_a", ("RUNNING",))
    assert running["status"] == "RUNNING"
    done = _wait(client, run_id, "tenant_validator_api_a", ("DONE",))
    assert done["results"][0]["validator_id"] == "V1"
    assert done["results"][0]["verdict"] == "PASS"            # both seeded faults are caught by the fake


def test_no_role_gating_on_runs():
    client = TestClient(_app([_Slow()], delay=0.01))
    t = "tenant_validator_api_noroles"
    assert _post(client, t, "bogus").status_code == 422
    assert client.post("/api/v1/validators/runs", json={"profile": "quick"}).status_code == 403   # no tenant header


def test_cancel_running_run():
    client = TestClient(_app([_Slow()], delay=30.0))
    t = "tenant_validator_api_cancel"
    run_id = _post(client, t, "quick").json()["run_id"]
    _wait(client, run_id, t, ("RUNNING",))
    d = client.delete(f"/api/v1/validators/runs/{run_id}", headers={"X-Tenant-Id": t})
    assert d.status_code == 200
    done = _wait(client, run_id, t, ("CANCELLED",), timeout=10.0)
    assert done["status"] == "CANCELLED"


def test_queue_limits_per_tenant_and_global():
    client = TestClient(_app([_Slow()], delay=30.0))
    a, b, c = "tenant_validator_q_a", "tenant_validator_q_b", "tenant_validator_q_c"
    ra = _post(client, a, "quick").json()["run_id"]
    _wait(client, ra, a, ("RUNNING",))
    ra2 = _post(client, a, "quick").json()["run_id"]              # same tenant: must queue
    assert client.get(f"/api/v1/validators/runs/{ra2}", headers={"X-Tenant-Id": a}).json()["status"] == "QUEUED"
    rb = _post(client, b, "quick").json()["run_id"]
    _wait(client, rb, b, ("RUNNING",))                            # second global slot
    rc = _post(client, c, "quick").json()["run_id"]               # global limit reached: must queue
    assert client.get(f"/api/v1/validators/runs/{rc}", headers={"X-Tenant-Id": c}).json()["status"] == "QUEUED"
    for rid, t in ((ra, a), (rb, b), (ra2, a), (rc, c)):
        client.delete(f"/api/v1/validators/runs/{rid}", headers={"X-Tenant-Id": t})


def test_recent_quick_run_is_reused_and_history_listed():
    client = TestClient(_app([_Slow()], delay=0.01))
    t = "tenant_validator_api_reuse"
    first = _post(client, t, "quick").json()["run_id"]
    _wait(client, first, t, ("DONE",))
    again = _post(client, t, "quick").json()["run_id"]
    assert again == first                                         # finished < 60 s ago: shown instead of a new run
    full = _post(client, t, "full").json()["run_id"]
    _wait(client, full, t, ("DONE",))
    hist = client.get("/api/v1/validators/runs", params={"profile": "full", "limit": 10}, headers={"X-Tenant-Id": t}).json()
    assert [r["run_id"] for r in hist["runs"]] == [full]
    assert client.get("/api/v1/validators/runs", params={"limit": 0}, headers={"X-Tenant-Id": t}).status_code == 422


def test_runs_are_tenant_scoped():
    client = TestClient(_app([_Slow()], delay=0.01))
    run_id = _post(client, "tenant_validator_api_own", "quick").json()["run_id"]
    r = client.get(f"/api/v1/validators/runs/{run_id}", headers={"X-Tenant-Id": "tenant_validator_api_other"})
    assert r.status_code == 422
    assert client.get("/api/v1/validators/runs/not-a-uuid", headers={"X-Tenant-Id": "tenant_validator_api_own"}).status_code == 422
