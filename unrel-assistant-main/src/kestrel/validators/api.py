"""Validator API (design.md §2.7 "API (added to §1.2.12)").

POST   /api/v1/validators/runs             {"profile": "quick"|"full"} -> 202 {"run_id"}
GET    /api/v1/validators/runs/{run_id}    -> ValidatorRun
DELETE /api/v1/validators/runs/{run_id}    cancel
GET    /api/v1/validators/runs?profile=full&limit=10  history

Runs execute in a worker thread with their own event loop, never inside the request. Limits: one active quick run per
tenant and two globally; further quick runs wait as QUEUED. A quick run for the same tenant that finished less than
60 s ago is returned instead of starting a new one. Full runs share one slot. Roles are caller-declared (D-22).
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..contracts import ErrorCode, PlatformError
from ..gateway.app import _identity
from .base import Env
from .fixtures import make_env
from .models import ValidatorResult, ValidatorRun
from .registry import all_validators
from .runner import run_profile

GLOBAL_QUICK_LIMIT = 2
TENANT_QUICK_LIMIT = 1
REUSE_WINDOW_S = 60.0
MAX_HISTORY = 100


@dataclass
class _Run:
    run_id: uuid.UUID
    profile: str
    tenant: str
    status: str = "QUEUED"
    results: list[ValidatorResult] = field(default_factory=list)
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def view(self) -> ValidatorRun:
        return ValidatorRun(run_id=self.run_id, profile=self.profile, status=self.status,  # type: ignore[arg-type]
                            tenant_id=self.tenant, results=list(self.results))


class RunManager:
    def __init__(self, make_env_fn, validators_fn, budget_s: float | None = None):
        self._make_env, self._validators, self._budget = make_env_fn, validators_fn, budget_s
        self._runs: dict[uuid.UUID, _Run] = {}
        self._queue: list[_Run] = []
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- lifecycle
    def submit(self, profile: str, tenant: str) -> _Run:
        with self._lock:
            if profile == "quick":
                recent = [r for r in self._runs.values() if r.profile == "quick" and r.tenant == tenant
                          and r.status == "DONE" and r.finished_at is not None
                          and time.time() - r.finished_at < REUSE_WINDOW_S]
                if recent:
                    return max(recent, key=lambda r: r.finished_at or 0.0)
            run = _Run(uuid.uuid4(), profile, tenant)
            self._runs[run.run_id] = run
            self._queue.append(run)
            self._pump()
            return run

    def get(self, run_id: uuid.UUID, tenant: str) -> _Run | None:
        r = self._runs.get(run_id)
        return r if r and r.tenant == tenant else None

    def history(self, tenant: str, profile: str | None, limit: int) -> list[_Run]:
        rs = [r for r in self._runs.values() if r.tenant == tenant and (profile is None or r.profile == profile)]
        return rs[-limit:][::-1]                                 # newest first (dict keeps insertion order)

    def cancel(self, run: _Run) -> None:
        with self._lock:
            if run.status == "QUEUED":
                self._queue.remove(run)
                run.status = "CANCELLED"
            elif run.status == "RUNNING":
                run.cancel.set()                    # the worker stops at the next poll and marks CANCELLED

    # ---------------------------------------------------------------- scheduling (caller holds the lock)
    def _active(self, profile: str | None = None, tenant: str | None = None) -> list[_Run]:
        return [r for r in self._runs.values() if r.status == "RUNNING"
                and (profile is None or r.profile == profile) and (tenant is None or r.tenant == tenant)]

    def _pump(self) -> None:
        for run in list(self._queue):
            if run.profile == "quick":
                ok = (len(self._active("quick")) < GLOBAL_QUICK_LIMIT
                      and not self._active("quick", run.tenant))
            else:
                ok = not self._active("full")
            if ok:
                self._queue.remove(run)
                run.status = "RUNNING"
                threading.Thread(target=self._work, args=(run,), daemon=True).start()

    def _work(self, run: _Run) -> None:
        results: list[ValidatorResult] = run.results
        try:
            env: Env = self._make_env()
            asyncio.run(run_profile(self._validators(), env, run.profile, results,
                                    **({"budget_s": self._budget} if self._budget else {}),
                                    cancelled=run.cancel.is_set))
        finally:
            with self._lock:
                run.status = "CANCELLED" if run.cancel.is_set() else "DONE"
                run.finished_at = time.time()
                self._pump()


def mount_validators(app: FastAPI, orchestrator: Any, *, make_env_fn=None, validators_fn=None,
                     budget_s: float | None = None) -> RunManager:
    """Mount the validator routes on an existing app. The orchestrator is used only for its settings."""
    settings = orchestrator.s
    manager = RunManager(make_env_fn or (lambda: make_env(settings)), validators_fn or all_validators, budget_s)
    app.state.validators = manager

    def _view_json(run: _Run) -> dict:
        return run.view().model_dump(mode="json")

    @app.post("/api/v1/validators/runs", status_code=202)
    async def start_run(request: Request):
        ident = _identity(request)
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise PlatformError(ErrorCode.PARSE_ERROR, "body must be JSON") from e
        profile = (body or {}).get("profile") if isinstance(body, dict) else None
        if profile not in ("quick", "full"):
            raise PlatformError(ErrorCode.INVALID_PARAMS, "profile must be 'quick' or 'full'")
        run = manager.submit(profile, ident["tenant_id"])
        return JSONResponse({"run_id": str(run.run_id), "status": run.status}, status_code=202)

    @app.get("/api/v1/validators/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        ident = _identity(request)
        run = _find(run_id, ident["tenant_id"])
        return _view_json(run)

    @app.delete("/api/v1/validators/runs/{run_id}")
    async def cancel_run(run_id: str, request: Request):
        ident = _identity(request)
        run = _find(run_id, ident["tenant_id"])
        manager.cancel(run)
        return _view_json(run)

    @app.get("/api/v1/validators/runs")
    async def list_runs(request: Request, profile: str | None = None, limit: int = 10):
        ident = _identity(request)
        if profile not in (None, "quick", "full"):
            raise PlatformError(ErrorCode.INVALID_PARAMS, "profile must be 'quick' or 'full'")
        if not 1 <= limit <= MAX_HISTORY:
            raise PlatformError(ErrorCode.INVALID_PARAMS, f"limit must be 1..{MAX_HISTORY}")
        return {"runs": [_view_json(r) for r in manager.history(ident["tenant_id"], profile, limit)]}

    def _find(run_id: str, tenant: str) -> _Run:
        try:
            rid = uuid.UUID(run_id)
        except ValueError as e:
            raise PlatformError(ErrorCode.INVALID_PARAMS, "run_id must be a UUID") from e
        run = manager.get(rid, tenant)
        if run is None:
            raise PlatformError(ErrorCode.INVALID_PARAMS, "unknown run")
        return run

    return manager
