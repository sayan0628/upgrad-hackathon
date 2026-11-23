"""Plugin contract and the PRE_EXECUTION pipeline.

NOTE: design.md specifies a Pluggy entry-point registry. This is a small in-process registry with the same
semantics (priority ordering, fail_closed, hot enable/disable); a pluggy adapter can wrap it later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import ErrorCode, PlatformError


@dataclass
class PreContext:
    tenant_id: str
    text: str                                   # evolves: raw -> redacted
    flags: dict[str, object] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


class PrePlugin(Protocol):
    name: str
    priority: int
    fail_closed: bool
    error_code: ErrorCode

    def run(self, ctx: PreContext) -> None: ...


class PrePipeline:
    def __init__(self, plugins: list[PrePlugin]):
        self._plugins = sorted(plugins, key=lambda p: p.priority)
        self._disabled: set[str] = set()

    def toggle(self, name: str, enabled: bool) -> None:
        (self._disabled.discard if enabled else self._disabled.add)(name)

    def run(self, ctx: PreContext) -> PreContext:
        t_all = time.perf_counter()
        for p in self._plugins:
            if p.name in self._disabled:
                continue
            t0 = time.perf_counter()
            try:
                p.run(ctx)
            except PlatformError:
                raise                       # a deliberate block
            except Exception as e:          # noqa: BLE001
                if p.fail_closed:           # plugin crashed: reject the request, never pass through
                    raise PlatformError(p.error_code, f"{p.name} failed closed: {type(e).__name__}") from e
            finally:
                ctx.timings_ms[p.name] = (time.perf_counter() - t0) * 1000
        ctx.timings_ms["hook.pre"] = (time.perf_counter() - t_all) * 1000
        return ctx
