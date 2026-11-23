"""The ten validators in matrix order (design.md §2.7)."""
from __future__ import annotations

from .base import Validator
from .v1_contract import V1Contract
from .v2_isolation import V2Isolation
from .v3_guardrails import V3Guardrails
from .v4_routing import V4Routing
from .v5_loop import V5Loop
from .v6_hitl import V6HITL
from .v7_grounding import V7Grounding
from .v8_audit import V8Audit
from .v9_load import V9Load
from .v10_telemetry import V10Telemetry


def all_validators() -> list[Validator]:
    return [V1Contract(), V2Isolation(), V3Guardrails(), V4Routing(), V5Loop(), V6HITL(), V7Grounding(), V8Audit(),
            V9Load(), V10Telemetry()]
