"""Claim and policy state machines (design.md §1.2.9, corpus workflows/claim_policy_workflows.md).
Cross-checked against data/insurance/golden/workflow_tables.json by tests."""
from __future__ import annotations

import re

from .contracts import ErrorCode, PlatformError

CLAIM_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "SUBMITTED": ("UNDER_REVIEW", "WITHDRAWN"),
    "UNDER_REVIEW": ("INFO_REQUESTED", "APPROVED", "DENIED", "WITHDRAWN"),
    "INFO_REQUESTED": ("UNDER_REVIEW",),
    "APPROVED": ("PAID",),
    "DENIED": ("APPEALED",),
    "APPEALED": ("UNDER_REVIEW",),
    "PAID": (),
    "WITHDRAWN": (),
}
POLICY_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "QUOTED": ("ISSUED",),
    "ISSUED": ("IN_FORCE",),
    "IN_FORCE": ("GRACE", "CANCELLED"),
    "GRACE": ("IN_FORCE", "LAPSED"),
    "LAPSED": ("REINSTATEMENT_REVIEW",),
    "REINSTATEMENT_REVIEW": ("IN_FORCE", "LAPSED"),
    "CANCELLED": (),
}


def machine_for(entity_id: str) -> dict[str, tuple[str, ...]] | None:
    return CLAIM_TRANSITIONS if entity_id.startswith("CLM-") else POLICY_TRANSITIONS if entity_id.startswith("POL-") else None


def normalise_state(text: str) -> str:
    return re.sub(r"[^A-Za-z]+", "_", text.strip()).strip("_").upper()


def check_transition(entity_id: str, current: str | None, from_state: str, to_state: str) -> None:
    """Order matters (and matches the golden dataset): unknown entity/state -> INVALID_PARAMS;
    stale from_state -> STATE_CONFLICT; disallowed move -> INVALID_PARAMS."""
    machine = machine_for(entity_id)
    if machine is None:
        raise PlatformError(ErrorCode.INVALID_PARAMS, f"{entity_id} has no workflow", {"remediation": "Only claims (CLM-) and policies (POL-) have workflows."})
    if current is None:
        raise PlatformError(ErrorCode.INVALID_PARAMS, f"unknown entity {entity_id}")
    if to_state not in machine:
        raise PlatformError(ErrorCode.INVALID_PARAMS, f"unknown state {to_state}", {"valid_states": sorted(machine)})
    if from_state != current:
        raise PlatformError(ErrorCode.STATE_CONFLICT, f"{entity_id} is {current}, not {from_state}", {"current_state": current})
    if to_state not in machine[current]:
        raise PlatformError(ErrorCode.INVALID_PARAMS, f"{entity_id} cannot move from {current} to {to_state}",
                            {"allowed": list(machine[current]), "remediation": f"From {current} the allowed next states are: {', '.join(machine[current]) or 'none (final state)'}."})
