"""Form schemas for Tier 3 submissions (design.md §1.2.3). Pure validation + schema-driven PII redaction."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable

from .contracts import ErrorCode, PlatformError
from .plugins import PiiRedactor, PreContext


@dataclass(frozen=True)
class Field:
    name: str
    check: Callable[[str], str | None]      # returns an error message, or None when valid
    pii: bool = False                       # redacted wholesale in args_redacted / audit


def _pattern(rx: str, hint: str):
    r = re.compile(rx)
    return lambda v: None if r.match(v) else f"must look like {hint}"

def _enum(*vals: str):
    return lambda v: None if v in vals else f"must be one of: {', '.join(vals)}"

def _text(lo: int, hi: int):
    return lambda v: None if lo <= len(v.strip()) <= hi else f"must be {lo}-{hi} characters"

def _date(allow_future: bool):
    def chk(v: str):
        try:
            x = dt.date.fromisoformat(v)
        except ValueError:
            return "must be a date in YYYY-MM-DD format"
        return None if allow_future or x <= dt.date.today() else "must not be in the future"
    return chk

def _number(lo: Decimal, hi: Decimal):
    def chk(v: str):
        try:
            x = Decimal(v)
        except InvalidOperation:
            return "must be a number"
        return None if lo <= x <= hi else f"must be between {lo} and {hi}"
    return chk

POLICY = Field("policy_id", _pattern(r"^POL-\d{6}$", "POL-123456"))

FORMS: dict[str, dict[str, Field]] = {
    "fnol": {f.name: f for f in (POLICY, Field("loss_date", _date(False)),
             Field("loss_type", _enum("collision", "theft", "fire", "water_damage", "medical", "other")),
             Field("description", _text(1, 500)), Field("estimated_amount", _number(Decimal(0), Decimal(10_000_000))))},
    "address_change": {f.name: f for f in (POLICY, Field("new_address", _text(5, 200), pii=True), Field("effective_date", _date(True)))},
    "beneficiary_change": {f.name: f for f in (POLICY, Field("beneficiary_name", _text(2, 100), pii=True),
             Field("relationship", _enum("spouse", "child", "parent", "sibling", "other")),
             Field("share_percent", _number(Decimal(1), Decimal(100))))},
    "cancellation_request": {f.name: f for f in (POLICY, Field("reason", _text(1, 300)), Field("effective_date", _date(True)))},
}


def validate_form(form_id: str, payload: dict[str, str]) -> dict[str, str]:
    schema = FORMS.get(form_id)
    if schema is None:
        raise PlatformError(ErrorCode.INVALID_PARAMS, f"unknown form {form_id!r}", {"available_forms": sorted(FORMS)})
    missing = [n for n in schema if n not in payload or not str(payload[n]).strip()]
    unexpected = [n for n in payload if n not in schema]
    bad = [{"field": n, "reason": msg} for n, f in schema.items() if n in payload and str(payload[n]).strip()
           and (msg := f.check(str(payload[n]).strip()))]
    bad += [{"field": n, "reason": "unexpected field"} for n in unexpected]
    if missing or bad:
        raise PlatformError(ErrorCode.INVALID_PARAMS, "form validation failed",
                            {"form_id": form_id, "missing": missing, "bad": bad,
                             "remediation": f"The {form_id} form needs: {', '.join(schema)}."})
    return {n: str(payload[n]).strip() for n in schema}


def redact_form(form_id: str, payload: dict[str, str]) -> dict[str, str]:
    """PII-flagged fields are replaced wholesale; remaining values go through the regex redactor."""
    schema, red, out = FORMS[form_id], PiiRedactor(), {}
    for k, v in payload.items():
        if schema[k].pii:
            out[k] = f"[REDACTED:{k}]"
        else:
            c = PreContext("", v)
            red.run(c)
            out[k] = c.text
    return out
