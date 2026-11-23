"""Injection shield (priority 20, fail-closed). Deterministic rule layer only (ADR-006).

The text is scanned in several views: the normalised text, its ROT13 form, and any base64 or hex run decoded.
Rules cover English plus a few high-frequency languages. Block rates (KPI-03 >= 99.5%) and benign false-positive
rates (KPI-06 <= 1%) are measured by tools/eval/eval_injection.py on a generated set, not assumed from this code.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re

from ..contracts import ErrorCode, PlatformError
from .base import PreContext
from .pii import normalise

_OVERRIDE_VERB = r"(?:ignore|disregard|forget|override|bypass|skip|drop|discard|abandon|set aside|no longer follow|neglect)"
_QUALIFIER = r"(?:previous|prior|above|earlier|all|system|your|the|every|everything|original|these|those|any)"
_TARGET = (r"(?:instructions?|prompts?|rules?|messages?|context|guidelines?|directives?|polic(?:y|ies)|guardrails?|"
           r"safety|(?:content|safety) filters?|restrictions?|everything you (?:were|have been) (?:told|given))")

_RULE_SRC: tuple[tuple[str, str], ...] = (
    ("override_instructions", rf"\b{_OVERRIDE_VERB}\b.{{0,40}}\b{_QUALIFIER}\b.{{0,40}}\b{_TARGET}"),
    ("override_instructions_short", rf"\b{_OVERRIDE_VERB}\b\s+(?:{_QUALIFIER}\s+)?{_TARGET}\b"),
    ("reveal_prompt", r"\b(reveal|show|print|repeat|output|leak|dump|tell me)\b.{0,40}\b(system prompt|hidden prompt|your instructions|initial prompt|developer message|secret|internal)\b.{0,30}\b(prompt|instructions?|message|tool list|tools|config)"),
    ("role_hijack", r"\b(you are now|act as|pretend (to be|you are)|from now on you|you are)\b.{0,60}\b(dan|jailbroken|unrestricted|without (?:any )?(?:content |safety )?(?:rules|restrictions|filters?|limits|guardrails)|no (?:rules|restrictions|filters?)|developer mode)"),
    ("jailbreak_terms", r"\b(jailbreak|do anything now|dan mode|developer mode enabled)\b"),
    ("fake_system_tag", r"(<\s*/?\s*(?:system|assistant)\s*>|<\|\s*(?:im_start|im_end|system|endoftext)\s*\|>|\[\s*/?\s*(?:system|inst)\s*\]|###\s*system)"),
    ("spoofed_role_label", r"(?:^|[\n.!?\]]\s*)(?:system|developer|admin)\s*(?:note|message|prompt|override)?\s*:\s*.{0,120}\b(?:override|ignore|disregard|disable|bypass|reveal|without|no longer|skip)\b"),
    ("safety_bypass", r"\b(disable|turn off|bypass|skip|remove|lift|circumvent|ignore)\b.{0,30}\b(safety|content filters?|guardrails?|moderation|safeguards?)\b|\bwithout (?:any )?(?:citations|sources|safety (?:policy|checks))\b"),
    ("exfil", r"\b(send|post|exfiltrate|forward|upload|dump)\b.{0,60}\b(to|at|into)\b.{0,40}(https?://|\bwebhook\b|\bftp://)"),
    ("tool_override", r"\b(call|invoke|run|execute)\b.{0,30}\b(submit_form|transition_workflow|approve_action)\b"),
    ("sqli_probe", r"(;\s*(drop|delete|truncate|update)\s+\w+|\bunion\s+select\b|'\s*or\s*'?1'?\s*=\s*'?1)"),
    # multi-language: override verb followed by an instruction noun, in order
    ("ml_override", r"(?:ignora[rd]?|ignorez|ignore les|ignoriere|oubliez?|oublie|vergiss|olvida[rs]?|desatiende|игнор\w*|忽略|无视|忘记).{0,60}?"
                    r"(?:instrucc|instruction|anweisung|instruç|инструк|指令|指示|提示|规则|規則|regles?|règles?|regeln|reglas|directiv|guideline|prompt)"),
    ("ml_override_jp", r"(?:指示|指令|プロンプト|ルール|ガイドライン).{0,30}?無視"),
    ("ml_reveal", r"(?:muestra|affiche|zeige|покажи|показ|显示|表示|revela|mostra)\w*.{0,40}?"
                  r"(?:prompt|提示|プロンプト|systemprompt|системн|systemprompt|sistema)"),
)

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (n, re.compile(p, re.I | re.S)) for n, p in _RULE_SRC)

_ENCODING_CUE = re.compile(r"rot-?13|base-?64|b64|\bhex\b|decode|decrypt|cipher|encoded|encrypted|unscramble", re.I)
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")
_HEX = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){8,}(?![0-9A-Fa-f])")


def _decoded_views(text: str) -> list[tuple[str, str]]:
    """Views the rules are also run on: ROT13 of the whole text, and base64 or hex runs that decode to text."""
    views = []
    # ROT13 is scanned only when the text names an encoding: a full second rule pass on every prompt costs
    # latency, and an unlabelled ROT13 payload is not an instruction the model is asked to decode.
    if _ENCODING_CUE.search(text):
        views.append(("rot13", codecs.decode(text, "rot13")))
    for m in _B64.finditer(text):
        try:
            raw = base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4), validate=True)
            views.append(("base64", raw.decode("utf-8")))
        except (binascii.Error, ValueError):
            pass
    for m in _HEX.finditer(text):
        try:
            views.append(("hex", bytes.fromhex(m.group(0)).decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            pass
    return views


def scan(text: str) -> list[str]:
    """Return the names of the rules that fire on the normalised text or on any decoded view."""
    hits = [n for n, rx in _RULES if rx.search(text)]
    for kind, view in _decoded_views(text):
        hits += [f"{kind}:{n}" for n, rx in _RULES if rx.search(view)]
    return hits


class InjectionShield:
    name = "injection_shield"
    priority = 20
    fail_closed = True
    error_code = ErrorCode.INJECTION_BLOCKED

    def run(self, ctx: PreContext) -> None:
        hits = scan(normalise(ctx.text))
        ctx.flags["injection_hits"] = hits
        if hits:
            raise PlatformError(ErrorCode.INJECTION_BLOCKED, "request blocked by injection shield", {"rules": hits})
