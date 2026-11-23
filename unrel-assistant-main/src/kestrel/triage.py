"""Triage (design.md FR-04). Rule-based placeholder behind the same interface as the distilled classifier that
replaces it (KPI-04 >= 97% must be measured on a labelled intent set; these rules are NOT evidence of that)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

Route = Literal["DIRECT_ANSWER", "HYBRID_RAG", "TRANSACTION_EXEC", "REJECT_OOS"]


@dataclass(frozen=True)
class TriageResult:
    route: Route
    complexity: Literal["low", "high"] = "high"
    risk_tier: int = 1
    reason: str | None = None     # e.g. "personal_account" for REJECT_OOS


class Triage(Protocol):
    def classify(self, text: str) -> TriageResult: ...


_TXN = re.compile(r"^\s*(please\s+|can you\s+|could you\s+)?(submit|file|transition|move|change|update|set|mark|approve|reject|"
                  r"create|register|release|hold|quarantine|recall)\b.*(\b(form|claim|policy|workflow|status|state|order|request|"
                  r"first notice of loss|address change|beneficiary change|cancellation)\b|(clm|pol|sku|lot)[-_ ]?\d)", re.I)
_OOS = re.compile(r"\b(write (me )?a (poem|song|story|joke)|weather|stock (tip|price)s?|lottery|horoscope|"
                  r"diagnos(e|is) (my|this)|legal advice|invest(ment)? advice)\b", re.I)
# Small-talk is recognised by VOCABULARY, not by a list of phrases: every word must come from this closed set. Any
# content word (grace, premium, claim, ...) makes the message a real question, so "Hi, what is the grace period?"
# is never mistaken for chat.
_CHAT_VOCAB = frozenset("hi hello hey hiya thanks thank thankyou you so much a lot very great good morning afternoon evening there "
                        "again please can could help me with i have question questions what do who are your is this and yes ok okay "
                        "cool awesome nice bye goodbye".split())
_CHAT_CORE = re.compile(r"\b(hi|hello|hey|hiya|thanks?|thank|good|great|bye|goodbye|help|do|who|cool|awesome|nice|ok|okay)\b", re.I)
# Questions about the caller's OWN account cannot be answered from the knowledge base. Matching generic policy text to
# them ("What is my deductible this year?") would be a confident-sounding wrong answer, so they are refused explicitly.
_PERSONAL = re.compile(r"(?i)\bmy (current|next|latest|last)\b|\b(this year|so far|to date)\b.*\bmy\b|\bmy\b.*\b(this year|so far|to date)\b|"
                       r"\b(have|has) (i|my \w+) (paid|been paid|been approved)\b|\bhave i paid\b|\bmy (policy number|payment history|account|agent'?s?)\b|"
                       r"\b(status of|why was|has) my (claim|policy|payment)\b|\bis my (policy|claim|coverage|payment)\b|\bmy coverage start\b|"
                       r"\bam i on\b|\bwho is the beneficiary on my\b|\bhow much (cash value|coverage) do i have\b|\bwhat is my (policy|premium|deductible|balance)\b", re.I)
_FACTOID = re.compile(r"^\s*(what|which|when|who|where|how (long|many|much|often|soon)|is|are|does|do|can|must)\b", re.I)
_COMPLEX = re.compile(r"\b(compare|difference|summari[sz]e|explain|why|steps?|procedure|list all|and then|versus|vs\.?|"
                      r"pros|cons|walk me through)\b", re.I)


class RuleTriage:
    def classify(self, text: str) -> TriageResult:
        if _PERSONAL.search(text):
            return TriageResult("REJECT_OOS", reason="personal_account")
        if _OOS.search(text):
            return TriageResult("REJECT_OOS")
        if _TXN.search(text):
            return TriageResult("TRANSACTION_EXEC", "high", 3)
        words = re.findall(r"[a-z']+", text.lower())
        if words and len(words) <= 8 and all(w in _CHAT_VOCAB for w in words) and _CHAT_CORE.search(text):
            return TriageResult("DIRECT_ANSWER", "low")
        low = bool(_FACTOID.match(text)) and not _COMPLEX.search(text) and len(text.split()) <= 25
        return TriageResult("HYBRID_RAG", "low" if low else "high")


class FixedTriage:
    """Test helper: always the same decision."""

    def __init__(self, route: Route, complexity: Literal["low", "high"] = "high"):
        self._r = TriageResult(route, complexity)

    def classify(self, text: str) -> TriageResult:
        return self._r
