"""PII redaction (priority 10, fail-closed). Regex + obfuscation normaliser stand-in for Presidio.

Presidio is the design's engine; this deterministic layer covers the obfuscated forms
(`name[at]domain[dot]com`, homoglyphs, spaced digits) and runs even where Presidio is unavailable.
Recall (KPI-02 >= 99.8%) must be measured on the validator set (§1.5), not assumed from this code.
"""
from __future__ import annotations

import re
import unicodedata

from ..contracts import ErrorCode
from .base import PreContext

# Common confusables (Cyrillic/Greek lookalikes) folded to ASCII after NFKC.
_CONFUSABLES = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
                              "і": "i", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ο": "o", "α": "a", "ε": "e",
                              # uppercase look-alikes (a shouted "IGNORЕ" is the same attack)
                              "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
                              "С": "C", "Т": "T", "Х": "X", "І": "I", "Ѕ": "S", "Ο": "O", "Α": "A", "Ε": "E"})
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")

_AT = re.compile(r"\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*", re.I)  # bracketed forms only; bare " at " is ordinary prose
_DOT = re.compile(r"\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*", re.I)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NB, _NA = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"   # numeric PII never starts/ends inside an alphanumeric token (hex ids, hashes)
_SSN = re.compile(_NB + r"\d{3}[-\s.]?\d{2}[-\s.]?\d{4}" + _NA)
_PHONE = re.compile(_NB + r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}" + _NA + r"(?!\d)")
_PHONE_INTL = re.compile(_NB + r"\+\d{1,3}(?:[\s.-]?\d{1,4}){2,5}" + _NA)                  # +44 7700 900123, +33 2 34 56 78 90
_PHONE_UK = re.compile(_NB + r"0\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}" + _NA)                        # 020 7946 0958, 07700 900123
_IBAN = re.compile(_NB + r"[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){10,30}" + _NA)                               # GB29 NWBK 6016 1331 9268 19 (lengths vary by country)
_DOB = re.compile(r"(?i)\b(?:born|dob|d\.o\.b\.?|date of birth|birthday)\b[:\s]{0,3}(?:on\s+|is\s+)?"
                  r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|"
                  r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
                  r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\d{4})")
_TITLED_NAME = re.compile(r"(?i)\b(?:dr|mr|mrs|ms|prof|sir|dame|mx)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_STREET = (r"(?:[A-Za-z]+\s+){1,3}(?:street|st|avenue|ave|drive|dr|road|rd|court|ct|lane|way|kingsway|boulevard|place)\b\.?"
           r"|(?:rue|avenue|boulevard|via|calle)\s+(?:(?:de|du|la|le|des|del)\s+)?[A-Za-z]+")
_ADDRESS = re.compile(r"(?i)\b\d{1,4}\s+(?:" + _STREET + r"),?\s*(?:[A-Za-z]+,?\s+){1,3}"
                      r"(?:[A-Z]{2}\s+\d{5}(?:-\d{4})?|\d{5}|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b")
_CARD = re.compile(_NB + r"(?:\d[ -]?){13,19}" + _NA)
# A well-formed UUID is an identifier, not PII (its last group can be 12 digits and look like a phone number).
# UUID-shaped tokens are protected from the numeric patterns. Trade-off: a phone number deliberately written in
# 8-4-4-4-12 hex shape would not be redacted; that is far less likely than ids appearing in queries and audit.
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_STASH = re.compile(r"\x00(\d+)\x00")
_SECRET = re.compile(r"\b(?:sk-or-v1-|sk-ant-|sk-|AKIA|ghp_)[A-Za-z0-9_-]{12,}\b")


_DIGIT_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_DW_RUN = re.compile(r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine)\b(?:\s+|\s*[-./]\s*)){3,}"
                     r"(?:zero|one|two|three|four|five|six|seven|eight|nine)\b", re.I)
_SPACED_DIGITS = re.compile(r"(?<!\d)\d(?:[ .()/-]{1,3}\d){2,}(?!\d)")
_DOT_IN_EMAIL = re.compile(r"[\w-]+(?:\s+dot\s+[\w-]+)+(?=@)|@[\w-]+(?:\s+dot\s+[\w-]+)+", re.I)


def _fold_digit_words(m: re.Match[str]) -> str:
    return re.sub(r"[a-z]+", lambda w: _DIGIT_WORDS[w.group(0).lower()], m.group(0), flags=re.I)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH.sub("", t).translate(_CONFUSABLES)
    t = re.sub(r"[ \t]{2,}", " ", t)                  # doubled spaces must not split a phrase or a number
    t = _DOT.sub(".", t)
    t = _AT.sub("@", t)
    t = _DOT_IN_EMAIL.sub(lambda m: re.sub(r"\s+dot\s+", ".", m.group(0), flags=re.I), t)
    t = _DW_RUN.sub(_fold_digit_words, t)
    # spaced / dotted digit runs ("2 9 8 7 2 3 2 4 7") are one number; keep dashes as separators
    return _SPACED_DIGITS.sub(lambda m: re.sub(r"[ .]", "", m.group(0)), t)


def _luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        s, alt = s + d, not alt
    return s % 10 == 0


class PiiRedactor:
    name = "pii_redactor"
    priority = 10
    fail_closed = True
    error_code = ErrorCode.PII_BLOCKED

    def run(self, ctx: PreContext) -> None:
        t = normalise(ctx.text)
        counts: dict[str, int] = {}
        protected: list[str] = []

        def stash(m: re.Match[str]) -> str:
            protected.append(m.group(0))
            return f"\x00{len(protected) - 1}\x00"

        t = _UUID.sub(stash, t)

        def sub(pattern: re.Pattern[str], label: str, text: str, guard=None) -> str:
            def repl(m: re.Match[str]) -> str:
                if guard and not guard(m.group(0)):
                    return m.group(0)
                counts[label] = counts.get(label, 0) + 1
                return f"[{label}]"
            return pattern.sub(repl, text)

        t = sub(_SECRET, "SECRET", t)
        t = sub(_EMAIL, "EMAIL", t)
        t = sub(_CARD, "CARD", t, guard=lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and _luhn(re.sub(r"\D", "", s)))
        t = sub(_IBAN, "IBAN", t)
        t = sub(_DOB, "DOB", t)
        t = sub(_ADDRESS, "ADDRESS", t)
        t = sub(_TITLED_NAME, "NAME", t)
        t = sub(_SSN, "SSN", t)
        t = sub(_PHONE_INTL, "PHONE", t)
        t = sub(_PHONE_UK, "PHONE", t)
        t = sub(_PHONE, "PHONE", t)
        ctx.text = _STASH.sub(lambda m: protected[int(m.group(1))], t)
        ctx.flags["pii_redactions"] = counts
