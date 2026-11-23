"""Identifier-preserving tokeniser: SKU-099, LOT-00044 and clause numbers (4.2) stay single tokens."""
from __future__ import annotations

import re

STOP = frozenset("a an the of to in on for and or is are was were be been being it its this that these those "
                 "with as at by from what which who whom when where how does do did can could should would will "
                 "i you we they he she them our your their my me us not no than then there here if any all".split())

_TOKEN = re.compile(r"(?i)\b(?:sku|lot|clm|pol)[-_ ]?\d+\b|\d+(?:\.\d+)+|[a-z][a-z']*|\d+")


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 6 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 5 and t.endswith("ed"):
        t = t[:-2]
    return t


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    out: list[str] = []
    for m in _TOKEN.finditer(text):
        tok = m.group(0).lower()
        if tok.startswith(("sku", "lot", "clm", "pol")) and any(c.isdigit() for c in tok):
            kind = tok[:3]
            digits = re.sub(r"\D", "", tok)
            out.append(f"{kind}-{digits.zfill({'sku': 3, 'lot': 5, 'clm': 5, 'pol': 6}[kind])}")
        elif tok[0].isdigit():
            out.append(tok)
        elif keep_stop or tok not in STOP:
            out.append(_stem(tok))
    return out
