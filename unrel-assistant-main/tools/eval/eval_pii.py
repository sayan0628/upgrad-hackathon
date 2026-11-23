"""Score PII recall (KPI-02) for PiiRedactor and Presidio on bench_results/eval/pii_spans.jsonl.

A span counts as caught when its identifying content (letters and digits after folding confusables,
zero-width characters and bracket obfuscations) no longer appears in the redacted output.
Usage: python3 tools/eval/eval_pii.py [--split test|dev] [--skip-presidio]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "bench_results" / "eval" / "pii_spans.jsonl"

_ZW = re.compile("[​‌‍⁠﻿]")
_BR_AT = re.compile(r"\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*", re.I)
_BR_DOT = re.compile(r"\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*", re.I)


def fold(s: str) -> str:
    """Independent of the redactor: NFKC, strip zero-width, map Cyrillic/Greek letters to their Latin look-alike name."""
    out = []
    for ch in unicodedata.normalize("NFKC", _ZW.sub("", s)):
        name = unicodedata.name(ch, "")
        if ("CYRILLIC" in name or "GREEK" in name) and name.split()[-1].isalpha() and len(name.split()[-1]) == 1:
            ch = name.split()[-1].lower()
        out.append(ch)
    return "".join(out)


def key(span: str) -> str:
    s = _BR_AT.sub("@", fold(span))
    s = _BR_DOT.sub(".", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def caught(span: str, redacted: str) -> bool:
    k = key(span)
    return not k or k not in re.sub(r"[^a-z0-9]", "", fold(redacted).lower())


def redactor_out(text: str) -> str:
    from kestrel.plugins import PiiRedactor
    from kestrel.plugins.base import PreContext
    ctx = PreContext("bench", text)
    PiiRedactor().run(ctx)
    return ctx.text


_presidio = None


def presidio_out(text: str) -> str:
    global _presidio
    if _presidio is None:
        from presidio_analyzer import AnalyzerEngine
        _presidio = AnalyzerEngine()
    res = _presidio.analyze(text=text, language="en")
    for r in sorted(res, key=lambda r: r.start, reverse=True):
        text = text[:r.start] + f"<{r.entity_type}>" + text[r.end:]
    return text


def score(rows, fn) -> dict[str, tuple[int, int]]:
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        ok = caught(r["span"], fn(r["text"]))
        tally[r["type"]][0] += ok
        tally[r["type"]][1] += 1
        tally["ALL"][0] += ok
        tally["ALL"][1] += 1
    return {k: (v[0], v[1]) for k, v in tally.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-presidio", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == a.split]
    systems = {"PiiRedactor": redactor_out}
    if not a.skip_presidio:
        systems["Presidio"] = presidio_out
    results = {name: score(rows, fn) for name, fn in systems.items()}
    types = sorted({t for r in results.values() for t in r if t != "ALL"}) + ["ALL"]
    print(f"split={a.split} n={len(rows)}")
    print(f"{'type':<14}{'n':>6}" + "".join(f"{name:>22}" for name in results))
    for t in types:
        n = next(v[t][1] for v in results.values() if t in v)
        cells = "".join(f"{v[t][0]}/{v[t][1]} ({100 * v[t][0] / v[t][1]:.1f}%)".rjust(22) for v in results.values())
        print(f"{t:<14}{n:>6}{cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
