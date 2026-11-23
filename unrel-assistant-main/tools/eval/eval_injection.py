"""Score the injection shield (KPI-03 block rate on attacks, KPI-06 false-positive rate on benign prompts).

Usage: python3 tools/eval/eval_injection.py [--split test|dev] [--show-misses N]
A prompt counts as blocked when InjectionShield raises PlatformError (the same path as the pipeline).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "bench_results" / "eval" / "injection.jsonl"


def blocked(text: str) -> bool:
    from kestrel.contracts import PlatformError
    from kestrel.plugins import InjectionShield, PreContext
    try:
        InjectionShield().run(PreContext("bench", text))
        return False
    except PlatformError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--show-misses", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in DATA.open(encoding="utf-8") if json.loads(l)["split"] == a.split]
    fam = defaultdict(lambda: [0, 0])
    misses = []
    for r in rows:
        if r["label"] == "attack":
            hit = blocked(r["text"])
            fam[r["family"]][0] += hit
            fam[r["family"]][1] += 1
            if not hit:
                misses.append(r["text"])
        else:
            fp = blocked(r["text"])
            fam["BENIGN (false positives)"][0] += fp
            fam["BENIGN (false positives)"][1] += 1
    atk = [v for k, v in fam.items() if not k.startswith("BENIGN")]
    tot = sum(v[0] for v in atk), sum(v[1] for v in atk)
    print(f"split={a.split}")
    for k in sorted(fam):
        h, n = fam[k]
        print(f"{k:<26} {h}/{n} ({100 * h / n:.1f}%)")
    print(f"ATTACK recall (KPI-03)    {tot[0]}/{tot[1]} ({100 * tot[0] / tot[1]:.2f}%)")
    b = fam["BENIGN (false positives)"]
    print(f"BENIGN false-positive     {b[0]}/{b[1]} ({100 * b[0] / b[1]:.2f}%)  (KPI-06)")
    for t in misses[:a.show_misses]:
        print("MISS:", repr(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
