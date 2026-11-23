"""Phase 0 model benchmark (design.md R-5, §3 Phase 0).

Measures TTFT, tokens/s and error behaviour of candidate models *through OpenRouter* at the real prompt
size (~1.4K input tokens, <=96 output tokens), and records which upstream provider served each request.

    python -m kestrel.llm.bench --list --max-price-per-mtok 3
    python -m kestrel.llm.bench --models a/b,c/d --n 10 --concurrency 2 --budget-usd 0.50

Hard caps: --n requests per model, --budget-usd across the whole run (stops when reported cost exceeds it).
The API key is read from OPENROUTER_API_KEY and never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

from ..config import Settings
from ..contracts import PlatformError
from .client import OpenRouterClient

FILLER = ("Policy excerpt: tenants must retain approved supplier certificates for seven years; lot recalls "
          "require written notice within 24 hours; SKU-099 and LOT-00044 are governed by clause 4.2. ")


def build_prompt(target_tokens: int) -> list[dict[str, str]]:
    chars = int(target_tokens * 2.7)   # this filler is token-dense (~2.7 chars/token); real count comes from usage
    context = (FILLER * (chars // len(FILLER) + 1))[:chars]
    return [{"role": "system", "content": "Answer strictly from the context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: Explain the retention and recall-notice rules in about 80 words."}]


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


async def list_models(settings: Settings, max_price: float | None) -> None:
    async with httpx.AsyncClient(timeout=20) as h:
        r = await h.get(settings.openrouter_base_url.rstrip("/") + "/models")
        r.raise_for_status()
    rows = []
    for m in r.json().get("data", []):
        p = m.get("pricing") or {}
        try:
            pin, pout = float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            continue
        if max_price is not None and (pin > max_price or pout > max_price * 5):
            continue
        rows.append((pin, pout, m["id"], m.get("context_length")))
    for pin, pout, mid, ctx in sorted(rows)[:60]:
        print(f"{mid:60s} in ${pin:7.3f}/M  out ${pout:7.3f}/M  ctx {ctx}")
    print(f"({len(rows)} models match)", file=sys.stderr)


async def one(client: OpenRouterClient, model: str, prompt: list[dict[str, str]], max_tokens: int, extra: dict | None = None) -> dict:
    t0 = time.perf_counter()
    rec: dict = {"model": model, "ok": False}
    n_tok = 0
    t_first = t_last = None
    try:
        async for ev in client.stream(prompt, model=model, max_tokens=max_tokens, extra_body=extra):
            if ev.kind == "reasoning":
                rec["reasoning_seen"] = True
                rec.setdefault("ttft_any_ms", ev.t_ms)
            if ev.kind == "token":
                rec.setdefault("ttft_any_ms", ev.t_ms)
                n_tok += 1
                t_first = t_first if t_first is not None else ev.t_ms
                t_last = ev.t_ms
            else:
                rec.update(ok=True, ttft_ms=ev.ttft_ms, total_ms=ev.total_ms, provider=ev.provider,
                           served_model=ev.model, prompt_tokens=ev.usage.prompt_tokens,
                           completion_tokens=ev.usage.completion_tokens, cost_usd=ev.usage.cost_usd)
        if rec["ok"] and t_first is not None and t_last and t_last > t_first and n_tok > 1:
            rec["chunks_per_s"] = round((n_tok - 1) / ((t_last - t_first) / 1000), 1)
            ct = rec.get("completion_tokens") or n_tok
            rec["tokens_per_s"] = round((ct - 1) / ((t_last - t_first) / 1000), 1)
    except PlatformError as e:
        rec.update(error=e.code.name, detail=e.message)
    except Exception as e:  # noqa: BLE001
        rec.update(error=type(e).__name__)
    rec["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return rec


async def run(models: list[str], n: int, conc: int, prompt_tokens: int, max_tokens: int, budget: float, extra: dict | None = None) -> dict:
    settings = Settings.from_env()
    client = OpenRouterClient(settings)
    prompt = build_prompt(prompt_tokens)
    spent = 0.0
    results: dict[str, list[dict]] = {}
    sem = asyncio.Semaphore(conc)
    stop = False
    try:
        for model in models:
            results[model] = []

            async def task() -> None:
                nonlocal spent, stop
                async with sem:
                    if stop:
                        return
                    rec = await one(client, model, prompt, max_tokens, extra)
                    spent += rec.get("cost_usd") or 0.0
                    if spent > budget:
                        stop = True
                    results[model].append(rec)

            await task()                                    # warm-up request, kept in the log but excluded below
            await asyncio.gather(*(task() for _ in range(n)))
            if stop:
                print(f"budget ${budget:.2f} reached (spent ${spent:.4f}); stopping", file=sys.stderr)
                break
    finally:
        await client.aclose()
    return {"prompt_tokens_target": prompt_tokens, "max_tokens": max_tokens, "concurrency": conc, "extra_body": extra,
            "spent_usd": round(spent, 5), "results": results}


def summarise(out: dict) -> list[dict]:
    rows = []
    for model, recs in out["results"].items():
        recs = recs[1:]                                     # drop warm-up
        ok = [r for r in recs if r["ok"]]
        ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
        tany = [r["ttft_any_ms"] for r in ok if r.get("ttft_any_ms") is not None]
        tps = [r["tokens_per_s"] for r in ok if r.get("tokens_per_s")]
        tot = [r["total_ms"] for r in ok]
        rows.append({
            "model": model, "n": len(recs), "ok": len(ok),
            "errors": sorted({r.get("error", "?") for r in recs if not r["ok"]}),
            "ttft_ms_p50": pct(ttft, 50), "ttft_ms_p95": pct(ttft, 95),
            "ttft_any_p50": pct(tany, 50), "reasoning_seen": any(r.get("reasoning_seen") for r in ok),
            "completion_tokens": statistics.median([r["completion_tokens"] for r in ok]) if ok else None,
            "tok_per_s_p50": pct(tps, 50), "total_ms_p50": pct(tot, 50), "total_ms_p95": pct(tot, 95),
            "prompt_tokens": statistics.median([r["prompt_tokens"] for r in ok]) if ok else None,
            "providers": sorted({r.get("provider") or "?" for r in ok}),
            "meets_R5_ttft_le_59ms": bool(ttft) and pct(ttft, 95) is not None and pct(ttft, 95) <= 59,
            "meets_R5_420_tok_s": bool(tps) and (pct(tps, 50) or 0) >= 420,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list models (no key needed, no cost)")
    ap.add_argument("--max-price-per-mtok", type=float, default=None)
    ap.add_argument("--models", default="")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=1400)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--budget-usd", type=float, default=0.50)
    ap.add_argument("--extra", default="", help='JSON merged into the request body, e.g. {"provider":{"sort":"latency"}}')
    a = ap.parse_args()
    if a.list:
        asyncio.run(list_models(Settings.from_env(), a.max_price_per_mtok))
        return
    models = [m for m in a.models.split(",") if m]
    if not models:
        ap.error("--models is required (use --list to find candidates)")
    out = asyncio.run(run(models, a.n, a.concurrency, a.prompt_tokens, a.max_tokens, a.budget_usd, json.loads(a.extra) if a.extra else None))
    rows = summarise(out)
    Path("bench_results").mkdir(exist_ok=True)
    path = Path("bench_results") / f"bench_{int(time.time())}.json"
    path.write_text(json.dumps({**out, "summary": rows}, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"\nraw results: {path}  (spent ${out['spent_usd']})", file=sys.stderr)


if __name__ == "__main__":
    main()
