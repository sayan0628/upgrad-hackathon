"""Train + grouped-evaluate the learned triage router (src/kestrel/triage_model.py).

    python3 tools/train_triage.py            # select by grouped CV on dev, fit on dev, score held-out ONCE, save artifact
    python3 tools/train_triage.py --latency 200

Split: rows are grouped by `source` (paraphrases of one golden question share a group). StratifiedGroupKFold with a
fixed seed gives 5 folds; fold 0 is the held-out set (never used for selection), folds 1-4 are the dev set on which the
candidate models are compared by 5-fold grouped CV. Exception: source "chat" is a bucket of 20 distinct greetings, not
paraphrases, so each of those rows is its own group (otherwise all DIRECT_ANSWER rows would fall in one fold).
Only data/insurance/golden/triage.jsonl is used, so the held-out set stays clean.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import platform
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_union  # noqa: E402

DATA = ROOT / "data" / "insurance" / "golden" / "triage.jsonl"
OUT = ROOT / "models" / "triage"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ["DIRECT_ANSWER", "HYBRID_RAG", "REJECT_OOS", "TRANSACTION_EXEC"]
SPLIT_SEED, CV_SEED = 20260, 7


def load():
    raw = DATA.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    groups = [r["source"] if r["source"] != "chat" else f"chat::{r['id']}" for r in rows]
    return rows, groups, hashlib.sha256(raw).hexdigest()


def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device="cpu")
    return np.asarray(m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)


def tfidf_union():
    return make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                      TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True))


def features(kind, tf, texts, E):
    """kind: 'tfidf' | 'emb' | 'both'. tf is a fitted union (or None for emb)."""
    parts = []
    if kind in ("tfidf", "both"):
        parts.append(tf.transform(texts))
    if kind in ("emb", "both"):
        parts.append(csr_matrix(E))
    return parts[0] if len(parts) == 1 else hstack(parts).tocsr()


def fit(kind, C, texts, y, E):
    tf = None if kind == "emb" else tfidf_union().fit(texts)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit(features(kind, tf, texts, E), y)
    return {"tfidf": tf, "clf": clf}


def predict(pipe, kind, texts, E):
    return pipe["clf"].predict(features(kind, pipe["tfidf"], texts, E))


CANDIDATES = [("tfidf", 0.3), ("tfidf", 1.0), ("tfidf", 3.0),
              ("emb", 1.0), ("emb", 10.0), ("emb", 30.0),
              ("both", 3.0), ("both", 10.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=int, default=200, help="number of classify() calls to time")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    rows, groups, sha = load()
    texts = np.array([r["text"] for r in rows], dtype=object)
    y = np.array([LABELS.index(r["label"]) for r in rows])
    g = np.array(groups)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SPLIT_SEED).split(np.zeros(len(y)), y, g))
    test_idx, dev_idx = folds[0][1], folds[0][0]
    cv = [(dev_idx[tr], dev_idx[va]) for tr, va in
          StratifiedGroupKFold(5, shuffle=True, random_state=CV_SEED).split(np.zeros(len(dev_idx)), y[dev_idx], g[dev_idx])]
    assert not set(g[test_idx]) & set(g[dev_idx]), "group leakage"

    print(f"rows={len(rows)} groups={len(set(groups))} dev={len(dev_idx)} heldout={len(test_idx)} sha256={sha}")
    print("heldout labels:", dict(collections.Counter(LABELS[i] for i in y[test_idx])))
    E = embed(list(texts))

    # ---- model selection: grouped 5-fold CV on the dev set only
    print("\nGrouped 5-fold CV on dev (selection):")
    results = []
    for kind, C in CANDIDATES:
        accs = []
        for tr, va in cv:
            p = fit(kind, C, texts[tr], y[tr], E[tr])
            accs.append(accuracy_score(y[va], predict(p, kind, texts[va], E[va])))
        results.append({"kind": kind, "C": C, "cv_mean": float(np.mean(accs)), "cv_folds": [float(x) for x in accs]})
        print(f"  {kind:6s} C={C:<5} cv_acc={np.mean(accs):.4f} folds={np.round(accs, 3).tolist()}")
    best = max(results, key=lambda r: r["cv_mean"])   # ties: first listed (simplest) wins
    print(f"selected: {best['kind']} C={best['C']} (cv {best['cv_mean']:.4f})")

    # ---- fit on all dev rows, score held-out ONCE
    pipe = fit(best["kind"], best["C"], texts[dev_idx], y[dev_idx], E[dev_idx])
    pred = predict(pipe, best["kind"], texts[test_idx], E[test_idx])
    yt = y[test_idx]
    acc = accuracy_score(yt, pred)
    p, r, f, s = precision_recall_fscore_support(yt, pred, labels=range(4), zero_division=0)
    cm = confusion_matrix(yt, pred, labels=range(4))
    print(f"\nHELD-OUT accuracy: {acc:.4f} ({int((pred == yt).sum())}/{len(yt)})")
    print(f"{'class':18s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'n':>4s}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:18s} {p[i]:6.3f} {r[i]:7.3f} {f[i]:6.3f} {s[i]:4d}")
    print("confusion matrix (rows=true, cols=pred), order:", LABELS)
    print(cm)
    errors = [(LABELS[yt[k]], LABELS[pred[k]], rows[test_idx[k]]["id"], texts[test_idx[k]])
              for k in range(len(yt)) if yt[k] != pred[k]]
    for t, pr, rid, tx in errors:
        print(f"  miss {rid}: {t} -> {pr} | {tx}")

    if a.no_save:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT / "model.joblib", compress=3)
    meta = {
        "name": "kestrel learned triage", "version": "1.0.0",
        "model": f"{best['kind']} LogisticRegression(C={best['C']}, class_weight=balanced)",
        "features": {"tfidf": "char_wb 2-5 + word 1-2, sublinear", "embedding": EMBED_MODEL},
        "classes": LABELS, "embed_model": EMBED_MODEL,
        "train_data": str(DATA.relative_to(ROOT)), "train_data_sha256": sha, "n_rows": len(rows),
        "split": {"method": "StratifiedGroupKFold(5) by source; fold 0 = held-out", "seed": SPLIT_SEED, "cv_seed": CV_SEED},
        "heldout_ids": [rows[i]["id"] for i in test_idx],
        "trained_on": "dev rows only (held-out rows excluded)", "n_train": int(len(dev_idx)),
        "cv_selection": results, "selected": best,
        "heldout_accuracy": round(float(acc), 4), "heldout_n": int(len(yt)),
        "heldout_per_class": {LABELS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                                          "n": int(s[i])} for i in range(4)},
        "heldout_confusion": cm.tolist(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {OUT / 'model.joblib'} ({(OUT / 'model.joblib').stat().st_size / 1024:.0f} KiB) and meta.json")

    if a.latency:
        from kestrel.triage_model import LearnedTriage
        lt = LearnedTriage(OUT)
        held = [rows[i]["text"] for i in test_idx]
        for t in held[:5]:
            lt.classify(t)  # warm-up
        ts = []
        for k in range(a.latency):
            t0 = time.perf_counter()
            lt.classify(held[k % len(held)])
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        pct = lambda q: ts[min(len(ts) - 1, int(round(q / 100 * len(ts) + 0.5)) - 1)]  # noqa: E731
        print(f"classify() latency over {len(ts)} calls: p50={pct(50):.2f} ms p95={pct(95):.2f} ms "
              f"max={ts[-1]:.2f} ms mean={statistics.mean(ts):.2f} ms")


if __name__ == "__main__":
    main()
