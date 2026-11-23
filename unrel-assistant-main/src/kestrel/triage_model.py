"""Learned triage (design.md §0.3 "Triage": distilled small classifier, no LLM, no temperature).

Route = logistic regression over [char+word TF-IDF, all-MiniLM-L6-v2 sentence embedding]. Trained offline by
tools/train_triage.py; the artifact in models/triage/ records the training-data SHA-256, the split and the measured
scores. complexity / risk_tier are NOT learned: they come from RuleTriage's deterministic logic, which the pipeline
depends on (complexity == "low" enables the extractive fast path).

Inference needs scikit-learn (joblib artifact), sentence-transformers and the cached all-MiniLM-L6-v2 weights.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from .triage import RuleTriage, TriageResult

DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "triage"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LearnedTriage:
    def __init__(self, artifact_dir: str | os.PathLike[str] = DEFAULT_DIR):
        import joblib
        from sentence_transformers import SentenceTransformer

        d = pathlib.Path(artifact_dir)
        self.meta: dict[str, Any] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self._pipe = joblib.load(d / "model.joblib")          # {"tfidf": FeatureUnion, "clf": LogisticRegression}
        # classifier was fit on integer indices into meta["classes"]
        self._classes = [str(c) for c in self.meta["classes"]]
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._emb = SentenceTransformer(self.meta.get("embed_model", EMBED_MODEL), device="cpu")
        self._rules = RuleTriage()

    def route_of(self, text: str) -> str:
        import numpy as np
        from scipy.sparse import csr_matrix, hstack

        e = self._emb.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        x = hstack([self._pipe["tfidf"].transform([text]), csr_matrix(np.asarray(e, dtype=np.float64))]).tocsr()
        return self._classes[int(self._pipe["clf"].predict(x)[0])]

    def classify(self, text: str) -> TriageResult:
        route = self.route_of(text)
        rule = self._rules.classify(text)
        if route == "REJECT_OOS":
            return TriageResult("REJECT_OOS")
        if route == "TRANSACTION_EXEC":
            return TriageResult("TRANSACTION_EXEC", "high", 3)
        # HYBRID_RAG / DIRECT_ANSWER: complexity from the rule logic (drives the extractive fast path)
        return TriageResult(route, rule.complexity if route == "HYBRID_RAG" else "low")  # type: ignore[arg-type]
