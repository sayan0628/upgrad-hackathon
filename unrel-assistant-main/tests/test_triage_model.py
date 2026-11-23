"""Loads the shipped learned-triage artifact and checks it against the held-out group split it was measured on.

The asserted accuracy is the value measured by tools/train_triage.py (62/64 = 96.88%), NOT the KPI-04 target. If the
artifact or the training data changes, re-run tools/train_triage.py and update MEASURED_HELDOUT_CORRECT here.
"""
import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "models" / "triage"
DATA = ROOT / "data" / "insurance" / "golden" / "triage.jsonl"
MEASURED_HELDOUT_CORRECT, MEASURED_HELDOUT_N = 62, 64

pytestmark = pytest.mark.slow


def _hf_cached() -> bool:
    return (pathlib.Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2").exists()


@pytest.fixture(scope="module")
def artifact():
    if not (ART / "model.joblib").exists() or not (ART / "meta.json").exists():
        pytest.skip("models/triage artifact not built; run tools/train_triage.py")
    if not _hf_cached():
        pytest.skip("all-MiniLM-L6-v2 not cached locally (offline)")
    from kestrel.triage_model import LearnedTriage
    return LearnedTriage(ART)


def test_artifact_matches_training_data(artifact):
    meta = artifact.meta
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == meta["train_data_sha256"], "training data changed; retrain"
    assert meta["heldout_n"] == MEASURED_HELDOUT_N
    assert meta["heldout_accuracy"] == round(MEASURED_HELDOUT_CORRECT / MEASURED_HELDOUT_N, 4)


def test_heldout_accuracy_as_measured(artifact):
    ids = set(artifact.meta["heldout_ids"])
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    held = [r for r in rows if r["id"] in ids]
    assert len(held) == MEASURED_HELDOUT_N
    # the held-out groups must not share a source with the training rows
    train_sources = {r["source"] for r in rows if r["id"] not in ids and r["source"] != "chat"}
    assert not {r["source"] for r in held if r["source"] != "chat"} & train_sources
    correct = sum(artifact.classify(r["text"]).route == r["label"] for r in held)
    assert correct == MEASURED_HELDOUT_CORRECT


def test_classify_returns_valid_result(artifact):
    from kestrel.triage import TriageResult
    res = artifact.classify("Move policy POL-100004 to IN_FORCE")
    assert isinstance(res, TriageResult)
    assert res.route == "TRANSACTION_EXEC" and res.risk_tier == 3
    assert artifact.classify("Hi there").route in {"DIRECT_ANSWER", "HYBRID_RAG", "REJECT_OOS", "TRANSACTION_EXEC"}
