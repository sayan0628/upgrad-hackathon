# Learned triage: results

Artifact: `models/triage/model.joblib` + `meta.json` (v1.0.0). Trainer: `tools/train_triage.py`.
Training data: `data/insurance/golden/triage.jsonl`, SHA-256 `fe6515bce898fd544dad359c2d3655fbb70a6b3cfc518694d8b96a3e4f56882e` (319 rows).

## Approach

- Features: char_wb 2-5 + word 1-2 TF-IDF, concatenated with the all-MiniLM-L6-v2 sentence embedding (384-d, normalised).
- Head: logistic regression, `class_weight=balanced`.
- Candidates compared by 5-fold grouped CV on the dev set: TF-IDF only (C 0.3/1/3), embedding only (C 1/10/30), TF-IDF + embedding (C 3/10).
- Selected: TF-IDF + embedding, C=3 (CV 99.2%). Ties go to the first-listed candidate.
- Groups: `source` (paraphrases of one golden question). Exception: the 20 `chat` rows (DIRECT_ANSWER greetings) are distinct utterances, so each is its own group. Otherwise all DIRECT_ANSWER would fall in one fold.
- Split: `StratifiedGroupKFold(5, seed 20260)`. Fold 0 (64 rows, 64 groups) is the held-out set. The remaining 255 rows are dev. Held-out groups share no source with dev. The held-out set was scored once, after the candidate was chosen by CV. Nothing was tuned against it.
- Final artifact trained on the 255 dev rows only, so the held-out score is not inflated by training on it.

## Grouped 5-fold CV on dev (selection)

| candidate | CV accuracy | folds |
|---|---|---|
| TF-IDF LR C=0.3 | 93.7% | 96.1 / 92.2 / 94.1 / 90.2 / 96.1 |
| TF-IDF LR C=1 | 93.3% | |
| TF-IDF LR C=3 | 92.9% | |
| embedding LR C=1 | 98.0% | |
| embedding LR C=10 | 98.4% | |
| embedding LR C=30 | 98.4% | |
| TF-IDF + embedding LR C=3 (**selected**) | **99.2%** | 100 / 100 / 98.0 / 98.0 / 100 |
| TF-IDF + embedding LR C=10 | 99.2% | |

## Held-out (64 rows, untouched split)

**Accuracy: 96.88% (62/64). Target KPI-04 is >= 97.0%. Not met.** At n=64, 97% needs at most one miss, so this is 2 misses from the target line.

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| DIRECT_ANSWER | 0.800 | 1.000 | 0.889 | 4 |
| HYBRID_RAG | 1.000 | 0.976 | 0.988 | 41 |
| REJECT_OOS | 1.000 | **0.889** | 0.941 | 9 |
| TRANSACTION_EXEC | 0.909 | 1.000 | 0.952 | 10 |

Confusion matrix (rows = true, columns = predicted; order DIRECT_ANSWER, HYBRID_RAG, REJECT_OOS, TRANSACTION_EXEC):

```
[[ 4  0  0  0]
 [ 0 40  0  1]
 [ 1  0  8  0]
 [ 0  0  0 10]]
```

Misses:
- R-298 REJECT_OOS -> DIRECT_ANSWER: "Translate good morning into French." This is an out-of-scope request with greeting vocabulary. The chat bucket is small (20 rows), so the model over-generalises greetings.
- R-116 HYBRID_RAG -> TRANSACTION_EXEC: "Can a SUBMITTED claim move directly to APPROVED?" This is a question about a state transition, phrased like a transaction command.

Failure pattern: the weakest class is REJECT_OOS (recall 0.889, 1 of 9 missed, to DIRECT_ANSWER). The main confusion is between the greeting/chat class and out-of-scope requests, plus HYBRID vs TRANSACTION for state-transition questions. Both are semantic edge cases, and the training set has only 20 chat rows and 48 REJECT_OOS rows.

## Deterministic overrides (measured on dev-OOF only)

Overrides from `RuleTriage` were tested on the dev out-of-fold predictions:
- `_TXN` override: no change (38/38 already right).
- `_OOS` override: no change (1 row, already right).
- `_PERSONAL` override: 99.2% -> 98.4%, because the labels mark personal-account questions such as "What is my policy number?" as HYBRID_RAG.

Decision: ship the model alone, with no overrides. `complexity` and `risk_tier` still come from `RuleTriage` logic (not learned).

## Latency (classify(), 200 calls, CPU, 2 torch threads)

| condition | p50 | p95 |
|---|---|---|
| under load (load avg ~5-9, another worker running `tools/retrieval_eval.py`) | 57.6 ms | 182.8 ms |
| quieter box, 200 calls on one sentence (ad-hoc profile, load avg ~5) | 27.0 ms | 34.8 ms |

Breakdown: the MiniLM encode alone is about 22 ms p50 (29 ms p95). TF-IDF, LR, and rules together are under 0.1 ms.

**KPI-05 (p95 <= 25 ms) is not met.** The embedding forward pass is the cost. Options, not done here: reuse the per-request embedding that dense retrieval already computes (design.md §1.4.1), or quantise. The TF-IDF-only model is sub-millisecond but scores 93.7% CV, which is far below target.

## Status

- Not the default. `runtime.py` and `gateway/app.py` still use `RuleTriage`, because held-out accuracy is below 97% and latency is above 25 ms.
- Versions: sklearn 1.9.1, sentence-transformers 6.1.0, Python per `meta.json`.
