# SMS Spam Detection: a reproducible, leakage-aware NLP system

TRACE is an educational English SMS classifier built with a static HTML/CSS/JavaScript frontend,
FastAPI, and a reproducible Python NLP pipeline. It lets visitors compare predictions for an
original message and a revision.

The portfolio release uses the evaluated V2 SMS spam classifier. Its historical evaluation does
not establish modern phishing detection. The separate [V3 experiment](docs/V3_DEVELOPMENT.md)
requires an explicit local opt-in and has not cleared independent evaluation.

![TRACE desktop composer](screenshots/trace-desktop.png)
![TRACE mobile comparison](screenshots/trace-mobile-comparison.png)

Classifies **English SMS** as *likely spam*, *likely ham* (legitimate), or *inconclusive*, with a
calibrated spam probability. Behind a small web UI sits an evaluation pipeline built to be trusted:
near-duplicate-aware splits, a locked and logged test set, cost-based decisions, and a model
artifact that rebuilds byte for byte and is served without scikit-learn, NumPy, or pickle.

> **Scope.** The model screens SMS for patterns seen in the 2011 SMS Spam Collection. It does
> **not** certify a message as safe, detect phishing or fraud in general, or classify e-mail.
> Modern smishing attacks may be missed. A ham prediction is not proof of safety. This application
> is for educational exploration, not security-critical decisions. See the [model card](docs/MODEL_CARD.md).

## Results

Leakage-controlled evaluation ([full report](docs/EVALUATION.md)). The v1 recipe is the original
project's pipeline (NLTK preprocessing, word counts, Naive Bayes), retrained on exactly the same
data for a fair comparison.

| | v1 recipe | **v2 (this model)** |
|---|---|---|
| Spam F1, 15-fold group CV (development) | 0.861 ± 0.044 | **0.949 ± 0.014** |
| Spam recall, CV | 0.761 | **0.913** |
| Expected cost / message, CV (false alarm = 3× a miss) | 0.0312 | **0.0149** (−52%) |
| Locked test: spam F1 [95% CI] | 0.914 [0.874, 0.948] | **0.980** [0.961, 0.996] |
| Locked test: missed spam / false alarms (of 128 / 903) | 17 / 4 | **5 / 0** |
| Locked test: PR-AUC | 0.959 | **0.994** |
| Serverless function bundle | 207 MiB | **20 MiB** |
| Prediction latency (typical SMS) | n/a | **0.17 ms** |

Treat the CV numbers as the planning estimate. The test set is held out but internal (same corpus
and era), with 128 spam messages.

**What the audit of the original project found:** 47% of the original holdout's spam messages had
a near-duplicate in its training data. On the leak-free part, the original model's spam recall was
0.84, not the reported 0.92 ([details](docs/EVALUATION.md#2-how-trustworthy-were-the-v1-numbers-python--m-spam_detector-audit-legacy)).

## How it works

```
text ─► abstain? (no letters / non-Latin / no known word) ─► normalize (NFKC, lowercase, mask URLs
     and numbers as zzurl / zzphone / zzcode / zznum) ─► word 1–2-grams + char 2–5-grams, TF-IDF
     ─► linear SVM score ─► Platt calibration ─► P(spam) ≥ 0.75 ? spam : ham
```

* **One preprocessing implementation** shared by training and serving ([ADR 0003](docs/adr/0003-preprocessing.md)).
* **Group-aware evaluation:** exact and near-duplicate campaigns never straddle a split ([ADR 0001](docs/adr/0001-evaluation-protocol.md)).
* **Model selection by expected cost** with a pre-registered one-standard-error rule over 11 candidates ([ADR 0002](docs/adr/0002-model-selection.md)).
* **Pickle-free, checksummed JSON artifact** executed by ~475 lines of standard-library Python, verified against scikit-learn to 6e-15 ([ADR 0004](docs/adr/0004-pickle-free-artifact.md)).
* **Explicit abstention** with machine-readable reasons ([ADR 0005](docs/adr/0005-abstention.md)).

Architecture and module map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (run the app)

Python 3.12. The release artifact is committed, so serving needs no dataset and no ML libraries.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. API docs at `/docs`.

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" -d '{\"text\":\"WINNER!! Claim your prize, call 09061701461\"}'
# {"label":"spam","spam_probability":0.9996,"reason":null,"model_version":"2.0.0"}
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | `{"text": "..."}` (1–5,000 code points) → `label`, `spam_probability`, `reason`, `model_version` |
| `GET /health` · `GET /ready` | Liveness · readiness (verified model loaded) |
| `GET /model` | Version, threshold, limits, evaluation summary and caveats |

Full contract, error codes, and the frontend handoff: [docs/API.md](docs/API.md).

## Train, evaluate, reproduce

```powershell
python -m pip install -r requirements-dev.txt
# download spam.csv into data/ (instructions and checksum: docs/DATASET.md)
python scripts/prepare_nltk.py                 # only to reproduce the v1 baseline
python -m spam_detector data                   # repair, dedupe, group, split -> reports/data/
python -m spam_detector audit-legacy           # quantify the original protocol's leakage
python -m spam_detector experiment             # 11 candidates x 15 group-CV folds -> reports/experiments/
python -m spam_detector reproduce              # rebuild models/spam-v2 and compare sha256
python -m spam_detector explain "Free entry! Txt WIN to 87121"
```

All parameters live in [`configs/pipeline.toml`](configs/pipeline.toml). `release` never overwrites
an artifact, and `evaluate` (locked test set) requires an explicit flag and logs every run.

## Tests

```powershell
python -m pytest --cov=spam_detector --cov=app          # dataset-gated tests skip without data/spam.csv
python -m ruff check .
npm --prefix tests/browser ci
npm --prefix tests/browser exec -- playwright install chromium
npm --prefix tests/browser test                          # browser regressions
```

Coverage includes featurizer parity with scikit-learn, export equivalence for LR/SVM/NB, artifact
tampering and incompatibility, leakage assertions, golden predictions (Unicode, emoji, other
scripts), the HTTP contract (413/422/503/500), privacy of logs, an import guard that keeps
scikit-learn out of the API process, and an end-to-end pipeline run on synthetic data.
CI (`.github/workflows/ci.yml`) runs lint and tests, an API smoke test with **only** the runtime
dependencies installed, and the Playwright suite.

## Deployment

Vercel, one Python function, no build step. Runbook, safeguards, and hosted checks:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Production deployment requires explicit owner approval.

## Documentation

| Document | What it covers |
|---|---|
| [LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | Every component explained, plus interview questions with answers |
| [EVALUATION.md](docs/EVALUATION.md) | Protocol, all results, error analysis, test-set access log |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use, limits, biases, performance |
| [DATASET.md](docs/DATASET.md) | Provenance, license, acquisition, data-quality repairs |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagram, modules, request lifecycle, dependencies |
| [API.md](docs/API.md) | HTTP contract and frontend integration guide |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Vercel build, preview runbook, rollback |
| [docs/adr/](docs/adr/) | Six architecture decision records |

## Improvements over the original project

| Area | Before | After |
|---|---|---|
| Data | Truncated 50 messages, mojibake (`å£` for `£`), HTML entities, dataset-only `<#>` markers | Repaired and validated, with a fingerprinted dataset |
| Evaluation | One random split, near-duplicate leakage, model chosen on the test set | Group-aware repeated CV, locked and logged test set, bootstrap CIs |
| Preprocessing | Deleted digits, `£`, `!`, URLs. Notebook and API differed | One shared normalizer that keeps typed placeholders |
| Model | Count NB, uncalibrated, 0.5 threshold | Word+char TF-IDF SVM, Platt-calibrated, cost-derived threshold |
| Responsible output | Abstained only on zero features | Abstains on no letters, non-Latin script, or unknown words, with a reason |
| Artifact | Pickles tied to one scikit-learn version | Checksummed JSON, byte-for-byte reproducible, compatibility-checked |
| Serving | 207 MiB bundle, NLTK download at build | 20 MiB, FastAPI only, readiness/model endpoints, privacy-safe logs, body limits, CSP |
| Engineering | 8 API tests | 157 Python tests + 19 browser tests, CI, ADRs, model card |

## Dataset and license

The SMS Spam Collection (Almeida & Gómez Hidalgo, 2011) is distributed by UCI under CC BY 4.0
([doi:10.24432/C5CC84](https://doi.org/10.24432/C5CC84)). The Kaggle CSV used here is not
redistributed in this repository, see [docs/DATASET.md](docs/DATASET.md). The original v1 notebook
is kept in `notebook/` for history. Its metrics come from the v1 protocol and are superseded by
[docs/EVALUATION.md](docs/EVALUATION.md).
