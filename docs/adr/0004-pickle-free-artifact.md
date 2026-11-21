# ADR 0004: Pickle-free JSON artifact and a standard-library inference runtime

**Status:** accepted · **Date:** 2026-11-21

## Context

v1 served two joblib pickles through scikit-learn. The Vercel function was **207.15 MiB** (close
to the 250 MB limit, achieved only by deleting upstream test folders at build time), and the
first hosted request took 6.7 s. Pickles also execute code on load and only load with the exact
scikit-learn version that wrote them.

## Decision

Every candidate we consider is linear in its feature vector: logistic regression and SVMs
directly, and Multinomial NB as `(log θ_spam − log θ_ham)·x + log-prior ratio`. So a model is fully
described by vocabulary, IDF weights, coefficients, intercept, Platt `(a, b)`, threshold, and
abstention policy.

* `spam_detector/features.py` re-implements the needed subset of `TfidfVectorizer`/`CountVectorizer`
  in pure Python. **Training uses the same code** (wrapped into a SciPy matrix), so there is one
  featurizer.
* `model.json` is written deterministically (sorted keys, no timestamps), so retraining reproduces
  it byte for byte. `metadata.json` holds its SHA-256 and the provenance and evidence.
* Loading verifies: files exist → checksum → schema version → preprocessing version → labels →
  array shapes → finite weights. Any failure gives `/ready` → 503 instead of a crash.
* Release building asserts that the exported model's scores equal scikit-learn's on all 4,125
  development messages (max difference 6e-15). Unit tests check the featurizer against
  scikit-learn's vectorizers and every classifier family's export.

## Consequences

| | v1 | v2 |
|---|---|---|
| Vercel function bundle (local `vercel build`) | 207.15 MiB | **20.05 MiB** |
| Runtime dependencies | FastAPI + scikit-learn, SciPy, NumPy, NLTK (+ data), joblib | FastAPI/Pydantic only |
| Model file | 1.4 MB pickles (code execution on load) | 3.08 MB JSON (data only), checksummed |
| Model load | NLTK + joblib | 0.05–0.19 s |
| Prediction latency | n/a | 0.17 ms p50 (typical SMS), ~5 ms at 5,000 characters |

Trade-offs: we own ~150 lines of featurization code (mitigated by reference-parity tests), and
non-linear models would need a different artifact format (an explicit future decision).
