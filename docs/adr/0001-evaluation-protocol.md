# ADR 0001: Group-aware evaluation with a locked test set

**Status:** accepted · **Date:** 2026-11-21

## Context

The v1 project dropped raw duplicates, made one random stratified 80/20 split, and chose between
TF-IDF, counts, unigrams/bigrams, and oversampling by comparing scores **on that same holdout**.
Three problems followed:

1. **Near-duplicate leakage.** Spam is sent in campaigns that reuse a template with different
   numbers or prizes. Reproducing the v1 split shows that 85 of 1,034 holdout messages, and **61 of
   131 holdout spam messages (47%)**, had a near-duplicate in the v1 training set
   (`reports/experiments/legacy_protocol_audit.json`).
2. **Selection on the test set.** The holdout chose the model, so its score is optimistically
   biased as an estimate of performance on new data.
3. **Single split, no uncertainty.** With 131 spam messages, one misclassification moves spam
   recall by 0.8 points. A single number hides that.

## Decision

* Deduplicate after canonicalization (NFKC, case-fold, whitespace), then group near-duplicates:
  identical *template* (digits/URLs masked) **or** character-5-gram Jaccard ≥ 0.5, merged with
  union-find. The 0.5 threshold was chosen by inspecting linked pairs (even the 0.5–0.6 band joins
  variants of one campaign) and no group mixes labels.
* Split once with `StratifiedGroupKFold` into development (80%) and a **locked test** set (20%).
  The split is identified by a committed SHA-256 fingerprint.
* All model selection, calibration, and threshold choice uses **5-fold × 3-repeat stratified group
  CV on development only**.
* The test set is scored only by `python -m spam_detector evaluate`, which requires an explicit
  flag and appends to `reports/evaluation/test_access_log.jsonl`. The artifact and selection
  report were committed (`75f2d0a`) *before* the first test run.
* Report 95% bootstrap confidence intervals on the test set.

## Consequences

* v2 and the v1 recipe are compared under identical conditions. The v1 *artifact* is not scored on
  the new test set because it was trained on messages that are now test messages.
* The test set is internal: same corpus, period, and sources, and it overlaps the v1 holdout that
  was seen during v1. Its numbers are **not** independent external validation (see
  [EVALUATION.md](../EVALUATION.md#what-these-numbers-do-and-do-not-mean)).
* Repeated-CV standard errors are optimistic because folds share data. They are used as a
  tie-break heuristic, not as a significance test.
