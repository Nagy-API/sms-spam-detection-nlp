# ADR 0002: Class-balanced linear SVM on word + character n-grams, Platt-calibrated, cost threshold 0.75

**Status:** accepted · **Date:** 2026-11-21

## Context

Eleven candidates were compared under [ADR 0001](0001-evaluation-protocol.md): the faithful v1
recipe, Multinomial Naive Bayes, logistic regression, and linear SVMs, with word, character, and
combined features, two regularization strengths, and class weighting
(`reports/experiments/cv_results.md`).

The product question is not "which model has the highest accuracy" but "how often does the
user pay for a mistake". Flagging a legitimate message (a bank code, a family message) is judged
**three times** as costly as letting one spam through. That ratio is a product assumption,
configurable in `configs/pipeline.toml`.

## Decision

* **Selection metric:** expected cost per message at the deployment operating point (calibrated
  probability ≥ 0.75), mean over 15 CV folds.
* **Selection rule:** pre-registered *one-standard-error* rule. Every exportable candidate within
  one SE of the best mean is "tied", and the cheapest to serve (fewest features) wins.
* **Result:** `svm_word12_char25_balanced` (best mean) and `svm_word12_char25` were tied. They have
  identical feature counts, so the better mean won.

| Candidate (CV, calibrated @ 0.75) | Cost / msg | Spam precision | Spam recall | Spam F1 | PR-AUC |
|---|---|---|---|---|---|
| **svm_word12_char25_balanced** | **0.0149** | 0.988 | 0.913 | 0.949 | 0.982 |
| lr_word12_char25 | 0.0162 | 0.988 | 0.903 | 0.943 | 0.982 |
| nb_word12_counts (v2 text, NB) | 0.0179 | 0.996 | 0.866 | 0.926 | 0.961 |
| legacy_nltk_nb (v1 recipe) | 0.0312 | 0.995 | 0.761 | 0.861 | 0.918 |

## Why this model

* **Character n-grams** (2–5, inside word boundaries) catch obfuscation and spelling variants
  ("FR33", "txt", "£1000") that word features miss. Combined with word 1–2 grams they gave the
  best ranking quality (PR-AUC) of every family.
* **Linear SVM** beat logistic regression by a small margin at the operating point. With Platt
  scaling its margins become well-calibrated probabilities (CV Brier 0.0091, ECE 0.007), so we
  lose nothing by not using a natively probabilistic model.
* **Class weighting** is not needed for ranking. It shifts the raw decision boundary, which Platt
  scaling plus the cost threshold then set explicitly. Balanced and unbalanced variants tie.
* **Naive Bayes stays the honest runner-up for simplicity:** half the features, highest precision,
  but it misses ~5 more spam per 100.

## Alternatives rejected

* **Transformers (DistilBERT, MiniLM embeddings).** See [ADR 0006](0006-no-transformer.md).
* **Gradient boosting / random forests on sparse n-grams.** Not evaluated. They are not linear, so
  they cannot use the pickle-free artifact ([ADR 0004](0004-pickle-free-artifact.md)), they are much
  larger, and tree ensembles are generally a weak fit for very high-dimensional sparse text
  features. Given the small remaining headroom, they were not worth the deployment cost.
* **Threshold tuned to maximize test or OOF F1.** F1 ignores the asymmetric cost and an empirically
  tuned threshold overfits. With calibrated probabilities the Bayes-optimal threshold follows
  directly from the cost ratio: `C_FP / (C_FP + C_FN) = 0.75`.
