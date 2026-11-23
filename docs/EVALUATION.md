# Evaluation

Every number on this page is produced by a command listed next to it and stored in `reports/`.
Configuration: [`configs/pipeline.toml`](../configs/pipeline.toml). Protocol rationale:
[ADR 0001](adr/0001-evaluation-protocol.md).

## 1. Data preparation (`python -m spam_detector data`)

Source: Kaggle *SMS Spam Collection Dataset* `spam.csv`, SHA-256 `440e6ea9…1dfc2`
([DATASET.md](DATASET.md)). Report: [`reports/data/dataset_report.json`](../reports/data/dataset_report.json).

| Step | Effect |
|---|---|
| Rows read | 5,572 (4,825 ham, 747 spam) |
| Split rows rejoined (text that spilled into unnamed columns) | 50 |
| Double-encoded byte runs repaired / left as-is | 495 / 137 |
| Rows with HTML entities unescaped | 309 |
| Exact duplicates removed (after NFKC, case-fold, whitespace) | 416 (v1 removed 403) |
| Duplicates with conflicting labels | 0 |
| Unique messages | 5,156 (4,515 ham, 641 spam) |
| Near-duplicate groups (template or char-5-gram Jaccard ≥ 0.5) | 4,848, of which 161 contain >1 message (max 15), 0 mixed-label |
| **Development split** | 4,125 messages, 513 spam (12.44%), 3,878 groups |
| **Locked test split** | 1,031 messages, 128 spam (12.42%), 970 groups, none shared |

Split fingerprint `221cdf6b…6077` lets anyone with the same CSV verify they reproduced the exact
split. Only an aggregate hash is published, never message text or per-message hashes.

## 2. How trustworthy were the v1 numbers? (`python -m spam_detector audit-legacy`)

The v1 split and the committed v1 pickles reproduce the v1 notebook figures exactly:

| v1 model on v1 holdout | Messages | Spam | Accuracy | Spam precision | Spam recall | Spam F1 | PR-AUC |
|---|---|---|---|---|---|---|---|
| All holdout messages | 1,034 | 131 | 0.985 | 0.960 | 0.916 | 0.938 | 0.961 |
| Without near-duplicates of training messages | 949 | 70 | 0.983 | 0.922 | **0.843** | **0.881** | 0.908 |

**61 of the 131 holdout spam messages (47%) had a near-duplicate in v1's training data.** All of
v1's misses were among the messages without one. The headline spam F1 of 0.94 measured campaign
recall more than generalization. Source:
[`reports/experiments/legacy_protocol_audit.json`](../reports/experiments/legacy_protocol_audit.json).

## 3. Model selection: development split only (`python -m spam_detector experiment`)

5-fold stratified **group** CV × 3 repeats = 15 folds. For each fold the featurizer's vocabulary
and IDF are fitted on the training part only. Two views:

* **Calibrated @ 0.75 (deployment procedure):** Platt scaling fitted by *cross-fitting* (the
  calibrator for fold *k* uses only the other folds' out-of-fold scores), then the cost threshold.
* **Default:** each classifier's own `score ≥ 0` rule.

Full tables: [`reports/experiments/cv_results.md`](../reports/experiments/cv_results.md). Mean ± std over folds:

| Candidate | Cost / msg ↓ | Spam precision | Spam recall | Spam F1 | FPR | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| **svm_word12_char25_balanced (selected)** | **0.0149 ± 0.0045** | 0.988 ± 0.009 | 0.913 ± 0.023 | 0.949 ± 0.014 | 0.0016 | 0.982 ± 0.007 | 0.0091 | 0.0069 |
| svm_word12_char25 | 0.0154 ± 0.0040 | 0.986 | 0.916 | 0.949 | 0.0018 | 0.983 | 0.0090 | 0.0062 |
| lr_word12_char25 | 0.0162 ± 0.0041 | 0.988 | 0.903 | 0.943 | 0.0016 | 0.982 | 0.0096 | 0.0067 |
| lr_word12_char36 | 0.0173 ± 0.0049 | 0.986 | 0.900 | 0.941 | 0.0018 | 0.981 | 0.0099 | 0.0064 |
| nb_word12_counts | 0.0179 ± 0.0052 | 0.996 | 0.866 | 0.926 | 0.0005 | 0.961 | 0.0122 | 0.0094 |
| lr_word12 | 0.0182 ± 0.0048 | 0.984 | 0.899 | 0.939 | 0.0021 | 0.976 | 0.0105 | 0.0084 |
| lr_char25 | 0.0190 ± 0.0052 | 0.984 | 0.892 | 0.935 | 0.0021 | 0.980 | 0.0103 | 0.0069 |
| lr_word12_char25_c1 | 0.0209 ± 0.0066 | 0.981 | 0.884 | 0.930 | 0.0025 | 0.978 | 0.0110 | 0.0073 |
| nb_word12_char25_counts | 0.0218 ± 0.0054 | 0.990 | 0.852 | 0.915 | 0.0013 | 0.958 | 0.0139 | 0.0101 |
| **legacy_nltk_nb (v1 recipe)** | 0.0312 ± 0.0077 | 0.995 | 0.761 ± 0.069 | 0.861 ± 0.044 | 0.0006 | 0.918 | 0.0234 | 0.0159 |

Findings:

* **Preprocessing mattered more than the classifier.** The same Naive Bayes on v2 text
  (`nb_word12_counts`) beats the v1 recipe (`legacy_nltk_nb`) on every ranking and cost metric.
  Keeping digits, `£`, `!` and URLs as typed placeholders recovers signal v1 deleted.
* **Character n-grams add ranking quality** (PR-AUC 0.976 → 0.982 for LR).
* **Calibration is good across the board** (ECE < 0.01 for the top candidates), which is what makes
  a principled cost threshold possible.
* **Stronger regularization hurts** (C=1 vs C=10 for LR).
* Selection used the pre-registered one-standard-error rule ([ADR 0002](adr/0002-model-selection.md)).

## 4. Final evaluation on the locked test set (`python -m spam_detector evaluate`)

Report: [`reports/evaluation/test_report.md`](../reports/evaluation/test_report.md). 95% bootstrap
intervals (2,000 resamples) in brackets. The v1 recipe was retrained on the same development split
and uses its own default decision rule (log-odds ≥ 0).

| System | Accuracy | Spam precision | Spam recall | Spam F1 | Macro F1 | Balanced acc. | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **v2** (threshold 0.75) | 0.995 [0.990, 0.999] | 1.000 [1.000, 1.000] | 0.961 [0.926, 0.992] | 0.980 [0.961, 0.996] | 0.989 | 0.981 | 0.994 [0.980, 1.000] | **0** | **5** |
| v1 recipe, retrained | 0.980 [0.971, 0.987] | 0.965 [0.929, 0.992] | 0.867 [0.807, 0.924] | 0.914 [0.874, 0.948] | 0.951 | 0.931 | 0.959 [0.930, 0.983] | 4 | 17 |

Confusion matrix v2: TN 903, FP 0, FN 5, TP 123. Calibration: Brier 0.0036, log loss 0.021, ECE 0.011.
Coverage: 1,031 / 1,031 predicted (no abstentions).

**Read the CV numbers as the planning estimate.** The test F1 (0.980) is above the CV mean
(0.949 ± 0.014). With 128 test spam messages, that gap is a handful of messages, and the test
interval is wide. Expect roughly 0.93–0.97 spam F1 on new data *from the same distribution*.
"Precision 1.000" means zero false positives among 903 test hams, not that false positives are
impossible: CV measured a false-positive rate of about 0.16%.

### Error analysis (described, not quoted, to avoid republishing corpus text)

The 5 missed spam messages:

* 3 scored P(spam) between 0.62 and 0.73, just under the 0.75 cost threshold. This is the
  deliberate price of protecting legitimate messages; at a 0.5 threshold they would be caught.
* 1 is a conversational "lonely, let's chat" lure with none of the usual spam vocabulary (P = 0.001).
* 1 reads like a personal complaint about a debt collector. Its spam label is arguably ambiguous.

Only 11 of 1,031 test messages fall between 0.25 and 0.75. The model is rarely uncertain on this
distribution.

### Test-set access log

`reports/evaluation/test_access_log.jsonl` has **two** entries, both before any other use of the
test results:

1. `eda8a83e…`: the artifact frozen in commit `75f2d0a` before any test access.
2. `5f79aee6…`: the same weights, calibration, and threshold, plus the word-level abstention rule
   and the full-width-Latin fix ([ADR 0005](adr/0005-abstention.md)). Those changes were motivated by
   handwritten golden cases, not by test results. The metrics were identical.

## 5. What these numbers do and do not mean

* They estimate performance on **English SMS resembling the 2011–2012 UK/Singapore corpus**.
* The test set is held out and leakage-controlled but **internal**: same corpus, same period, and
  overlapping the v1 holdout seen during v1. It is **not** an independent external evaluation.
  None was possible because no licensed, independent SMS dataset was available in this phase.
  `python -m spam_detector evaluate-csv your.csv` runs one as soon as such data exists.
* The corpus has almost no modern smishing. A handwritten bank-verification phishing text scores
  0.45 and is **not** flagged (`tests/golden/golden_predictions.json`).
* Probabilities are calibrated on this distribution. Under distribution shift they are estimates,
  not guarantees.

## 6. Reproduce everything

```powershell
python -m pip install -r requirements-dev.txt
python scripts/prepare_nltk.py              # only needed for the v1 baseline
python -m spam_detector data                # dataset report + split fingerprint
python -m spam_detector audit-legacy        # v1 protocol audit
python -m spam_detector experiment          # ~2.5 minutes on a laptop
python -m spam_detector reproduce           # rebuilds the release; checks model.json byte-for-byte
python -m pytest -m dataset                 # dataset-gated checks, incl. byte-for-byte reproduction
```

`release` refuses to overwrite `models/spam-v2`, and `evaluate` refuses to run without
`--i-understand-this-uses-the-locked-test-set` (and logs every run).
## TRACE V3 release candidate evidence

The V3 head remains experimental. The historical metrics in this document describe V2. The
[V3 development matrices](../reports/v3/development.json) describe source-confounded research
baselines, not the packaged composite or independent phishing accuracy. The packaged head
uses the published V2 word vocabulary, separate development calibration groups and an
experimental operating point. Local API and Chrome checks exercised actual predictions for
all four outcomes. These integration checks do not estimate generalization or false-positive rates.

The independent cohort requirements are recorded in [V3 development](V3_DEVELOPMENT.md).
They remain open. No frozen independent test was scored for this release candidate.
