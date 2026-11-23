# Locked test-set evaluation

Artifact `models/spam-v2` (v2.0.0, sha256 `5f79aee6ba36…`), evaluated 2026-11-21T20:10:27+00:00.
Test set: 1031 messages, 128 spam, 970 near-duplicate groups (none shared with development).

> Internal held-out evaluation. The test messages come from the same corpus, collection period and sources as the training data, and the original v1 holdout (which overlaps this test set) was seen during the v1 project. This is not an independent external evaluation.

Brackets: 95% percentile bootstrap intervals (2,000 resamples).

| System | Accuracy | Spam precision | Spam recall | Spam F1 | Macro F1 | Balanced acc. | PR-AUC | FP | FN | Confusion |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 selected (threshold 0.75) | 0.9952 [0.990, 0.999] | 1.0000 [1.000, 1.000] | 0.9609 [0.926, 0.992] | 0.9801 [0.961, 0.996] | 0.9887 | 0.9805 | 0.9935 [0.980, 1.000] | 0 | 5 | TN=903 FP=0 FN=5 TP=123 |
| v1 recipe (retrained, same split) | 0.9796 [0.971, 0.987] | 0.9652 [0.929, 0.992] | 0.8672 [0.807, 0.924] | 0.9136 [0.874, 0.948] | 0.9510 | 0.9314 | 0.9591 [0.930, 0.983] | 4 | 17 | TN=899 FP=4 FN=17 TP=111 |

Calibration of the selected model: Brier 0.0036, log loss 0.0212, ECE 0.0108. Expected cost per message 0.0048.
Coverage: 1031 predicted, 0 inconclusive ({}); inconclusive answers count as not-spam above.
