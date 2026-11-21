# Model card: SMS spam detector v2.0.0

| | |
|---|---|
| Artifact | `models/spam-v2/model.json`, SHA-256 `5f79aee6ba36e0f4e678010a1f5be95213a934cd29eb85bdb72d9eced526f34a` |
| Type | Linear SVM (class-balanced, C=1) on TF-IDF word 1–2-grams + character 2–5-grams (70,091 features), Platt-calibrated |
| Output | `spam` / `ham` / `inconclusive`, calibrated P(spam), abstention reason |
| Decision rule | spam if calibrated P(spam) ≥ 0.75 (false positive judged 3× as costly as a miss) |
| Training data | 4,125 deduplicated English SMS (513 spam) from the SMS Spam Collection (development split) |
| Code | `spam_detector/` at the commit recorded in `models/spam-v2/metadata.json` |
| Reproducible | Yes: `python -m spam_detector reproduce` rebuilds `model.json` byte-for-byte |

## Intended use

* Screening **English SMS** for patterns associated with unsolicited commercial or prize/premium-rate
  spam, as a **decision aid** (flag, sort, warn), with a human able to see the message.
* Educational and portfolio demonstration of a reproducible ML system.

## Out of scope / not validated

* Deciding whether a message is **safe**, legitimate, or non-malicious. A "ham" result is not a
  safety verdict.
* Detection of **phishing/smishing, fraud, malware links**, or social engineering in general. The
  training data predates most modern smishing. A handwritten bank-verification lure scores 0.45 and
  is not flagged.
* **E-mail**, chat apps, social media posts, or long documents (inputs are capped at 5,000
  characters; the training messages have a median length of 61 characters).
* **Non-English** text. Non-Latin scripts are refused (`unsupported_script`), but other Latin-script
  languages are classified as if they were English and results are unreliable.
* Automated blocking or deletion without human review.

## Performance

Under the leakage-controlled protocol in [EVALUATION.md](EVALUATION.md):

| | Spam precision | Spam recall | Spam F1 | FPR | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| CV, development (15 folds, mean ± std) | 0.988 ± 0.009 | 0.913 ± 0.023 | 0.949 ± 0.014 | 0.0016 | 0.982 | 0.0091 |
| Locked test (1,031 msgs, 128 spam) | 1.000 [1.000, 1.000] | 0.961 [0.926, 0.992] | 0.980 [0.961, 0.996] | 0.000 | 0.994 | 0.0036 |

Compared with the v1 recipe retrained on the same split, v2 misses 5 instead of 17 test spam
messages and raises 0 instead of 4 false alarms.

Runtime (standard library only): model load 0.05–0.19 s, 0.17 ms per typical SMS, ~5 ms at 5,000
characters, ~12 MiB Python heap for the model (63 MiB peak process RSS on Vercel, including the runtime).

## Data

The SMS Spam Collection (Almeida & Gómez Hidalgo, 2011; UCI, CC BY 4.0), in the Kaggle CSV
representation. Messages come mostly from UK SMS spam reports (Grumbletext), the NUS SMS Corpus
(Singapore students), and a UK PhD thesis corpus. Repairs, deduplication and grouping are described
in [DATASET.md](DATASET.md).

Known biases:

* **Era:** about 2004–2011. Spam types are premium-rate prizes, ringtones, and dating lines.
  Today's dominant lures (parcel delivery, bank or tax verification, crypto) are under-represented.
* **Dialect:** a large share of ham is Singaporean English with slang ("lor", "lar", "ü"). Ham in
  other dialects is less represented, which may affect the false-positive rate for other users.
* **Geography:** UK currency (£), UK premium number formats (09xx, 08xx), and short codes.

## Ethical and safety considerations

* **False reassurance is the main risk.** Users may read "ham" as "safe". The API and UI therefore
  say "likely spam / likely ham", show no safety badge, and abstain on out-of-distribution input.
* **Asymmetric costs are a product assumption** (3:1). Other deployments should set their own ratio
  in `configs/pipeline.toml`, retrain, and re-evaluate.
* **Privacy:** the API never logs message text, and validation errors do not echo input. The training
  corpus contains phone numbers and personal messages, so it is not redistributed in this repository.

## Caveats

* The locked test set is internal (same corpus and period), so it is not an independent evaluation.
* Probabilities are calibrated on this corpus. Under distribution shift they are approximate.
* Adversaries can evade a bag-of-n-grams model (for example by paraphrasing or using images and
  links). This model is not hardened against adaptive attackers.

## Maintenance

To retrain or adjust it, change `configs/pipeline.toml`, then run `experiment` → `release` (new
version directory) → `evaluate` once → `scripts/attach_evaluation_summary.py` →
`scripts/make_golden.py`. Review the golden-file diff and the reports in the pull request.
