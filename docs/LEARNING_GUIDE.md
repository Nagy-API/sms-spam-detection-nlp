# Learning guide: how this system works, and why

This guide walks through the project in the order data flows through it. Each section names the
file to read alongside it. Suggested study path:

1. Read sections 1–2 for the big picture, then run the commands in section 12.
2. Read `spam_detector/text.py` → `features.py` → `inference.py` (runtime, ~475 lines total).
3. Read `training/dedup.py` → `splits.py` → `candidates.py` → `experiment.py` → `release.py`.
4. Read the ADRs in `docs/adr/` and the [evaluation report](EVALUATION.md).
5. Answer the interview questions in section 14 out loud *before* reading the model answers.

---

## 1. System architecture

Two worlds share two modules:

* **Offline** (laptop or CI, `requirements-train.txt`): load and repair data → deduplicate and
  group → split → cross-validate candidates → build the release artifact → evaluate once on the
  locked test set.
* **Online** (Vercel function, `requirements.txt`): load and verify the artifact → for each request:
  abstention checks → normalize → featurize → linear score → calibrate → threshold.
* **Shared:** `text.py` (normalization) and `features.py` (n-grams and TF-IDF), pure standard
  library.

Diagram and module table: [ARCHITECTURE.md](ARCHITECTURE.md).

**Why this shape?** The most common production ML bug is *training/serving skew*: the model was
trained on features computed one way and served features computed another way. v1 had it (the
notebook and the API lemmatized differently). Sharing the exact code makes skew impossible rather
than merely unlikely.

## 2. Data-to-prediction pipeline, end to end

```
spam.csv bytes
 └─ repair_encoding (mac_roman→cp1252→utf-8)  ─ rejoin split columns ─ html.unescape       data.py
     └─ canonical_key dedupe ─ template/Jaccard groups (union-find)                       dedup.py
         └─ StratifiedGroupKFold: development 80% │ locked test 20%                        splits.py
             └─ repeated group CV on development: fit featurizer + classifier per fold     experiment.py
                 └─ select by expected cost (one-SE rule)
                     └─ OOF scores → Platt (a,b); refit on all development; export JSON    release.py
                         └─ model.json + metadata.json (sha256)                             artifact.py
                             └─ API: verify → abstain? → normalize → n-grams → TF-IDF
                                     → w·x+b → sigmoid(a·s+b') → ≥0.75 ? spam : ham         inference.py
```

## 3. Preprocessing techniques (`spam_detector/text.py`)

| Technique | What it does | Why |
|---|---|---|
| **Unicode NFKC** | Maps compatibility characters to canonical ones: full-width `ＦＲＥＥ` → `FREE`, ligatures, superscripts | Phones and keyboards produce many encodings of "the same" text, and spammers use look-alikes |
| **Punctuation folding** | `’ ‘ “ ” – — …` → ASCII | A curly apostrophe typed on a phone must match the straight one in training data |
| **Lowercasing** | `URGENT` → `urgent` | Halves the vocabulary. The case signal is weak once `!`, numbers and URLs are kept. (Not separately ablated, so it is a reasonable default rather than a proven win.) |
| **Placeholder masking** | e-mails → `zzemail`, URLs → `zzurl`, ≥7 digits → `zzphone`, 4–6 → `zzcode`, 1–3 → `zznum` | Keeps *what kind* of thing was there (strong spam signal) while discarding the exact value (campaign-specific and noisy). Generalizes to unseen numbers |
| **Redaction-marker mapping** | `<#>`, `<DECIMAL>`, `<TIME>` → `zznum` and similar | These markers appear only in ham (an anonymization artifact). Left alone, they teach a shortcut that never exists in real SMS |
| **Whitespace collapse** | runs of spaces/tabs/newlines → one space | Formatting is not content |

Deliberately **not** done, unlike v1: stop-word removal ("you", "now", "to", "your" are
spam-indicative in SMS), stemming/lemmatization (character n-grams already capture shared word
stems), and deleting digits and punctuation (that deleted the best features).

`PREPROCESSING_VERSION` is written into every artifact. Change the normalizer and you must bump
it, and old artifacts will refuse to load with new code. That is a deliberate compatibility gate.

## 4. Feature extraction (`spam_detector/features.py`)

**Bag of n-grams.** A message becomes a very long, mostly-zero vector with one column per n-gram
seen in training (70,091 columns), where most messages activate ~100–200 columns.

Two blocks are concatenated:

* **Word 1–2-grams**, tokens `[^\W_]+` plus `£ $ € ! ?`: `"call zzphone now"` →
  `call, zzphone, now, call zzphone, zzphone now`. Bigrams capture short phrases ("free entry",
  "call zzphone").
* **Character 2–5-grams within word boundaries** (`char_wb`): `"free"` → `" f", "fr", "re", "ee",
  "e ", " fr", "fre", …, " free "`. They give robustness to misspellings and obfuscation (`fr33`
  shares `" fr"`), word stems, and words never seen whole. `min_df=2` drops character n-grams seen
  in only one training message, which are noise and bloat.

**TF-IDF weighting**, per block:

* `tf = 1 + ln(count)` (*sublinear*): the 5th "win" adds less than the 1st.
* `idf = ln((1 + N) / (1 + df)) + 1`: rare n-grams weigh more than ubiquitous ones. It is computed on
  **training documents only**.
* **L2 normalization** per block, so a long message does not get a bigger score just for being long.

Out-of-vocabulary n-grams are ignored at inference. The model can only use what it saw.

**Parity.** `tests/unit/test_features.py` checks that this code produces exactly the same
vocabulary and matrix values as scikit-learn's `TfidfVectorizer`/`CountVectorizer` (tolerance
1e-12). That is how we justify owning ~150 lines instead of importing a 100 MB library at runtime.

## 5. The selected algorithm: a linear SVM, calibrated

**Linear model.** `score(x) = w · x + b`, a weighted sum of the message's n-gram weights. Positive
weights push towards spam ("zzphone", "claim", "£"), negative towards ham ("lor", "at zznum",
"hey"). `python -m spam_detector explain "…"` prints the largest terms `w_j · x_j`. For a linear
model this decomposition is exact.

**SVM training objective.** A linear SVM (`LinearSVC`, liblinear) minimizes

```
½‖w‖² + C · Σ_i weight(y_i) · max(0, 1 − y_i (w·x_i + b))²      (squared hinge loss)
```

It looks for the hyperplane that separates spam from ham with the largest margin, paying a penalty
for points inside the margin or on the wrong side. `C=1` trades margin width against training
errors (smaller C means more regularization). `class_weight="balanced"` up-weights errors on the
rarer class (spam is 12%) by `n / (2 · n_class)`.

**Platt scaling.** SVM scores are margins, not probabilities. We fit
`P(spam | s) = 1 / (1 + exp(−(a·s + b)))`, a one-feature logistic regression, on **out-of-fold**
scores (scores the model produced for messages it was *not* trained on). In-sample scores are
over-confident, which is why out-of-fold scores are required. Result: `a ≈ 6.2, b ≈ 0.65`.
One subtlety to be able to explain: the out-of-fold scores come from models trained on 80% of the
development data, but the calibrator is applied to the final model trained on 100%, whose margins
can be slightly larger. This is the standard "cross-validated Platt" compromise. We checked it did
not hurt on held-out data (locked-test ECE 0.011, Brier 0.0036). The alternative,
`CalibratedClassifierCV`, would ship an ensemble of 5 models (5× the artifact).

**Decision.** With calibrated probabilities, the threshold that minimizes expected cost is
`p* = C_FP / (C_FP + C_FN)`. We set a false positive (a real message flagged) at 3× the cost of a
miss, so `p* = 0.75`. Derivation: flag when the expected cost of flagging,
`(1 − p)·C_FP`, is below the expected cost of not flagging, `p·C_FN`.

## 6. Why this algorithm

Full comparison in [EVALUATION.md](EVALUATION.md#3-model-selection-development-split-only). In short:

* Linear models are the right family for **sparse, high-dimensional text with ~4k examples**: fast
  to train (< 1 s), hard to overfit with regularization, interpretable, and exportable as a few
  arrays.
* Among linear models, the SVM had the lowest expected cost at the operating point. Logistic
  regression was close; Naive Bayes was more precise but missed ~5 more spam per 100.
* Word + character features beat either alone.
* The decision followed a **pre-registered rule** (one-standard-error, then fewest features), so
  we did not pick a favourite after looking at the numbers.
* Transformers were not run: small measurable headroom, a 10× bigger deployment, and the real
  weakness is dated data ([ADR 0006](adr/0006-no-transformer.md)).

## 7. How the model was evaluated

* **Metrics that respect imbalance:** spam precision ("when it says spam, how often is it right"),
  spam recall ("what fraction of spam is caught"), F1, balanced accuracy, PR-AUC (ranking quality,
  threshold-free, imbalance-aware), false-positive rate, and **expected cost per message** (the
  product objective). Accuracy is reported but not used for decisions: always answering "ham"
  scores 87.6%.
* **Calibration:** Brier score (mean squared error of probabilities), log loss, and expected
  calibration error (ECE: the average gap between predicted probability and observed spam rate
  across 10 bins).
* **Uncertainty:** mean ± std over 15 CV folds, and 95% bootstrap intervals on the test set.
* **Protocol:** repeated stratified *group* K-fold on development for every decision, and the
  locked test set once at the end. That test run happened twice for a documented reason; both runs
  are logged.

## 8. How leakage was prevented

| Leak | Where it could happen | Prevention | Verified by |
|---|---|---|---|
| Duplicate messages across splits | 416 exact or case/whitespace duplicates | Canonical-key dedup before splitting | `test_pipeline_with_dataset.py::test_no_leakage_between_development_and_test` |
| Campaign variants across splits | 161 near-duplicate groups | Template key + Jaccard ≥ 0.5 groups, `StratifiedGroupKFold` | `assert_no_group_leakage` runs inside every split |
| Vocabulary/IDF from validation data | Fitting the vectorizer before splitting | `FeatureSpace.fit` is called inside each CV fold on training indices only | `test_vocabulary_comes_only_from_training_texts` |
| Calibration on its own evaluation data | Platt fitted on the fold it scores | Cross-fitting: fold *k*'s calibrator uses other folds only | `experiment.py::cross_validate_candidate` |
| Model selection on the test set | Comparing candidates on the holdout (v1) | Selection only on development CV. Test scored after the artifact was committed (`75f2d0a`) | `test_access_log.jsonl`, Git history |
| Dataset shortcut features | `<#>` markers only in ham | Mapped to number placeholders | `test_text.py` |

The v1 audit shows why this matters: 47% of v1's holdout spam had a near-duplicate in training,
and v1's recall falls from 0.92 to 0.84 on the leak-free part.

## 9. Training vs inference

| | Training (`training/`) | Inference (`inference.py`) |
|---|---|---|
| Libraries | scikit-learn, SciPy, NumPy | standard library |
| Featurizer | `FeatureSpace.fit` learns vocabulary + IDF, then `transform` | `transform` only, with frozen vocabulary + IDF |
| Data structure | SciPy CSR matrix of all messages | `dict[column → value]` for one message |
| Classifier | `LinearSVC.fit` (optimization) | `sum(w[c] * v) + b` (one dot product) |
| Calibration | `fit_platt` on OOF scores | `sigmoid(a·s + b)` |
| Output | artifact + reports | label, probability, reason |
| Speed | ~1 s per fit, ~2.5 min full experiment | ~0.17 ms per message |

Same code path: `normalize` and the n-gram analyzers. The release step **asserts** that the
exported runtime and scikit-learn agree on every development message (max difference 6e-15).

## 10. Artifacts: generation and loading (`spam_detector/artifact.py`)

* `release` fits the model, converts NumPy floats to Python floats, and writes `model.json` with
  sorted keys and no timestamps. That makes it deterministic, so `reproduce` can rebuild it and
  compare SHA-256. Seeding `LinearSVC` (`random_state=0`) was required: liblinear shuffles samples,
  and an unseeded run produced slightly different weights. This was caught and fixed before release.
* `metadata.json` stores `model_sha256`, dataset SHA-256, split fingerprint, Git commit, library
  versions, the parity result, and the evaluation summary.
* `.gitattributes` marks `model.json` as binary so Git never rewrites line endings, which would
  break the checksum on Windows checkouts.
* `load_artifact` verifies existence → checksum → JSON → schema version → preprocessing version →
  labels → array lengths → finite weights. Every failure is an `ArtifactError` with an actionable
  message.
* `save_artifact` refuses to overwrite. A new model gets a new version directory.
* v1's pickles are preserved, unmodified, in `models/legacy-v1/` for the audit.

## 11. FastAPI integration (`app/main.py`)

* **App factory** `create_app(model_dir)` makes tests able to build apps with broken or tampered
  artifacts. Module-level `app = create_app()` is what Vercel imports.
* **Lifespan** loads the model once per process (per serverless instance). A failure is stored, not
  raised: `/health` stays 200 (liveness), `/ready` and `/predict` return 503 (readiness).
* **Pydantic models** define the contract (`PredictRequest`, `PredictResponse`). They generate the
  OpenAPI schema, and the API test asserts the schema.
* **Middleware**, outermost first: a pure-ASGI body-size limit (rejects > 64 KiB, including chunked
  bodies, before JSON parsing), then request context (request ID, timing, security headers, CSP,
  one JSON log line).
* **Custom 422 handler** strips the `input` field that FastAPI would echo back (privacy).
* `predict` is a plain `def`, so FastAPI runs it in a thread pool. Inference is CPU-bound,
  microseconds to milliseconds, and bounded by the 5,000-character limit, so no extra timeout is
  needed. This is documented instead of adding unused machinery.

## 12. Reproducing the experiments

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
# put spam.csv in data/ (docs/DATASET.md), then:
python scripts/prepare_nltk.py           # only for the v1 baseline
python -m spam_detector data
python -m spam_detector audit-legacy
python -m spam_detector experiment       # writes reports/experiments/cv_results.{json,md}
python -m spam_detector reproduce        # rebuilds models/spam-v2 in a temp dir, compares sha256
python -m pytest                         # all tests, including dataset-gated ones
python -m spam_detector explain "WINNER!! call 09061701461 to claim your £900 prize"
```

To try your own candidate, add a `[[candidates]]` block to `configs/pipeline.toml` and run
`python -m spam_detector experiment --only your_name lr_word12_char25` (writes `*_partial` reports).

## 13. Limitations and failure cases

* **Modern smishing is missed.** "Your account is suspended. Verify now at <url>" scores 0.45
  (ham). The corpus predates these lures.
* **Conversational spam without spam vocabulary** ("hello darling, let's chat…") is missed. A
  bag-of-n-grams model cannot understand intent.
* **Near-threshold spam:** 3 of the 5 test misses scored 0.62–0.73. The 3:1 cost ratio trades these
  for zero false alarms on the test set.
* **A message that is only a URL** scores as spam (URLs are almost always spam in this corpus).
* **Other Latin-script languages** are classified as English (unreliable). Non-Latin scripts abstain.
* **Adversarial robustness:** paraphrasing, image-only spam, and link shorteners are out of reach.
* **Distribution shift:** calibration holds on this corpus only.
* **Evaluation:** the test set is internal and small (128 spam), so intervals are wide.
* **Calibration transfer:** Platt parameters are fitted on 80%-data models and applied to the
  100%-data model (see section 5). This is fine on this distribution, but it is not guaranteed.

## 14. Interview questions about this implementation

<details><summary><b>Why did you not trust the original 98% accuracy?</b></summary>

Accuracy on a 13%-spam dataset is dominated by ham. The holdout also selected the model, and near
half of its spam had near-duplicates in training. I reproduced the original split and showed spam
recall fell from 0.92 to 0.84 on the leak-free part. I report spam precision/recall/F1, PR-AUC, and
expected cost instead, and evaluate with grouped splits.
</details>

<details><summary><b>What is group K-fold and why use it here?</b></summary>

It keeps all members of a group in the same fold. Spam campaigns reuse templates, so random
splitting puts siblings on both sides and measures memorization. I built groups with template
keys and character-shingle Jaccard similarity, merged them with union-find, and used
`StratifiedGroupKFold` to also keep the spam ratio balanced.
</details>

<details><summary><b>How did you pick the Jaccard threshold, and what if it is wrong?</b></summary>

By inspecting linked pairs in similarity bands: even 0.5–0.6 links variants of one campaign, and no
group mixed labels. A threshold that is too low over-groups (harmless: slightly coarser splits), and
one that is too high leaks. I erred low. It is a config value and a documented judgment call.
</details>

<details><summary><b>Why have a development set and a test set if you do cross-validation?</b></summary>

CV is used for *decisions* (features, model, C, calibration, threshold). Any set used for decisions
becomes optimistic. The locked test set is touched only after the artifact is frozen, giving one
unbiased final estimate. I log every access, and it was scored twice for a documented reason.
</details>

<details><summary><b>Why calibrate, and how do you know calibration worked?</b></summary>

The threshold is derived from costs, which requires probabilities to mean what they say. SVM margins
are not probabilities, so I applied Platt scaling on out-of-fold scores. I check with Brier score
and ECE under cross-fitting (CV ECE ≈ 0.007) and on the test set (ECE 0.011).
</details>

<details><summary><b>Where does the 0.75 threshold come from?</b></summary>

Bayes decision theory: flag when `p·C_FN > (1−p)·C_FP`, so `p > C_FP/(C_FP+C_FN)`. With a false
alarm judged 3× worse than a miss, that is 0.75. The ratio is a product assumption in config. I did
not tune the threshold on test data.
</details>

<details><summary><b>Why not use the model with the best mean CV score?</b></summary>

With 11 candidates, the best mean partly reflects noise. I used the one-standard-error rule: all
candidates within one SE of the best are tied, and I pick the cheapest to serve. Here the two SVM
variants tied and had equal size, so the better mean won. I also note that repeated-CV SEs are
optimistic because folds overlap.
</details>

<details><summary><b>How can a linear SVM run without scikit-learn in production?</b></summary>

Prediction is a dot product. I export vocabulary, IDF, weights, intercept, and Platt parameters to
JSON, and reimplemented the featurizer in pure Python, which training also uses. Parity tests
against scikit-learn's vectorizers and `decision_function` (≤ 1e-9), plus an assertion during
release, guarantee equivalence. The bundle went from 207 MiB to 20 MiB, and there is no pickle to
execute.
</details>

<details><summary><b>How does Naive Bayes fit in a linear artifact?</b></summary>

`log P(spam|x) − log P(ham|x) = Σ_j x_j (log θ_spam,j − log θ_ham,j) + log π_spam − log π_ham`. That is
linear in the count vector, so the coefficients are the log-probability differences and the
intercept is the log prior ratio. A unit test checks it against `predict_joint_log_proba`.
</details>

<details><summary><b>Why character n-grams?</b></summary>

They are robust to misspellings, obfuscation ("fr33"), slang, and inflection, and give partial
credit for unseen words. Combined with word n-grams they had the best PR-AUC. `char_wb` keeps
n-grams inside word boundaries, so they do not create spurious cross-word fragments.
</details>

<details><summary><b>What happens with an empty, Arabic, or gibberish message?</b></summary>

Blank text gives 422. No letters gives `inconclusive/no_letters`. Mostly non-Latin letters give
`unsupported_script`. No known word n-gram gives `no_recognized_features`. I added the last rule
after golden cases showed `qzxwvj` getting a confident "ham" from character overlap. On
development folds it abstains on only 0.1% of messages.
</details>

<details><summary><b>How do you make training reproducible?</b></summary>

I pin dependencies, use fixed seeds (split, CV, and liblinear's `random_state`), sort the
vocabulary, write deterministic JSON, fingerprint the dataset and split, and record config hash and
Git commit. `python -m spam_detector reproduce` rebuilds `model.json` and compares SHA-256. I found
and fixed an unseeded `LinearSVC` this way.
</details>

<details><summary><b>How would you detect the model degrading in production?</b></summary>

Logs carry label, reason, and length per request (never text), so I can monitor the spam rate, the
abstention rate, and the length distribution for drift. True performance needs labels: user "not
spam / is spam" feedback, or periodic labeled samples scored with `evaluate-csv`. Calibration
drift shows as the predicted spam rate diverging from labeled samples.
</details>

<details><summary><b>Biggest weakness and what you would do next?</b></summary>

Data. The corpus is about 15 years old and lacks modern smishing (a bank-verification lure scores
0.45). Next steps: obtain a licensed recent SMS/smishing set, run external evaluation, retrain the
linear models on combined data, and only then consider a small transformer served separately.
</details>

<details><summary><b>Why not log the SMS text for debugging?</b></summary>

It is personal data (codes, addresses, conversations). The request ID plus label, reason, and
length is enough to debug the service. Content debugging uses `explain` locally on a reproduced
input. Validation errors also stop echoing input.
</details>
