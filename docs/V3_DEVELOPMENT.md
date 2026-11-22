# TRACE V3 development protocol

Run from the repository root with the training environment installed. The v2 Kaggle snapshot must already exist at data/spam.csv and match the v2 configuration hash.

    python -m spam_detector.training.v3 acquire
    python -m spam_detector.training.v3 benchmark
    python -m spam_detector.training.v3 export

Raw snapshots are stored under ignored data/trace-v3 and excluded from the deployment. The acquisition command verifies pinned SHA-256 hashes before installing a download and verifies cached files on reuse. Source identifiers and aggregate results can be published. Do not publish records, sender metadata, vocabulary derived from unreviewed text, or per-message fingerprints. Normalization masks recognized URLs, email addresses and numbers. It does not guarantee removal of names or all personal information.

The NCSU and IMC CSV snapshots were acquired on 2026-11-22. The [development report](../reports/v3/development.json) gives their source URLs, hashes, licenses, filtering counts, and confusion matrices. NCSU's pinned export has seven data fields under a six-field header. The parser reads the named message column and discards the extra field.

The experimental export fits a binary word TF-IDF logistic regression phishing head on grouped development data, then fits a Platt calibration on separate development groups. Its fixed 0.90 operating point is for integration testing. The proxy calibration cannot justify a real-world probability or threshold. The export writes the same checksummed JSON format as V2 under `data/trace-v3/candidate`. Keep this directory out of Git and deployment. Learned vocabulary can retain sensitive fragments.

To exercise the experiment locally, set `TRACE_EXPERIMENTAL_MODEL_DIR=data/trace-v3/candidate` before starting Uvicorn and open `/?model=v3`. The opt-in endpoint is `POST /predict/v3`. It accepts the same bounded JSON input and returns the V2 historical spam score where available, an experimental version and flag, and a separate `phishing` verdict when the verified head fires. The default endpoint and UI serve V2. A missing or invalid experimental artifact yields an error on the experimental endpoint while leaving V2 available. Do not set the experimental variable in production.

The benchmark uses the v2 historical development labels for ham and spam and treats the NCSU research positives as phishing proxies. A Latin-script filter is not English verification. IMC English reports outside the source's ordinary spam bucket provide overlap quarantine only. The IMC categories do not establish fraud intent on an individual record. Preserve the v2 locked historical partition and exclude any development cluster touching it or the IMC snapshot. Group all sources together before partitioning so identical or related templates do not cross boundaries. Compare word and character TF-IDF logistic regression and v2 on a deterministic grouped development fold. V2 has already seen the historical development corpus, so its historical development scores are descriptive and not a fair held-out comparison. No threshold or calibration decision is justified by this run.

66,687 of the 68,029 raw NCSU messages retain a URL placeholder after v2 normalization. The benchmark repeats the word model with that token removed and reports validation URL counts by class. This tests one possible shortcut. The class/source confound remains even if a model also scores well without the token.

Before locking a release evaluation, obtain and fingerprint an independent, rights-cleared source of real modern English benign messages alongside audited modern phishing labels. Require legitimate delivery, banking, verification and link-bearing notifications and count campaigns rather than treating variants as independent. Record source and collection dates, adjudicate ambiguous records, and quarantine overlap with development sources. Freeze the candidate, calibration, thresholds and split before scoring.

Preliminary release requirements to register with the final cohort: at least 200 independent modern phishing campaigns and 500 independent benign groups, with phishing recall at least 90% and benign false-positive rate at most 2%, counting any spam or phishing alert as a false positive. Count abstentions as missed positives for recall and report coverage separately. Report campaign-resampled 95% intervals and source and notification slices. Require historical spam recall and precision within two percentage points of v2 on its untouched locked split. Any adjustments to these requirements must be justified and recorded before evaluation. A missing cohort or failed gate prevents model promotion. The public API continues to use v2 until a candidate passes.

Source references and known limitations are in [the source audit](V3_SOURCE_AUDIT.md).
