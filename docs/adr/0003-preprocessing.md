# ADR 0003: One normalizer, signal-preserving, without NLTK

**Status:** accepted · **Date:** 2026-11-21

## Context

v1 preprocessing deleted URLs, then replaced *every* non-letter with a space, lowercased, removed
NLTK stop words, POS-tagged, and lemmatized. It lived in `app/main.py`, while the notebook had its
own copy without POS-aware lemmatization. The two had drifted apart: the notebook's reported
metrics came from different features than the API served.

Consequences: digits, phone numbers, `£`, `!`, and URLs (the strongest spam cues) never reached
the model. Inputs with no letters were silently classified as ham. Serving needed NLTK plus about
40 MB of NLTK data downloaded at build time.

## Decision

`spam_detector/text.py::normalize` is the **only** text transformation, used by training,
evaluation, and the API:

1. Unicode NFKC (full-width/compatibility forms → canonical), typographic quotes and dashes → ASCII.
2. Lowercase.
3. Mask volatile values into typed placeholder words: e-mail → `zzemail`, URL → `zzurl`,
   ≥ 7 digits → `zzphone`, 4–6 digits → `zzcode`, 1–3 digits → `zznum`. The model learns "contains
   a premium-rate number" instead of memorizing `09061701461`.
4. Map the corpus's anonymization markers (`<#>`, `<DECIMAL>`, `<TIME>`, `<URL>`, `<EMAIL>`, which
   appear **only in ham**) to the placeholder of what they replaced, removing a dataset shortcut.
5. Collapse whitespace.

No stop-word removal, stemming, or lemmatization: SMS are short, function words such as "you",
"now", "to" are discriminative, and character n-grams provide robustness to inflection and
misspelling.

`PREPROCESSING_VERSION` is stored in every artifact. The loader refuses an artifact whose version
differs from the code.

## Consequences

* The API needs no NLTK, which removes a build step and ~40 MB.
* Training/serving skew is impossible by construction, and parity tests check it.
* The v1 recipe is preserved verbatim in `spam_detector/training/baseline.py` for comparison only.
