# ADR 0005: When the model answers "inconclusive"

**Status:** accepted · **Date:** 2026-11-21

## Context

A classifier always produces *a* number. For inputs unlike anything in training, that number is
the class prior dressed up as an answer. v1 abstained only when no vocabulary feature matched.
With character n-grams, almost every Latin string matches something, so that rule alone would
never fire (handwritten golden cases showed `qzxwvj` → "ham", P(spam)=0.0016).

## Decision

The model abstains (`label: "inconclusive"`, `spam_probability: null`) with a machine-readable
`reason`:

| reason | rule | example |
|---|---|---|
| `no_letters` | the text has no Unicode letters | `12345`, `🙂🙂`, `!!!` |
| `unsupported_script` | < 50% of letters are Latin (after NFKC) | Arabic, Cyrillic, CJK |
| `no_recognized_features` | no **word** 1–2-gram is in the vocabulary | `qzxwvj`, transliterated non-English |

The rules are part of `model.json` (`abstention`), so they version with the model.

We did **not** add a probability band (for example 0.25–0.75 → inconclusive). Calibration is
good and the threshold already encodes the cost trade-off. On the test set only 11 of 1,031
messages fall in 0.25–0.75, so a band would mostly hide the model's honest uncertainty, which the
probability already conveys to API clients.

## Consequences and process note

* Measured with fold-local vocabularies on the development set, the word rule abstains on 4 of
  4,125 messages (0.1%, all ham, mostly non-English). On the locked test set it abstains on none.
* The word rule and a fix for full-width Latin detection were added **after** the first locked-test
  run (the trigger was handwritten golden cases, not test results). Model weights, calibration,
  and threshold are unchanged, and the test set was scored a second time with identical metrics.
  Both runs are in `reports/evaluation/test_access_log.jsonl`.
* Mostly-English messages with some other script are still classified.
* Non-English Latin-script text (French, Spanish) is **not** detected and is classified as if it
  were English. This is a documented limitation.
