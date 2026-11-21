# ADR 0006: No transformer model (for now)

**Status:** accepted · **Date:** 2026-11-21 · **Revisit when:** a newer, independent SMS/smishing
evaluation set is available

## Context

Fine-tuned transformers (BERT-family) are the default for many text-classification tasks. On this
corpus, the dataset's original paper (Almeida, Gómez Hidalgo & Yamakami, 2011) found a linear SVM to
be the strongest of the methods it compared, and our linear model already reaches CV spam F1 0.949
and locked-test F1 0.980 under a stricter, leakage-controlled protocol.

## Decision

Do not train or serve a transformer in this phase. **No transformer experiment was run**, so this
ADR makes no claim about its measured accuracy on our split.

Reasons:

* **Headroom is small and unmeasurable here.** The locked test set has 128 spam messages. The 95%
  interval on spam recall is already [0.926, 0.992], so a difference of one or two messages
  could not be distinguished from noise.
* **Deployment cost is large.** PyTorch plus a small model is hundreds of MB, beyond Vercel's
  function size limit, and requires a different host (container or GPU/CPU inference service)
  with a cold start of seconds instead of tens of milliseconds.
* **The real weakness is data, not model capacity.** The corpus is from ~2011 and contains almost
  no modern smishing (parcel, bank-verification, crypto lures). A modern phishing example scores
  P(spam)=0.45 and is not flagged. A larger model trained on the same data would inherit this.

## What would change the decision

An independent, licensed, recent SMS/smishing dataset. With it: (1) measure the current model with
`python -m spam_detector evaluate-csv`, (2) if it degrades, try retraining linear models on the
combined data first, then (3) compare against a small transformer (for example ONNX-exported
MiniLM) served from a dedicated service, using the same group-aware protocol.
