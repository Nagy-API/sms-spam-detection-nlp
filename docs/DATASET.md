# Dataset: provenance, acquisition, and quality

## Provenance and license

* **Original corpus:** *SMS Spam Collection v.1*, Tiago A. Almeida and José María Gómez Hidalgo
  (2011), published by the UCI Machine Learning Repository under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation DOI:
  [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
* **Representation used here:** the Kaggle *SMS Spam Collection Dataset* `spam.csv` (columns
  `v1` = label, `v2` = text, plus three mostly empty unnamed columns), Latin-1/Mac Roman encoded.
* **Not in this repository.** The Kaggle listing does not state a redistribution license for its
  re-encoded file, and the messages contain real phone numbers and personal conversations. Only
  aggregate statistics and a split fingerprint are committed.

## Getting the file

1. Download `spam.csv` from the
   [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
2. Save it as `data/spam.csv`.
3. Verify the fingerprint. The pipeline also refuses a different file:

```powershell
(Get-FileHash data/spam.csv -Algorithm SHA256).Hash   # 440E6EA9FA825578ABFDD7B7932EF8393D72EF86C0C33F64676705CE40B1DFC2
```

## Quality issues found and how they are handled

All handled in memory by `spam_detector/data.py` and `spam_detector/training/dedup.py`. The file on
disk is never modified.

| Issue | Evidence | Handling |
|---|---|---|
| **Truncated messages.** Commas broke CSV quoting, so text spilled into the unnamed columns | 50 rows | Rejoined with `,`. v1 silently trained on the truncated text |
| **Mojibake.** UTF-8 text decoded as Windows-1252 and re-saved as Mac Roman (`£` stored as `E5 A3`, `’` as `89 DB AA`) | 632 non-ASCII byte runs | 495 reversed exactly (`mac_roman → cp1252 → utf-8`). 137 ambiguous runs (mostly a lone byte that appears to be a truncated `ü`, Singlish for "you") kept as-is and counted |
| **HTML entities** (`&amp;`, `&lt;`, `&gt;`) | 309 rows | Unescaped |
| **Anonymization markers** `<#>`, `<DECIMAL>`, `<TIME>`, `<URL>`, `<EMAIL>` | 310 occurrences, **all in ham** | Mapped to the placeholder of the value they replaced, so the model cannot learn "marker ⇒ ham" |
| **Exact duplicates** (incl. case/whitespace variants) | 416 | Collapsed to one message. 0 label conflicts |
| **Near-duplicate campaigns** | 161 groups, up to 15 messages | Kept, but each group stays on one side of every split |
| **Class imbalance** | 12.4% spam after deduplication | Stratified splits, imbalance-aware metrics, cost-based threshold. Class weighting evaluated as a candidate |

## Composition (after deduplication)

| | Messages | Spam | Ham |
|---|---|---|---|
| All | 5,156 | 641 (12.4%) | 4,515 |
| Development | 4,125 | 513 | 3,612 |
| Locked test | 1,031 | 128 | 903 |

Message length: median 61 characters, 99th percentile 276, maximum 910.

## Limitations

See the [model card](MODEL_CARD.md#data). In short: English only, about 2004–2011, UK and Singapore
sources, little modern smishing.

## Using another dataset

`python -m spam_detector evaluate-csv path.csv --label-column label --text-column text` scores the
current model on any labeled CSV. Labels `spam`, `smishing`, and `1` count as spam. Retraining on
new data requires a new loader validated like `data.py` and a new dataset fingerprint in
`configs/pipeline.toml`.
