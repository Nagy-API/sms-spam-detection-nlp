"""Audit the original (v1) evaluation protocol.

Reproduces the v1 split exactly (raw Latin-1 read, ``drop_duplicates`` on the
raw label/text pair, ``train_test_split(test_size=0.2, random_state=42,
stratify=label)``), re-scores the committed v1 pickles on that holdout, and
measures how many holdout messages had an exact or near duplicate in the v1
training set. The headline v1 metrics are then recomputed on the subset of
holdout messages *without* such a duplicate.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from spam_detector.data import Message
from spam_detector.training.baseline import legacy_preprocess
from spam_detector.training.dedup import canonical_key, near_duplicate_groups, template_key
from spam_detector.training.metrics import classification_metrics


def _legacy_rows(path: Path) -> list[tuple[str, str]]:
    reader = csv.reader(io.StringIO(path.read_bytes().decode("latin-1"), newline=""))
    next(reader)
    seen, rows = set(), []
    for row in reader:
        pair = (row[0], row[1])  # v1 kept only columns v1/v2 (truncating 50 split messages)
        if pair not in seen:
            seen.add(pair)
            rows.append(pair)
    return rows


def audit_legacy_protocol(data_path: Path, legacy_model_dir: Path, jaccard: float) -> dict[str, Any]:
    rows = _legacy_rows(data_path)
    labels = [label for label, _ in rows]
    indices = np.arange(len(rows))
    train_index, test_index = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

    vectorizer = joblib.load(legacy_model_dir / "vectorizer.pkl")
    model = joblib.load(legacy_model_dir / "model.pkl")
    test_texts = [rows[i][1] for i in test_index]
    x = vectorizer.transform([legacy_preprocess(t) for t in test_texts])
    y = np.array([int(rows[i][0] == "spam") for i in test_index])
    spam_column = list(model.classes_).index("spam")
    probability = model.predict_proba(x)[:, spam_column]
    predicted = (probability >= 0.5).astype(int)

    train_canonical = {canonical_key(rows[i][1]) for i in train_index}
    train_template = {template_key(rows[i][1]) for i in train_index}
    messages = [Message(i, label, text) for i, (label, text) in enumerate(rows)]
    groups, _, _ = near_duplicate_groups(messages, jaccard)
    train_groups = {groups[i] for i in train_index}

    exact = np.array([canonical_key(rows[i][1]) in train_canonical for i in test_index])
    template = np.array([template_key(rows[i][1]) in train_template for i in test_index])
    near = np.array([groups[i] in train_groups for i in test_index])
    clean = ~near

    return {
        "reproduced_split": {"train": len(train_index), "test": len(test_index),
                             "test_spam": int(y.sum())},
        "zero_feature_test_messages": int((x.getnnz(axis=1) == 0).sum()),
        "leakage": {
            "test_with_case_or_whitespace_duplicate_in_train": int(exact.sum()),
            "test_with_same_template_in_train": int(template.sum()),
            "test_with_near_duplicate_in_train": int(near.sum()),
            "spam_test_with_near_duplicate_in_train": int((near & (y == 1)).sum()),
            "jaccard_threshold": jaccard,
        },
        "v1_artifact_on_v1_holdout": classification_metrics(y, predicted, probability, probability),
        "v1_artifact_on_v1_holdout_without_near_duplicates": {
            "messages": int(clean.sum()),
            "spam": int(y[clean].sum()),
            **classification_metrics(y[clean], predicted[clean], probability[clean], probability[clean]),
        },
    }
