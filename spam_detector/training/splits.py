"""Deterministic, group-aware, stratified splitting.

Protocol (see ``docs/EVALUATION.md``):

* The deduplicated corpus is split once into **development** (~80%) and a
  **locked test set** (~20%) with ``StratifiedGroupKFold``: every
  near-duplicate group stays on one side, and the spam ratio is preserved.
* Model selection, hyper-parameter choice, calibration, and threshold choice
  use only the development set, through repeated group-aware K-fold CV.
* The locked test set is scored once, after the release candidate is frozen.

The split is identified by a fingerprint: SHA-256 over sorted lines of
``sha256(canonical text) | label | split``. It is committed to the repository so
anyone with the same CSV can verify they reproduced the exact split, while no
message text (or even per-message hashes) is published.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from spam_detector.data import Message
from spam_detector.training.dedup import canonical_key


@dataclass(frozen=True)
class DatasetSplit:
    development: list[Message]
    development_groups: list[int]
    test: list[Message]
    test_groups: list[int]

    def fingerprint(self) -> str:
        lines = sorted(
            f"{hashlib.sha256(canonical_key(m.text).encode()).hexdigest()}|{m.label}|{name}"
            for name, messages in (("development", self.development), ("test", self.test))
            for m in messages
        )
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    def summary(self) -> dict[str, object]:
        def describe(messages: Sequence[Message], groups: Sequence[int]) -> dict[str, object]:
            spam = sum(m.is_spam for m in messages)
            return {
                "messages": len(messages),
                "spam": spam,
                "ham": len(messages) - spam,
                "spam_rate": round(spam / len(messages), 4),
                "groups": len(set(groups)),
            }

        return {
            "development": describe(self.development, self.development_groups),
            "test": describe(self.test, self.test_groups),
            "fingerprint": self.fingerprint(),
        }


def split_development_test(
    messages: Sequence[Message], groups: Sequence[int], test_fraction: float, seed: int
) -> DatasetSplit:
    n_splits = round(1 / test_fraction)
    labels = np.array([m.is_spam for m in messages])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    development_index, test_index = next(splitter.split(np.zeros(len(messages)), labels, groups))
    split = DatasetSplit(
        development=[messages[i] for i in development_index],
        development_groups=[groups[i] for i in development_index],
        test=[messages[i] for i in test_index],
        test_groups=[groups[i] for i in test_index],
    )
    assert_no_group_leakage(split.development_groups, split.test_groups)
    return split


def assert_no_group_leakage(left: Sequence[int], right: Sequence[int]) -> None:
    shared = set(left) & set(right)
    if shared:
        raise AssertionError(f"{len(shared)} near-duplicate groups appear on both sides of a split")


def cross_validation_folds(
    labels: Sequence[int], groups: Sequence[int], n_folds: int, repeats: int, seed: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, fold, train_index, validation_index)`` for repeated group-aware CV."""
    y = np.asarray(labels)
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, validation) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            assert_no_group_leakage([groups[i] for i in train], [groups[i] for i in validation])
            yield repeat, fold, train, validation
