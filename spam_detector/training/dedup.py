"""Duplicate removal and near-duplicate grouping.

Why this matters: spam is sent in campaigns. The corpus contains the same
template many times with a different phone number, prize amount, or casing.
If one copy lands in the training set and another in the test set, the test
measures *memorization of a campaign*, not generalization, and the reported
score is inflated. Two levels of protection:

1. **Exact duplicates** (after Unicode/case/whitespace canonicalization) are
   collapsed to a single message. Duplicates with conflicting labels would be
   label noise; they are dropped and reported (the current corpus has none).
2. **Near duplicates** are clustered into *groups*. Two messages are linked if
   their template forms (numbers and URLs masked) are identical, or if the
   Jaccard similarity of their character 5-gram shingle sets is at least the
   configured threshold. Linked messages are merged with union-find, and the
   splitter keeps every group entirely on one side of every split.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from spam_detector.data import Message

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_DIGITS_RE = re.compile(r"\d+")
_NON_WORD_RE = re.compile(r"[\W_]+")


def canonical_key(text: str) -> str:
    """Key for exact-duplicate detection: NFKC, case-folded, whitespace-collapsed."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def template_key(text: str) -> str:
    """Key for template detection: canonical form with URLs/digits masked and punctuation dropped."""
    text = _URL_RE.sub(" u ", canonical_key(text))
    text = _DIGITS_RE.sub(" 0 ", text)
    return " ".join(_NON_WORD_RE.sub(" ", text).split())


def shingles(text: str, size: int = 5) -> set[str]:
    key = template_key(text)
    if len(key) <= size:
        return {key}
    return {key[i : i + size] for i in range(len(key) - size + 1)}


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Smallest index becomes the root so group ids are deterministic.
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


@dataclass
class DedupReport:
    input_messages: int = 0
    exact_duplicates_removed: int = 0
    conflicting_label_messages_removed: int = 0
    output_messages: int = 0
    groups: int = 0
    multi_message_groups: int = 0
    largest_group: int = 0
    mixed_label_groups: int = 0
    linked_by_template: int = 0
    linked_by_jaccard: int = 0
    jaccard_threshold: float = 0.0
    label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def remove_exact_duplicates(messages: Sequence[Message]) -> tuple[list[Message], int, int]:
    """Keep the first occurrence of each canonical text; drop label-conflicting texts entirely."""
    labels_by_key: dict[str, set[str]] = defaultdict(set)
    for message in messages:
        labels_by_key[canonical_key(message.text)].add(message.label)
    kept: list[Message] = []
    seen: set[str] = set()
    duplicates = conflicts = 0
    for message in messages:
        key = canonical_key(message.text)
        if len(labels_by_key[key]) > 1:
            conflicts += 1
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(message)
    return kept, duplicates, conflicts


def near_duplicate_groups(
    messages: Sequence[Message], jaccard_threshold: float
) -> tuple[list[int], int, int]:
    """Assign a group id to every message. Returns ``(group_ids, template_links, jaccard_links)``."""
    n = len(messages)
    union_find = _UnionFind(n)

    template_links = 0
    first_with_template: dict[str, int] = {}
    for i, message in enumerate(messages):
        key = template_key(message.text)
        if key in first_with_template:
            union_find.union(first_with_template[key], i)
            template_links += 1
        else:
            first_with_template[key] = i

    # Pairwise Jaccard via sparse binary matrix products: |A∩B| = (X Xᵀ)_ab.
    vocabulary: dict[str, int] = {}
    rows, cols = [], []
    for i, message in enumerate(messages):
        for shingle in shingles(message.text):
            rows.append(i)
            cols.append(vocabulary.setdefault(shingle, len(vocabulary)))
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(vocabulary))
    )
    matrix.data[:] = 1.0  # a shingle counts once per message
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    intersections = sparse.triu(matrix @ matrix.T, k=1).tocoo()
    unions = sizes[intersections.row] + sizes[intersections.col] - intersections.data
    similar = intersections.data / unions >= jaccard_threshold
    jaccard_links = 0
    for a, b in zip(intersections.row[similar], intersections.col[similar], strict=True):
        if union_find.find(int(a)) != union_find.find(int(b)):
            jaccard_links += 1
        union_find.union(int(a), int(b))

    roots = [union_find.find(i) for i in range(n)]
    dense_ids: dict[int, int] = {}
    group_ids = [dense_ids.setdefault(root, len(dense_ids)) for root in roots]
    return group_ids, template_links, jaccard_links


def deduplicate_and_group(
    messages: Sequence[Message], jaccard_threshold: float
) -> tuple[list[Message], list[int], DedupReport]:
    report = DedupReport(input_messages=len(messages), jaccard_threshold=jaccard_threshold)
    unique, report.exact_duplicates_removed, report.conflicting_label_messages_removed = (
        remove_exact_duplicates(messages)
    )
    group_ids, report.linked_by_template, report.linked_by_jaccard = near_duplicate_groups(
        unique, jaccard_threshold
    )
    sizes = Counter(group_ids)
    labels_per_group: dict[int, set[str]] = defaultdict(set)
    for message, group in zip(unique, group_ids, strict=True):
        labels_per_group[group].add(message.label)
    report.output_messages = len(unique)
    report.groups = len(sizes)
    report.multi_message_groups = sum(1 for s in sizes.values() if s > 1)
    report.largest_group = max(sizes.values())
    report.mixed_label_groups = sum(1 for labels in labels_per_group.values() if len(labels) > 1)
    report.label_counts = dict(sorted(Counter(m.label for m in unique).items()))
    return unique, group_ids, report
