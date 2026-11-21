"""Feature extraction: n-gram analyzers and TF-IDF / count weighting in pure Python.

Why not just use scikit-learn's ``TfidfVectorizer`` at inference time?
Because serving it requires scikit-learn, SciPy and NumPy (most of the former
~207 MiB serverless bundle) and a pickle file (which executes arbitrary code when
loaded). This module re-implements exactly the parts we need, is used by
*both* training and serving, and is verified against scikit-learn's reference
implementation in ``tests/unit/test_features.py``.

Vocabulary
    A *feature block* (for example "word 1-2 grams" or "character 2-5 grams")
    maps each n-gram to a column. Blocks are concatenated: block ``b`` occupies
    columns ``[offset_b, offset_b + len(vocabulary_b))``.

Weighting (per block)
    * ``count``: raw n-gram counts (what Multinomial Naive Bayes expects).
    * ``tfidf``: ``tf * idf`` with ``idf = ln((1 + n) / (1 + df)) + 1`` (the
      "smooth idf" formula used by scikit-learn) and optional sublinear
      ``tf = 1 + ln(count)``, followed by optional L2 normalization so long
      messages do not get larger scores just for being long.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

AnalyzerKind = Literal["word", "char_wb"]
Weighting = Literal["count", "tfidf"]

#: A sparse feature vector: column index -> value. Columns absent from the
#: mapping are zero. Messages are short, so a dict is both simple and fast.
SparseVector = dict[int, float]

# Words are runs of letters/digits (placeholders such as "zzphone" included),
# plus the symbols that carry spam signal on their own.
_WORD_TOKEN_RE = re.compile(r"[^\W_]+|[£$€!?]")
_MULTIPLE_WHITESPACE_RE = re.compile(r"\s\s+")


def word_tokens(text: str) -> list[str]:
    """Split normalized text into word tokens, keeping ``£ $ € ! ?`` as tokens."""
    return _WORD_TOKEN_RE.findall(text)


def word_ngrams(tokens: Sequence[str], min_n: int, max_n: int) -> list[str]:
    """Return contiguous word n-grams joined by single spaces."""
    ngrams: list[str] = []
    for n in range(min_n, max_n + 1):
        ngrams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return ngrams


def char_wb_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    """Character n-grams taken only inside word boundaries, padded with spaces.

    This mirrors scikit-learn's ``analyzer="char_wb"``: each whitespace-separated
    word is padded as ``" word "`` and n-grams never span two words. A word
    shorter than ``n`` contributes its padded form once.
    """
    ngrams: list[str] = []
    text = _MULTIPLE_WHITESPACE_RE.sub(" ", text)
    for word in text.split():
        padded = f" {word} "
        length = len(padded)
        for n in range(min_n, max_n + 1):
            offset = 0
            ngrams.append(padded[offset : offset + n])
            while offset + n < length:
                offset += 1
                ngrams.append(padded[offset : offset + n])
            if offset == 0:  # the padded word is shorter than n: counted once
                break
    return ngrams


@dataclass(frozen=True)
class BlockConfig:
    """Hyper-parameters of one feature block. Stored verbatim in the artifact."""

    analyzer: AnalyzerKind
    ngram_range: tuple[int, int]
    weighting: Weighting = "tfidf"
    sublinear_tf: bool = True
    l2_normalize: bool = True
    min_df: int = 1

    def __post_init__(self) -> None:
        low, high = self.ngram_range
        if not 1 <= low <= high:
            raise ValueError(f"invalid ngram_range {self.ngram_range}")
        if self.min_df < 1:
            raise ValueError("min_df must be >= 1")

    def analyze(self, normalized_text: str) -> list[str]:
        low, high = self.ngram_range
        if self.analyzer == "word":
            return word_ngrams(word_tokens(normalized_text), low, high)
        if self.analyzer == "char_wb":
            return char_wb_ngrams(normalized_text, low, high)
        raise ValueError(f"unknown analyzer {self.analyzer!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "weighting": self.weighting,
            "sublinear_tf": self.sublinear_tf,
            "l2_normalize": self.l2_normalize,
            "min_df": self.min_df,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BlockConfig:
        low, high = data["ngram_range"]  # type: ignore[misc]
        return cls(
            analyzer=data["analyzer"],  # type: ignore[arg-type]
            ngram_range=(int(low), int(high)),
            weighting=data.get("weighting", "tfidf"),  # type: ignore[arg-type]
            sublinear_tf=bool(data.get("sublinear_tf", True)),
            l2_normalize=bool(data.get("l2_normalize", True)),
            min_df=int(data.get("min_df", 1)),  # type: ignore[arg-type]
        )


@dataclass
class FeatureBlock:
    """A fitted block: configuration plus the vocabulary and IDF learned from training data."""

    config: BlockConfig
    terms: list[str] = field(default_factory=list)
    idf: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = {term: i for i, term in enumerate(self.terms)}

    @property
    def size(self) -> int:
        return len(self.terms)

    def fit(self, analyzed_documents: Iterable[Sequence[str]]) -> FeatureBlock:
        """Learn vocabulary and document frequencies from *training* documents only."""
        document_frequency: Counter[str] = Counter()
        n_documents = 0
        for ngrams in analyzed_documents:
            n_documents += 1
            document_frequency.update(set(ngrams))
        # Sorting makes column order independent of corpus order, so the
        # artifact is byte-for-byte reproducible.
        self.terms = sorted(t for t, df in document_frequency.items() if df >= self.config.min_df)
        self._index = {term: i for i, term in enumerate(self.terms)}
        self.idf = [
            math.log((1 + n_documents) / (1 + document_frequency[t])) + 1.0 for t in self.terms
        ]
        return self

    def transform(self, ngrams: Sequence[str], offset: int = 0) -> SparseVector:
        """Weight one analyzed document. Out-of-vocabulary n-grams are ignored."""
        counts: Counter[int] = Counter()
        index = self._index
        for gram in ngrams:
            column = index.get(gram)
            if column is not None:
                counts[column] += 1
        config = self.config
        values: dict[int, float] = {}
        for column, count in counts.items():
            if config.weighting == "count":
                values[column] = float(count)
            else:
                tf = 1.0 + math.log(count) if config.sublinear_tf else float(count)
                values[column] = tf * self.idf[column]
        if config.l2_normalize and values:
            norm = math.sqrt(sum(v * v for v in values.values()))
            values = {c: v / norm for c, v in values.items()}
        return {column + offset: value for column, value in values.items()}


class FeatureSpace:
    """An ordered collection of feature blocks forming one feature vector."""

    def __init__(self, blocks: Sequence[FeatureBlock]) -> None:
        if not blocks:
            raise ValueError("a feature space needs at least one block")
        self.blocks = list(blocks)
        self.offsets: list[int] = []
        self._recompute_offsets()

    @classmethod
    def from_configs(cls, configs: Sequence[BlockConfig]) -> FeatureSpace:
        return cls([FeatureBlock(config) for config in configs])

    def _recompute_offsets(self) -> None:
        self.offsets, total = [], 0
        for block in self.blocks:
            self.offsets.append(total)
            total += block.size

    @property
    def size(self) -> int:
        return sum(block.size for block in self.blocks)

    def analyze(self, normalized_text: str) -> list[list[str]]:
        """Extract raw n-grams for every block. Pure function of the text, so cacheable."""
        return [block.config.analyze(normalized_text) for block in self.blocks]

    def fit(self, analyzed_corpus: Sequence[Sequence[Sequence[str]]]) -> FeatureSpace:
        for i, block in enumerate(self.blocks):
            block.fit(document[i] for document in analyzed_corpus)
        self._recompute_offsets()
        return self

    def transform(self, analyzed_document: Sequence[Sequence[str]]) -> SparseVector:
        vector: SparseVector = {}
        for block, offset, ngrams in zip(self.blocks, self.offsets, analyzed_document, strict=True):
            vector.update(block.transform(ngrams, offset))
        return vector

    def feature_name(self, column: int) -> str:
        """Human-readable name such as ``word:free entry`` (for explanations and reports)."""
        for block, offset in zip(reversed(self.blocks), reversed(self.offsets), strict=True):
            if column >= offset:
                return f"{block.config.analyzer}:{block.terms[column - offset]}"
        raise IndexError(column)
