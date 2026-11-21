"""Model candidates under one interface: fit on texts, return linear scores.

Every modern candidate uses the *same* featurizer as production
(:mod:`spam_detector.features`) and differs only in feature blocks and
classifier. All three classifier families are linear in the feature vector,
so any of them can be exported to the pickle-free JSON artifact:

* Logistic regression: ``score = w.x + b`` is the log-odds of spam.
* Linear SVM: ``score = w.x + b`` is a signed margin (not a probability).
* Multinomial Naive Bayes: ``log P(spam|x) - log P(ham|x)
  = (log θ_spam - log θ_ham).x + (log π_spam - log π_ham)``, also linear in
  the count vector ``x``.

The legacy NLTK baseline (:mod:`spam_detector.training.baseline`) implements
the same ``fit``/``scores`` protocol so it is evaluated identically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from spam_detector.features import BlockConfig, FeatureSpace
from spam_detector.text import normalize


RANDOM_STATE = 0


class FittedModel(Protocol):
    def scores(self, texts: Sequence[str]) -> np.ndarray: ...


class AnalysisCache:
    """Memoizes normalization + n-gram extraction, which do not depend on training data.

    During cross-validation the same message is featurized for every fold and
    every candidate; only vocabulary/IDF (fold-dependent) must be recomputed.
    Caching the fold-independent part makes the whole experiment ~10x faster
    without any risk of leakage.
    """

    def __init__(self) -> None:
        self._normalized: dict[str, str] = {}
        self._ngrams: dict[tuple[BlockConfig, str], list[str]] = {}

    def analyze(self, configs: Sequence[BlockConfig], text: str) -> list[list[str]]:
        normalized = self._normalized.get(text)
        if normalized is None:
            normalized = self._normalized[text] = normalize(text)
        result = []
        for config in configs:
            key = (config, text)
            ngrams = self._ngrams.get(key)
            if ngrams is None:
                ngrams = self._ngrams[key] = config.analyze(normalized)
            result.append(ngrams)
        return result


def to_csr(space: FeatureSpace, analyzed: Sequence[Sequence[Sequence[str]]]) -> sparse.csr_matrix:
    indptr, indices, values = [0], [], []
    for document in analyzed:
        vector = space.transform(document)
        for column in sorted(vector):
            indices.append(column)
            values.append(vector[column])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float64), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
        shape=(len(analyzed), space.size),
    )


@dataclass
class FittedLinearModel:
    """A trained featurizer + linear weights. ``scores`` is ``X @ coef + intercept``."""

    space: FeatureSpace
    coefficients: np.ndarray
    intercept: float
    estimator: str
    cache: AnalysisCache = field(repr=False)

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        configs = [block.config for block in self.space.blocks]
        return to_csr(self.space, [self.cache.analyze(configs, t) for t in texts])

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform(texts) @ self.coefficients + self.intercept


@dataclass(frozen=True)
class CandidateSpec:
    """One row of ``configs/pipeline.toml`` ``[[candidates]]``."""

    name: str
    classifier: str
    blocks: tuple[BlockConfig, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateSpec:
        return cls(
            name=data["name"],
            classifier=data["classifier"],
            blocks=tuple(BlockConfig.from_dict(b) for b in data.get("blocks", [])),
            params=dict(data.get("params", {})),
            description=data.get("description", ""),
        )

    @property
    def exportable(self) -> bool:
        return self.classifier != "legacy_nltk_nb"

    def fit(self, texts: Sequence[str], labels: Sequence[int], cache: AnalysisCache) -> FittedModel:
        if self.classifier == "legacy_nltk_nb":
            from spam_detector.training.baseline import LegacyBaseline

            return LegacyBaseline.fit(texts, labels)

        space = FeatureSpace.from_configs(self.blocks)
        analyzed = [cache.analyze(self.blocks, t) for t in texts]
        space.fit(analyzed)  # vocabulary and IDF from the training fold only
        x = to_csr(space, analyzed)
        y = np.asarray(labels)
        params = dict(self.params)
        # liblinear's dual coordinate descent shuffles samples; without a fixed seed
        # two runs give slightly different weights and the artifact is not reproducible.
        params.setdefault("random_state", RANDOM_STATE)

        if self.classifier == "logistic_regression":
            model = LogisticRegression(max_iter=5000, **params).fit(x, y)
            coefficients, intercept = model.coef_.ravel(), float(model.intercept_[0])
        elif self.classifier == "linear_svm":
            model = LinearSVC(max_iter=20000, **params).fit(x, y)
            coefficients, intercept = model.coef_.ravel(), float(model.intercept_[0])
        elif self.classifier == "multinomial_nb":
            params.pop("random_state")  # Naive Bayes is deterministic and takes no seed
            model = MultinomialNB(**params).fit(x, y)
            coefficients = model.feature_log_prob_[1] - model.feature_log_prob_[0]
            intercept = float(model.class_log_prior_[1] - model.class_log_prior_[0])
        else:
            raise ValueError(f"unknown classifier {self.classifier!r}")
        return FittedLinearModel(space, np.asarray(coefficients, dtype=np.float64), intercept, self.classifier, cache)


def fit_platt(scores: np.ndarray, labels: Sequence[int]) -> tuple[float, float]:
    """Fit ``p = sigmoid(a * score + b)`` by (essentially unregularized) logistic regression."""
    model = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(scores).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic
