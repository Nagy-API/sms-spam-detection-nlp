"""Inference: turn raw SMS text into a calibrated, possibly abstaining, prediction.

The pipeline is::

    raw text --(abstention checks)--> normalize --> n-grams --> sparse TF-IDF vector
             --> linear score w.x + b --> Platt calibration --> P(spam)
             --> compare with the cost-derived threshold --> "spam" / "ham"

This module imports only the standard library and sibling pure-Python modules,
so the production API runs without scikit-learn, SciPy, NumPy, or NLTK. A
parity test checks that its scores match scikit-learn's to within 1e-9.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spam_detector.artifact import LinearModelArtifact, LoadedArtifact, load_artifact
from spam_detector.features import SparseVector
from spam_detector.text import letter_profile, normalize

Label = Literal["spam", "ham", "inconclusive"]
#: Why a prediction is inconclusive. ``None`` for a normal spam/ham answer.
InconclusiveReason = Literal["no_letters", "unsupported_script", "no_recognized_features"]


@dataclass(frozen=True)
class Prediction:
    label: Label
    spam_probability: float | None
    reason: InconclusiveReason | None = None
    #: Raw linear score before calibration (log-odds-like). Useful for analysis.
    score: float | None = None
    #: Number of non-zero features the model recognized in the message.
    recognized_features: int = 0


@dataclass(frozen=True)
class Contribution:
    feature: str
    weight: float


class SpamClassifier:
    """Stateless predictor around a verified :class:`LinearModelArtifact`. Thread-safe."""

    def __init__(self, artifact: LinearModelArtifact) -> None:
        self.artifact = artifact
        self._space = artifact.feature_space
        self._coefficients = artifact.coefficients
        self._word_columns = [
            range(offset, offset + block.size)
            for block, offset in zip(self._space.blocks, self._space.offsets, strict=True)
            if block.config.analyzer == "word"
        ]

    @classmethod
    def load(cls, directory: Path) -> tuple[SpamClassifier, LoadedArtifact]:
        loaded = load_artifact(directory)
        return cls(loaded.artifact), loaded

    # ------------------------------------------------------------------ internals
    def vectorize(self, text: str) -> SparseVector:
        return self._space.transform(self._space.analyze(normalize(text)))

    def score_vector(self, vector: SparseVector) -> float:
        coefficients = self._coefficients
        return self.artifact.intercept + sum(coefficients[c] * v for c, v in vector.items())

    def _abstention_reason(self, text: str) -> InconclusiveReason | None:
        letters, latin = letter_profile(text)
        if letters == 0:
            return "no_letters"
        if latin / letters < self.artifact.abstention.min_latin_letter_ratio:
            return "unsupported_script"
        return None

    def _has_known_word(self, vector: SparseVector) -> bool:
        if not self._word_columns:
            return True  # the model has no word block; the rule does not apply
        return any(column in columns for column in vector for columns in self._word_columns)

    # --------------------------------------------------------------------- public
    def predict(self, text: str) -> Prediction:
        reason = self._abstention_reason(text)
        if reason is not None:
            return Prediction(label="inconclusive", spam_probability=None, reason=reason)
        vector = self.vectorize(text)
        if not vector or (self.artifact.abstention.require_known_word and not self._has_known_word(vector)):
            return Prediction(label="inconclusive", spam_probability=None, reason="no_recognized_features")
        score = self.score_vector(vector)
        probability = self.artifact.calibration.probability(score)
        label: Label = "spam" if probability >= self.artifact.threshold else "ham"
        return Prediction(
            label=label,
            spam_probability=probability,
            score=score,
            recognized_features=len(vector),
        )

    def explain(self, text: str, top_k: int = 8) -> list[Contribution]:
        """Largest per-feature contributions ``w_j * x_j`` (positive pushes towards spam).

        For a linear model this decomposition is exact: the score is the
        intercept plus the sum of these contributions. It explains the model's
        arithmetic, not whether the message is actually safe.
        """
        vector = self.vectorize(text)
        contributions = [
            Contribution(self._space.feature_name(c), self._coefficients[c] * v) for c, v in vector.items()
        ]
        contributions.sort(key=lambda item: abs(item.weight), reverse=True)
        return contributions[:top_k]
