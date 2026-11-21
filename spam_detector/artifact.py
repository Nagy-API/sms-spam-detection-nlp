"""Model artifact format: a portable, pickle-free, checksummed linear model.

An artifact is a directory with two files:

``model.json``
    Everything needed to compute a prediction: preprocessing version, feature
    blocks (vocabulary, IDF, weights), intercept, probability calibration,
    decision threshold, and abstention policy. It is written deterministically
    (sorted keys, no timestamps) so retraining on the same data with the same
    code reproduces it byte for byte.

``metadata.json``
    Provenance and evidence: SHA-256 of ``model.json``, dataset and split
    fingerprints, library versions, Git commit, creation time, and the
    evaluation metrics. Keeping volatile fields here keeps ``model.json``
    reproducible.

Why JSON instead of joblib/pickle? Unpickling executes code, so a tampered
``.pkl`` is a remote-code-execution vector, and pickles are tied to the exact
scikit-learn version that wrote them (the legacy artifacts only load with
scikit-learn 1.8.0). Every candidate we evaluate is a linear model, so its
whole state is a few arrays that JSON represents safely and portably.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spam_detector.features import BlockConfig, FeatureBlock, FeatureSpace
from spam_detector.text import PREPROCESSING_VERSION

#: Version of the on-disk layout. Bump on any incompatible change to the JSON.
ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILENAME = "model.json"
METADATA_FILENAME = "metadata.json"
LABELS = ("ham", "spam")


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, corrupted, or incompatible with this code."""


@dataclass(frozen=True)
class PlattCalibration:
    """Maps a raw linear score to a probability: ``p = 1 / (1 + exp(-(a * score + b)))``.

    Platt scaling is a one-feature logistic regression fitted on out-of-fold
    scores. It corrects a classifier whose ranking is good but whose scores are
    not probabilities (SVM margins, Naive Bayes log-odds, class-weighted logistic
    regression). ``a=1, b=0`` is the identity for a plain logistic regression.
    """

    a: float = 1.0
    b: float = 0.0

    def probability(self, score: float) -> float:
        z = self.a * score + self.b
        # Numerically stable logistic function for large |z|.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


@dataclass(frozen=True)
class AbstentionPolicy:
    """When the model should answer "inconclusive" instead of guessing.

    ``min_latin_letter_ratio``: the training data is English SMS. If fewer than
    this fraction of the letters are Latin script, the input is out of
    distribution and the model abstains rather than extrapolating.
    """

    min_latin_letter_ratio: float = 0.5


@dataclass
class LinearModelArtifact:
    """In-memory form of ``model.json``."""

    model_version: str
    feature_space: FeatureSpace
    coefficients: list[float]
    intercept: float
    calibration: PlattCalibration
    threshold: float
    abstention: AbstentionPolicy = field(default_factory=AbstentionPolicy)
    preprocessing_version: str = PREPROCESSING_VERSION
    estimator: str = "unknown"
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.coefficients) != self.feature_space.size:
            raise ArtifactError(
                f"{len(self.coefficients)} coefficients for {self.feature_space.size} features"
            )
        if not 0.0 < self.threshold < 1.0:
            raise ArtifactError(f"threshold must be in (0, 1), got {self.threshold}")
        if not all(math.isfinite(v) for v in (*self.coefficients, self.intercept)):
            raise ArtifactError("non-finite model weight")

    # ----------------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        blocks = []
        for block, offset in zip(self.feature_space.blocks, self.feature_space.offsets, strict=True):
            blocks.append(
                {
                    "config": block.config.to_dict(),
                    "terms": block.terms,
                    "idf": block.idf,
                    "coefficients": self.coefficients[offset : offset + block.size],
                }
            )
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "estimator": self.estimator,
            "labels": list(LABELS),
            "preprocessing_version": self.preprocessing_version,
            "feature_blocks": blocks,
            "intercept": self.intercept,
            "calibration": {"method": "platt", "a": self.calibration.a, "b": self.calibration.b},
            "decision": {"positive_label": "spam", "threshold": self.threshold},
            "abstention": {"min_latin_letter_ratio": self.abstention.min_latin_letter_ratio},
        }

    def to_json_bytes(self) -> bytes:
        # sort_keys + fixed separators + repr-precise floats => deterministic bytes.
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return text.encode("utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearModelArtifact:
        schema = data.get("schema_version")
        if schema != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(
                f"unsupported artifact schema {schema!r}; this code reads {ARTIFACT_SCHEMA_VERSION}"
            )
        if data.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ArtifactError(
                "artifact was trained with preprocessing "
                f"{data.get('preprocessing_version')!r} but this code implements "
                f"{PREPROCESSING_VERSION!r}; retrain or deploy matching code"
            )
        if list(data.get("labels", [])) != list(LABELS):
            raise ArtifactError(f"unexpected labels {data.get('labels')!r}")
        try:
            blocks, coefficients = [], []
            for raw in data["feature_blocks"]:
                block = FeatureBlock(BlockConfig.from_dict(raw["config"]), list(raw["terms"]), list(raw["idf"]))
                if not len(block.terms) == len(block.idf) == len(raw["coefficients"]):
                    raise ArtifactError("feature block arrays have different lengths")
                blocks.append(block)
                coefficients.extend(float(c) for c in raw["coefficients"])
            calibration = data["calibration"]
            if calibration.get("method") != "platt":
                raise ArtifactError(f"unsupported calibration {calibration.get('method')!r}")
            return cls(
                model_version=str(data["model_version"]),
                estimator=str(data.get("estimator", "unknown")),
                feature_space=FeatureSpace(blocks),
                coefficients=coefficients,
                intercept=float(data["intercept"]),
                calibration=PlattCalibration(float(calibration["a"]), float(calibration["b"])),
                threshold=float(data["decision"]["threshold"]),
                abstention=AbstentionPolicy(float(data["abstention"]["min_latin_letter_ratio"])),
                preprocessing_version=str(data["preprocessing_version"]),
                schema_version=int(schema),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"malformed model.json: {error!r}") from error


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def save_artifact(artifact: LinearModelArtifact, directory: Path, metadata: Mapping[str, Any]) -> str:
    """Write ``model.json`` and ``metadata.json``; return the model checksum.

    Refuses to overwrite an existing artifact: published model versions are
    immutable, so a new model always gets a new version directory.
    """
    model_path = directory / MODEL_FILENAME
    if model_path.exists():
        raise ArtifactError(f"{model_path} already exists; artifacts are immutable, bump the version")
    directory.mkdir(parents=True, exist_ok=True)
    content = artifact.to_json_bytes()
    model_path.write_bytes(content)
    checksum = sha256_bytes(content)
    full_metadata = {**metadata, "model_version": artifact.model_version, "model_sha256": checksum}
    (directory / METADATA_FILENAME).write_text(
        json.dumps(full_metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return checksum


@dataclass(frozen=True)
class LoadedArtifact:
    artifact: LinearModelArtifact
    metadata: dict[str, Any]
    sha256: str
    directory: Path


def load_artifact(directory: Path) -> LoadedArtifact:
    """Load and verify an artifact directory.

    Verification order: files exist -> checksum matches metadata -> JSON parses
    -> schema/preprocessing/labels compatible -> array shapes consistent. Any
    failure raises :class:`ArtifactError` with an actionable message.
    """
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME
    for path in (model_path, metadata_path):
        if not path.is_file():
            raise ArtifactError(f"missing artifact file: {path}")
    content = model_path.read_bytes()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"metadata.json is not valid JSON: {error}") from error
    checksum = sha256_bytes(content)
    if metadata.get("model_sha256") != checksum:
        raise ArtifactError(
            f"checksum mismatch for {model_path}: metadata says {metadata.get('model_sha256')!r}, "
            f"file is {checksum!r}"
        )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"model.json is not valid JSON: {error}") from error
    artifact = LinearModelArtifact.from_dict(data)
    if metadata.get("model_version") != artifact.model_version:
        raise ArtifactError("model_version differs between model.json and metadata.json")
    return LoadedArtifact(artifact=artifact, metadata=metadata, sha256=checksum, directory=directory)
