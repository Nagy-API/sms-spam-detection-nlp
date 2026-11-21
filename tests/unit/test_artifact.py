import json
from pathlib import Path

import pytest

from spam_detector.artifact import (
    ArtifactError,
    LinearModelArtifact,
    PlattCalibration,
    load_artifact,
    save_artifact,
)
from tests.conftest import build_toy_artifact


def test_round_trip_preserves_everything(toy_artifact_dir: Path) -> None:
    original = build_toy_artifact()
    loaded = load_artifact(toy_artifact_dir)
    assert loaded.artifact.to_json_bytes() == original.to_json_bytes()
    assert loaded.metadata["model_sha256"] == loaded.sha256


def test_serialization_is_deterministic() -> None:
    assert build_toy_artifact().to_json_bytes() == build_toy_artifact().to_json_bytes()


def test_artifacts_are_immutable(toy_artifact_dir: Path) -> None:
    with pytest.raises(ArtifactError, match="immutable"):
        save_artifact(build_toy_artifact(), toy_artifact_dir, {})


def test_tampered_model_fails_checksum(toy_artifact_dir: Path) -> None:
    model = toy_artifact_dir / "model.json"
    model.write_bytes(model.read_bytes().replace(b'"intercept":-1.0', b'"intercept":9.0'))
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_artifact(toy_artifact_dir)


@pytest.mark.parametrize("missing", ["model.json", "metadata.json"])
def test_missing_files_are_reported(toy_artifact_dir: Path, missing: str) -> None:
    (toy_artifact_dir / missing).unlink()
    with pytest.raises(ArtifactError, match=f"missing artifact file.*{missing}"):
        load_artifact(toy_artifact_dir)


def _rewrite(directory: Path, mutate) -> None:
    """Edit model.json and re-sign it, to test validation beyond the checksum."""
    data = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    mutate(data)
    content = json.dumps(data).encode()
    (directory / "model.json").write_bytes(content)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    from spam_detector.artifact import sha256_bytes

    metadata["model_sha256"] = sha256_bytes(content)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema_version=99), "unsupported artifact schema"),
        (lambda d: d.update(preprocessing_version="0.1"), "preprocessing"),
        (lambda d: d.update(labels=["spam", "ham"]), "unexpected labels"),
        (lambda d: d["feature_blocks"][0]["coefficients"].pop(), "different lengths"),
        (lambda d: d["calibration"].update(method="isotonic"), "unsupported calibration"),
        (lambda d: d.pop("intercept"), "malformed"),
        (lambda d: d["decision"].update(threshold=1.5), "threshold"),
        (lambda d: d.update(model_version="9.9.9"), "model_version differs"),
    ],
)
def test_incompatible_or_malformed_artifacts_are_rejected(toy_artifact_dir: Path, mutate, message: str) -> None:
    _rewrite(toy_artifact_dir, mutate)
    with pytest.raises(ArtifactError, match=message):
        load_artifact(toy_artifact_dir)


def test_invalid_json_is_reported(toy_artifact_dir: Path) -> None:
    (toy_artifact_dir / "metadata.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match=r"metadata.json is not valid JSON"):
        load_artifact(toy_artifact_dir)


def test_non_finite_weights_are_rejected() -> None:
    artifact = build_toy_artifact()
    with pytest.raises(ArtifactError, match="non-finite"):
        LinearModelArtifact(
            model_version="x", feature_space=artifact.feature_space,
            coefficients=[float("nan")] * artifact.feature_space.size, intercept=0.0,
            calibration=PlattCalibration(), threshold=0.5,
        )


@pytest.mark.parametrize(("score", "expected"), [(0.0, 0.5), (1000.0, 1.0), (-1000.0, 0.0)])
def test_platt_probability_is_numerically_stable(score: float, expected: float) -> None:
    assert PlattCalibration(1.0, 0.0).probability(score) == pytest.approx(expected)
