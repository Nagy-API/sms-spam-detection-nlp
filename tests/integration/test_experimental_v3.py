from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import MAX_BODY_BYTES, V3_MODEL_DIR, create_app
from spam_detector.artifact import load_artifact, save_artifact
from tests.conftest import RELEASE_DIR


def test_production_serves_bundled_head_with_experimental_status(monkeypatch) -> None:
    monkeypatch.setenv("VERCEL_ENV", "production")
    with TestClient(create_app(RELEASE_DIR)) as client:
        result = client.get("/model/v3")
        assert result.status_code == 200
        assert result.json()["phishing_status"] == "experimental"
        assert result.json()["independent_phishing_validation"] is False
        assert result.json()["baseline_sha256"] == client.get("/ready").json()["model_sha256"]
        assert client.get("/ready").json()["model_version"] == "2.0.0"


def test_bundled_head_vocabulary_contains_only_published_words() -> None:
    baseline = load_artifact(RELEASE_DIR)
    head = load_artifact(V3_MODEL_DIR)
    published = {term for block in baseline.artifact.feature_space.blocks if block.config.analyzer == "word"
                 for term in block.terms}
    assert all(set(block.terms) <= published for block in head.artifact.feature_space.blocks)
    assert head.metadata["baseline_sha256"] == baseline.sha256


def test_v3_preserves_v2_contract_and_uses_verified_head(tmp_path, toy_artifact_dir) -> None:
    toy = load_artifact(toy_artifact_dir).artifact
    directory = tmp_path / "head"
    save_artifact(replace(toy, model_version="3.0.0-dev.1"), directory, {
        "task": "experimental_phishing_head",
        "baseline_sha256": load_artifact(RELEASE_DIR).sha256,
    })
    payload = {"text": "win prize cash free"}
    with TestClient(create_app(RELEASE_DIR, experimental_model_dir=directory)) as client:
        baseline = client.post("/predict", json=payload)
        result = client.post("/predict/v3", json=payload)
        assert baseline.json()["model_version"] == "2.0.0"
        assert "experimental" not in baseline.json()
        assert result.status_code == 200
        assert result.headers["cache-control"] == "no-store"
        assert result.json() == {
            "label": "phishing",
            "spam_probability": baseline.json()["spam_probability"],
            "reason": None,
            "model_version": "3.0.0-dev.1",
            "experimental": True,
        }
        assert client.post("/predict/v3", json={"text": "123"}).json()["reason"] == "no_letters"
        assert client.post("/predict/v3", json={"text": "🙂" * 5001}).status_code == 422
        assert client.post("/predict/v3", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413


def test_invalid_experimental_head_does_not_disable_v2(tmp_path, toy_artifact_dir) -> None:
    corrupted = tmp_path / "corrupt"
    corrupted.mkdir()
    (corrupted / "model.json").write_text("{}", encoding="utf-8")
    (corrupted / "metadata.json").write_bytes((V3_MODEL_DIR / "metadata.json").read_bytes())
    incompatible = tmp_path / "incompatible"
    artifact = load_artifact(V3_MODEL_DIR).artifact
    save_artifact(artifact, incompatible, {"task": "experimental_phishing_head", "baseline_sha256": "0" * 64})
    for directory in (tmp_path / "missing", toy_artifact_dir, corrupted, incompatible):
        with TestClient(create_app(RELEASE_DIR, experimental_model_dir=directory)) as client:
            assert client.post("/predict/v3", json={"text": "hi"}).status_code == 503
            assert client.get("/model/v3").status_code == 503
            assert client.get("/ready").json()["model_version"] == "2.0.0"
