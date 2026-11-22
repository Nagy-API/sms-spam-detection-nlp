from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from spam_detector.artifact import load_artifact, save_artifact
from tests.conftest import RELEASE_DIR


def test_v3_is_opt_in() -> None:
    with TestClient(create_app(RELEASE_DIR)) as client:
        assert client.post("/predict/v3", json={"text": "Hello"}).status_code != 200


def test_production_disables_experimental_route(monkeypatch, toy_artifact_dir) -> None:
    monkeypatch.setenv("VERCEL_ENV", "production")
    monkeypatch.setenv("TRACE_EXPERIMENTAL_MODEL_DIR", str(toy_artifact_dir))
    with TestClient(create_app(RELEASE_DIR)) as client:
        assert client.post("/predict/v3", json={"text": "Hello"}).status_code == 405
        assert client.get("/ready").json()["model_version"] == "2.0.0"


def test_v3_preserves_v2_contract_and_uses_verified_head(tmp_path, toy_artifact_dir) -> None:
    toy = load_artifact(toy_artifact_dir).artifact
    directory = tmp_path / "head"
    save_artifact(replace(toy, model_version="3.0.0-dev.1"), directory, {"task": "experimental_phishing_head"})
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


def test_invalid_experimental_head_does_not_disable_v2(tmp_path, toy_artifact_dir) -> None:
    for directory in (tmp_path / "missing", toy_artifact_dir):
        with TestClient(create_app(RELEASE_DIR, experimental_model_dir=directory)) as client:
            assert client.post("/predict/v3", json={"text": "hi"}).status_code == 503
            assert client.get("/ready").json()["model_version"] == "2.0.0"
