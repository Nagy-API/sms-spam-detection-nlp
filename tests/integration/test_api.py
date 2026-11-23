"""HTTP contract of the FastAPI service, against the committed release artifact."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_BODY_BYTES, create_app
from tests.conftest import RELEASE_DIR

HAM = "See you at lunch tomorrow"
SPAM = "Congratulations! You won a free prize. Call now to claim."
GOLDEN = json.loads((Path(__file__).parents[1] / "golden" / "golden_predictions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(RELEASE_DIR)) as test_client:
        yield test_client


def test_health_is_liveness_only(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_ready_reports_the_verified_model(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["model_version"] == "2.0.0"
    assert body["model_sha256"] == GOLDEN["model_sha256"]


def test_model_endpoint_exposes_contract_and_evidence(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["labels"] == ["spam", "ham", "inconclusive"]
    assert body["max_message_chars"] == 5000
    assert 0 < body["decision_threshold"] < 1
    assert "locked_test" in body["evaluation"] and "caveat" in body["evaluation"]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["text"][:30])
def test_predictions_match_golden_file(client: TestClient, case: dict) -> None:
    response = client.post("/predict", json={"text": case["text"]})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == case["label"] and body["reason"] == case["reason"]
    if case["spam_probability"] is None:
        assert body["spam_probability"] is None
    else:
        assert body["spam_probability"] == round(case["spam_probability"], 4)
    assert body["model_version"] == "2.0.0"
    assert set(body) == {"label", "spam_probability", "reason", "model_version"}


@pytest.mark.parametrize("payload", [{}, {"text": 123}, {"text": ""}, {"text": " \t\n "}, {"text": "\u0085\u001c"}])
def test_invalid_input_is_rejected_without_echoing_it(client: TestClient, payload: dict) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", "text"]
    assert set(detail) == {"type", "loc", "msg"}  # no "input": rejected text is never echoed


def test_non_json_body_is_rejected(client: TestClient) -> None:
    response = client.post("/predict", content=b"not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize("padding", [" ", "🙂"])
def test_length_limit_counts_unicode_code_points(client: TestClient, padding: str) -> None:
    text = HAM + padding * (5000 - len(HAM))
    assert client.post("/predict", json={"text": text}).status_code == 200
    rejected = client.post("/predict", json={"text": text + padding})
    assert rejected.status_code == 422 and rejected.json()["detail"][0]["type"] == "string_too_long"


def test_worst_case_json_escaping_fits_the_body_limit(client: TestClient) -> None:
    body = json.dumps({"text": "🙂" * 5000}, ensure_ascii=True).encode()
    assert len(body) <= MAX_BODY_BYTES
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 200 and response.json()["label"] == "inconclusive"


def test_oversized_body_is_rejected_before_parsing(client: TestClient) -> None:
    response = client.post("/predict", content=b"x" * (MAX_BODY_BYTES + 1),
                           headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"]["type"] == "payload_too_large"


def test_oversized_chunked_body_is_rejected(client: TestClient) -> None:
    def chunks():
        for _ in range(20):
            yield b"y" * 8192

    response = client.post("/predict", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_security_and_tracing_headers(client: TestClient) -> None:
    response = client.post("/predict", json={"text": HAM}, headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    page = client.get("/")
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert "content-security-policy" not in client.get("/docs").headers  # Swagger UI loads from a CDN


def test_logs_contain_metadata_but_never_message_text(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    secret = "my bank PIN is 4321 call me"
    logger = logging.getLogger("spam_detector.api")
    logger.addHandler(caplog.handler)
    try:
        client.post("/predict", json={"text": secret})
    finally:
        logger.removeHandler(caplog.handler)
    records = [r for r in caplog.records if r.getMessage() == "request"]
    assert records and records[-1].fields["path"] == "/predict"
    assert records[-1].fields["chars"] == len(secret)
    assert "PIN" not in caplog.text and "4321" not in caplog.text


def test_frontend_is_served_and_calls_the_same_origin_api(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    script_url = re.search(r'src="(script\.js\?v=[^"]+)"', page.text)
    assert script_url
    script = client.get("/" + script_url[1]).text
    endpoint = re.search(r'const API_URL = "([^"]+)"', script)
    assert endpoint and endpoint[1] == "/predict"
    assert client.get("/style.css").status_code == 200


def test_openapi_documents_the_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    request = schema["components"]["schemas"]["PredictRequest"]["properties"]["text"]
    assert request["maxLength"] == 5000 and request["minLength"] == 1
    response = schema["components"]["schemas"]["PredictResponse"]["properties"]
    assert set(response) == {"label", "spam_probability", "reason", "model_version"}
    assert {"413", "422", "503"} <= set(schema["paths"]["/predict"]["post"]["responses"])


# ----------------------------------------------------------------- failure modes
def test_broken_artifact_gives_503_not_a_crash(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "missing-model")) as broken:
        assert broken.get("/health").status_code == 200
        for response in (broken.get("/ready"), broken.post("/predict", json={"text": HAM}), broken.get("/model")):
            assert response.status_code == 503
            assert response.json()["detail"]["type"] == "model_unavailable"


def test_tampered_artifact_is_refused(tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(RELEASE_DIR, copy)
    model = copy / "model.json"
    model.write_bytes(model.read_bytes().replace(b'"threshold":0.75', b'"threshold":0.05'))
    with TestClient(create_app(copy)) as tampered:
        assert tampered.get("/ready").status_code == 503


def test_unexpected_errors_return_a_generic_500(monkeypatch: pytest.MonkeyPatch) -> None:
    from spam_detector.inference import SpamClassifier

    def explode(self, text):
        raise RuntimeError("boom with secret text")

    monkeypatch.setattr(SpamClassifier, "predict", explode)
    with TestClient(create_app(RELEASE_DIR), raise_server_exceptions=False) as failing:
        response = failing.post("/predict", json={"text": HAM})
    assert response.status_code == 500
    assert response.json()["detail"]["type"] == "internal_error"
    assert "secret" not in response.text


def test_model_dir_can_be_set_by_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPAM_MODEL_DIR", str(tmp_path / "nowhere"))
    with TestClient(create_app()) as client:
        assert client.get("/ready").status_code == 503


# ----------------------------------------------------- regressions from hosted review
@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "PATCH", "DELETE"])
def test_wrong_method_on_predict_is_405_not_404(client: TestClient, method: str) -> None:
    # Found on the hosted Preview: the static mount at "/" answered GET /predict with 404.
    response = client.request(method, "/predict")
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


@pytest.mark.parametrize(
    "supplied",
    ["x" * 129, "has space", "new\nline", "<script>", "ünïcode".encode("latin-1"), ""],
)
def test_unsafe_request_ids_are_replaced(client: TestClient, supplied: str | bytes) -> None:
    response = client.get("/health", headers={"x-request-id": supplied} if supplied else {})
    echoed = response.headers["x-request-id"]
    assert echoed != supplied
    assert re.fullmatch(r"[0-9a-f]{32}", echoed)


def test_safe_request_ids_are_kept(client: TestClient) -> None:
    for supplied in ("abc-123", "trace.id:42_X", "a" * 128):
        assert client.get("/health", headers={"x-request-id": supplied}).headers["x-request-id"] == supplied


def test_request_logs_report_peak_memory(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    import sys

    logger = logging.getLogger("spam_detector.api")
    logger.addHandler(caplog.handler)
    try:
        client.get("/health")
    finally:
        logger.removeHandler(caplog.handler)
    fields = [r for r in caplog.records if r.getMessage() == "request"][-1].fields
    assert "peak_rss_mib" in fields
    if sys.platform != "win32":  # the resource module does not exist on Windows
        assert fields["peak_rss_mib"] > 0
