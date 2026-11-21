"""Deployment guarantees: a light runtime, a verified artifact, and an unchanged contract."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from spam_detector.artifact import load_artifact
from spam_detector.benchmark import benchmark_artifact
from spam_detector.cli import main as cli_main
from tests.conftest import PROJECT_ROOT, RELEASE_DIR

HEAVY_MODULES = ["sklearn", "scipy", "numpy", "nltk", "joblib", "pandas", "spam_detector.training"]


def _run_isolated(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(PROJECT_ROOT)],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def test_api_runtime_never_imports_training_dependencies(tmp_path: Path) -> None:
    result = _run_isolated(
        f"""
        import sys
        sys.path.insert(0, sys.argv[1])
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as client:
            assert client.post('/predict', json={{'text': 'win a free prize now'}}).status_code == 200
        loaded = [m for m in {HEAVY_MODULES!r} if m in sys.modules]
        assert not loaded, loaded
        """,
        tmp_path,  # also proves paths do not depend on the working directory
    )
    assert result.returncode == 0, result.stderr


def test_committed_release_is_valid_and_within_budget() -> None:
    loaded = load_artifact(RELEASE_DIR)
    metadata = loaded.metadata
    assert metadata["parity_with_scikit_learn"]["max_abs_score_difference"] < 1e-9
    assert metadata["environment"]["git_dirty_code_or_config"] is False
    assert metadata["decision"]["threshold"] == loaded.artifact.threshold == 0.75
    assert loaded.artifact.abstention.require_known_word is True
    assert (RELEASE_DIR / "model.json").stat().st_size < 5 * 2**20  # bundle-size budget: 5 MiB


def test_release_metrics_are_traceable_to_reports() -> None:
    metadata = load_artifact(RELEASE_DIR).metadata
    test_report = json.loads((PROJECT_ROOT / "reports/evaluation/test_report.json").read_text(encoding="utf-8"))
    cv = json.loads((PROJECT_ROOT / "reports/experiments/cv_results.json").read_text(encoding="utf-8"))
    assert test_report["artifact"]["sha256"] == metadata["model_sha256"]
    assert cv["selection"]["selected"] == metadata["candidate"]["name"]
    assert metadata["evaluation_summary"]["locked_test"]["spam_f1"] == round(
        test_report["selected_model"]["metrics"]["spam_f1"], 4
    )
    assert test_report["split_fingerprint"] == cv["data"]["split"]["fingerprint"] == metadata["dataset"]["split_fingerprint"]


def test_test_set_access_is_logged_for_every_evaluated_artifact() -> None:
    log = (PROJECT_ROOT / "reports/evaluation/test_access_log.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in log]
    assert entries[-1]["model_sha256"] == load_artifact(RELEASE_DIR).sha256
    assert len(entries) == 2  # documented in docs/EVALUATION.md; update both together


def test_cli_verify_passes_on_the_release(capsys) -> None:
    cli_main(["verify", "--artifact", str(RELEASE_DIR)])
    assert "golden predictions match" in capsys.readouterr().out


def test_inference_is_fast_enough_for_a_serverless_function() -> None:
    result = benchmark_artifact(RELEASE_DIR, repeats=200)
    assert result["load_seconds"] < 3.0
    assert result["latency"]["typical_sms"]["p95_ms"] < 20
    assert result["latency"]["max_length_5000_chars"]["p95_ms"] < 200


def test_vercel_bundle_excludes_training_material() -> None:
    config = json.loads((PROJECT_ROOT / "vercel.json").read_text(encoding="utf-8"))
    function = config["functions"]["app/main.py"]
    assert "models/spam-v2/**" in function["includeFiles"]
    for excluded in ("models/legacy-v1/**", "spam_detector/training/**", "nltk_data/**", "data/**"):
        assert excluded in function["excludeFiles"]
    runtime = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for heavy in ("scikit-learn", "scipy", "numpy", "nltk", "joblib"):
        assert heavy not in runtime.split("#")[0] or f"\n{heavy}=" not in runtime
