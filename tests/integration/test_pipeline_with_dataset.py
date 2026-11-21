"""End-to-end checks that need data/spam.csv (skipped automatically when it is absent)."""

from __future__ import annotations

import json

import pytest

from spam_detector.config import PipelineConfig
from spam_detector.text import PLACEHOLDER_TOKENS
from tests.conftest import PROJECT_ROOT, RELEASE_DIR

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def prepared():
    from spam_detector.training.experiment import prepare_data

    return prepare_data(PipelineConfig.load())


def test_dataset_report_matches_the_committed_one(prepared) -> None:
    committed = json.loads((PROJECT_ROOT / "reports/data/dataset_report.json").read_text(encoding="utf-8"))
    assert prepared.report() == committed


def test_no_leakage_between_development_and_test(prepared) -> None:
    from spam_detector.training.dedup import canonical_key, template_key

    split = prepared.split
    assert not set(split.development_groups) & set(split.test_groups)
    for key in (canonical_key, template_key):
        assert not {key(m.text) for m in split.development} & {key(m.text) for m in split.test}


def test_placeholder_tokens_never_occur_in_real_messages(prepared) -> None:
    texts = [m.text.lower() for m in prepared.split.development + prepared.split.test]
    assert not [t for t in PLACEHOLDER_TOKENS if any(t in text for text in texts)]


def test_split_fingerprint_matches_the_release(prepared) -> None:
    metadata = json.loads((RELEASE_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert prepared.split.fingerprint() == metadata["dataset"]["split_fingerprint"]
    assert prepared.load_report.sha256 == metadata["dataset"]["sha256"]


@pytest.mark.slow
def test_release_artifact_reproduces_byte_for_byte(tmp_path, prepared) -> None:
    from spam_detector.artifact import sha256_file
    from spam_detector.training.release import build_release, resolve_candidate

    config = PipelineConfig.load()
    spec = resolve_candidate(config, PROJECT_ROOT / "reports/experiments/cv_results.json")
    rebuilt = build_release(config, prepared, spec, tmp_path / "rebuilt")
    assert sha256_file(rebuilt / "model.json") == sha256_file(RELEASE_DIR / "model.json")


def test_legacy_baseline_is_reproduced_from_its_preserved_pickles() -> None:
    nltk_data = PROJECT_ROOT / "nltk_data"
    if not nltk_data.is_dir():
        pytest.skip("NLTK data not prepared (python scripts/prepare_nltk.py)")
    from spam_detector.training.legacy_audit import audit_legacy_protocol

    config = PipelineConfig.load()
    result = audit_legacy_protocol(config.resolve(config.data["path"]), PROJECT_ROOT / "models/legacy-v1",
                                   config.data["near_duplicate_jaccard"])
    committed = json.loads((PROJECT_ROOT / "reports/experiments/legacy_protocol_audit.json").read_text(encoding="utf-8"))
    assert result == committed
    # The v1 README/notebook figures: accuracy 0.98, spam F1 0.94 on the v1 holdout.
    assert round(result["v1_artifact_on_v1_holdout"]["spam_f1"], 2) == 0.94
