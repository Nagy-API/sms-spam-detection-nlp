"""The whole offline pipeline on a synthetic corpus, so CI covers it without the real dataset."""

from __future__ import annotations

import random
from pathlib import Path

from spam_detector.config import PipelineConfig
from spam_detector.inference import SpamClassifier
from spam_detector.training.experiment import prepare_data, run_experiments
from spam_detector.training.release import build_release, evaluate_locked_test, resolve_candidate
from spam_detector.training.report import cv_markdown, locked_test_markdown

SPAM_WORDS = "win winner prize claim free cash urgent txt reward bonus now £ ! guaranteed award entry".split()
HAM_WORDS = "see you at lunch home later dinner tonight thanks call me when ok sure mum work bus late".split()


def _write_corpus(path: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    lines = ["v1,v2,,,"]
    for i in range(240):
        if i % 5 == 0:
            words = " ".join(rng.choice(SPAM_WORDS) for _ in range(rng.randint(4, 9)))
            text = f"{words} call 0906{rng.randint(1000000, 9999999)}"
        else:
            text = " ".join(rng.choice(HAM_WORDS) for _ in range(rng.randint(3, 9))) + f" {i}"
        lines.append(f'{"spam" if i % 5 == 0 else "ham"},"{text}",,,')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config(tmp_path: Path) -> PipelineConfig:
    data = tmp_path / "synthetic.csv"
    _write_corpus(data)
    config = PipelineConfig.load()
    raw = dict(config.raw)
    raw["data"] = {"path": str(data), "near_duplicate_jaccard": 0.5}
    raw["cross_validation"] = {"folds": 3, "repeats": 1, "seed": 1}
    raw["candidates"] = [c for c in raw["candidates"] if c["name"] in {"nb_word12_counts", "lr_word12"}]
    raw["release"] = {**raw["release"], "candidate": "auto"}
    return PipelineConfig(raw, config.path)


def test_pipeline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    prepared = prepare_data(config)
    assert prepared.load_report.rows == 240
    assert not set(prepared.split.development_groups) & set(prepared.split.test_groups)

    results = run_experiments(config, prepared)
    assert {c["name"] for c in results["candidates"]} == {"nb_word12_counts", "lr_word12"}
    assert results["selection"]["selected"] in {"nb_word12_counts", "lr_word12"}
    assert "Cross-validation results" in cv_markdown(results)

    cv_path = tmp_path / "cv.json"
    import json

    cv_path.write_text(json.dumps(results), encoding="utf-8")
    spec = resolve_candidate(config, cv_path)
    artifact_dir = build_release(config, prepared, spec, tmp_path / "artifact")

    classifier, _ = SpamClassifier.load(artifact_dir)
    assert classifier.predict("WINNER! claim your £500 prize, call 09061234567").label == "spam"
    assert classifier.predict("see you at dinner tonight mum").label == "ham"

    report = evaluate_locked_test(config, prepared, artifact_dir, tmp_path / "log.jsonl", include_baseline=False)
    assert report["selected_model"]["metrics"]["spam_recall"] > 0.9
    assert (tmp_path / "log.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert "Locked test-set evaluation" in locked_test_markdown(report)
