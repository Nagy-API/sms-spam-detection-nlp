"""Build the release artifact and run the one-time locked-test evaluation.

Release procedure (all on the development split):

1. Re-run 5-fold group CV for the selected candidate to get out-of-fold scores.
2. Fit Platt calibration on those out-of-fold scores (never on training-fit
   scores, which are over-confident).
3. Fit the final model on the whole development split.
4. Export to JSON, reload it with the pure-Python runtime, and assert its
   scores match scikit-learn's to 1e-9 on every development message.

Locked-test evaluation scores the exported artifact (exactly what production
serves) plus the legacy baseline retrained under the same protocol. Every run
is appended to ``reports/evaluation/test_access_log.jsonl`` so repeated
"peeking" at the test set is visible in version control.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from spam_detector.artifact import (
    AbstentionPolicy,
    LinearModelArtifact,
    PlattCalibration,
    load_artifact,
    save_artifact,
)
from spam_detector.config import PipelineConfig
from spam_detector.inference import SpamClassifier
from spam_detector.training.candidates import AnalysisCache, CandidateSpec, FittedLinearModel, fit_platt
from spam_detector.training.experiment import PreparedData, environment_info
from spam_detector.training.metrics import bayes_threshold, bootstrap_intervals, classification_metrics
from spam_detector.training.splits import cross_validation_folds

PARITY_TOLERANCE = 1e-9


def resolve_candidate(config: PipelineConfig, cv_results_path: Path) -> CandidateSpec:
    name = config.release["candidate"]
    if name == "auto":
        name = json.loads(cv_results_path.read_text(encoding="utf-8"))["selection"]["selected"]
    for raw in config.candidates:
        if raw["name"] == name:
            spec = CandidateSpec.from_dict(raw)
            if not spec.exportable:
                raise ValueError(f"candidate {name} cannot be exported as a linear artifact")
            return spec
    raise KeyError(f"candidate {name!r} not found in {config.path}")


def build_release(config: PipelineConfig, prepared: PreparedData, spec: CandidateSpec, output_dir: Path) -> Path:
    development = prepared.split.development
    texts = [m.text for m in development]
    labels = np.array([m.is_spam for m in development])
    cache = AnalysisCache()

    # 1-2. Out-of-fold scores -> Platt calibration.
    cv = config.cross_validation
    out_of_fold = np.full(len(texts), np.nan)
    for _, _, train, validation in cross_validation_folds(
        labels, prepared.split.development_groups, cv["folds"], 1, cv["seed"]
    ):
        model = spec.fit([texts[i] for i in train], labels[train], cache)
        out_of_fold[validation] = model.scores([texts[i] for i in validation])
    a, b = fit_platt(out_of_fold, labels)

    # 3. Final fit on all development data.
    start = time.perf_counter()
    final = spec.fit(texts, labels, cache)
    fit_seconds = time.perf_counter() - start
    assert isinstance(final, FittedLinearModel)
    threshold = bayes_threshold(config.decision["false_positive_cost"], config.decision["false_negative_cost"])
    artifact = LinearModelArtifact(
        model_version=config.release["model_version"],
        estimator=f"{spec.classifier}:{spec.name}",
        feature_space=final.space,
        coefficients=[float(c) for c in final.coefficients],
        intercept=float(final.intercept),
        calibration=PlattCalibration(a, b),
        threshold=threshold,
        abstention=AbstentionPolicy(config.release["min_latin_letter_ratio"]),
    )

    # 4. Round-trip through JSON and compare with scikit-learn on every message.
    runtime = SpamClassifier(LinearModelArtifact.from_dict(json.loads(artifact.to_json_bytes())))
    reference = final.scores(texts)
    exported = np.array([runtime.score_vector(runtime.vectorize(t)) for t in texts])
    max_difference = float(np.max(np.abs(reference - exported)))
    if max_difference > PARITY_TOLERANCE:
        raise AssertionError(f"exported model diverges from scikit-learn by {max_difference}")

    metadata = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "candidate": {"name": spec.name, "classifier": spec.classifier, "params": dict(spec.params),
                      "description": spec.description},
        "training": {
            "split": "development",
            "messages": len(texts),
            "spam": int(labels.sum()),
            "n_features": final.space.size,
            "fit_seconds": round(fit_seconds, 3),
        },
        "calibration": {"method": "platt", "fitted_on": f"{cv['folds']}-fold group CV out-of-fold scores",
                        "a": a, "b": b},
        "decision": {**config.decision, "threshold": threshold,
                     "rule": "flag as spam when calibrated P(spam) >= C_FP / (C_FP + C_FN)"},
        "parity_with_scikit_learn": {"messages": len(texts), "max_abs_score_difference": max_difference},
        "dataset": {"sha256": prepared.load_report.sha256, "split_fingerprint": prepared.split.fingerprint()},
        "environment": environment_info(config),
        "intended_use": "Screening English SMS for likely spam. Not a safety or fraud guarantee.",
    }
    save_artifact(artifact, output_dir, metadata)
    return output_dir


def _legacy_baseline_test_metrics(prepared: PreparedData, costs: dict[str, float]) -> dict[str, Any]:
    from spam_detector.training.baseline import LegacyBaseline

    development, test = prepared.split.development, prepared.split.test
    model = LegacyBaseline.fit([m.text for m in development], [m.is_spam for m in development])
    scores = model.scores([m.text for m in test])
    y = np.array([m.is_spam for m in test])
    predicted = (scores >= 0).astype(int)
    return {
        "description": "v1 recipe retrained on the development split, default NB decision (log-odds >= 0)",
        "metrics": classification_metrics(y, predicted, scores, **costs),
        "bootstrap_95ci": bootstrap_intervals(y, predicted, scores),
    }


def evaluate_locked_test(config: PipelineConfig, prepared: PreparedData, artifact_dir: Path,
                         log_path: Path, include_baseline: bool = True) -> dict[str, Any]:
    loaded = load_artifact(artifact_dir)
    classifier = SpamClassifier(loaded.artifact)
    test = prepared.split.test
    y = np.array([m.is_spam for m in test])
    predictions = [classifier.predict(m.text) for m in test]
    costs = {
        "false_positive_cost": config.decision["false_positive_cost"],
        "false_negative_cost": config.decision["false_negative_cost"],
    }
    covered = np.array([p.label != "inconclusive" for p in predictions])
    probability = np.array([p.spam_probability if p.spam_probability is not None else 0.0 for p in predictions])
    # Inconclusive answers are not flagged as spam, so for end-to-end metrics they count as "ham".
    predicted = np.array([int(p.label == "spam") for p in predictions])

    report: dict[str, Any] = {
        "evaluated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "artifact": {"directory": artifact_dir.as_posix(), "model_version": loaded.artifact.model_version,
                     "sha256": loaded.sha256},
        "test_set": prepared.split.summary()["test"],
        "split_fingerprint": prepared.split.fingerprint(),
        "caveat": (
            "Internal held-out evaluation. The test messages come from the same corpus, collection period and "
            "sources as the training data, and the original v1 holdout (which overlaps this test set) was seen "
            "during the v1 project. This is not an independent external evaluation."
        ),
        "coverage": {"predicted": int(covered.sum()), "inconclusive": int((~covered).sum()),
                     "inconclusive_reasons": {r: sum(p.reason == r for p in predictions)
                                              for r in {p.reason for p in predictions if p.reason}}},
        "selected_model": {
            "decision_threshold": loaded.artifact.threshold,
            "metrics": classification_metrics(y, predicted, probability, probability, **costs),
            "bootstrap_95ci": bootstrap_intervals(y, predicted, probability),
        },
    }
    if include_baseline:
        report["legacy_baseline"] = _legacy_baseline_test_metrics(prepared, costs)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"evaluated_at": report["evaluated_at"], "model_sha256": loaded.sha256,
                              "model_version": loaded.artifact.model_version,
                              "split_fingerprint": report["split_fingerprint"]}) + "\n")
    return report
