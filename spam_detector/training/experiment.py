"""Dataset preparation, repeated group-aware cross-validation, and model selection.

Only the *development* split is touched here. The locked test set is never
passed to anything in this module (see :mod:`spam_detector.training.release`).

For each candidate and each CV fold we record two views:

``at_default``
    The classifier's own decision rule (score >= 0). Shows how the raw model
    behaves before any post-processing.
``calibrated``
    The deployment procedure: Platt-calibrate scores and apply the cost-derived
    threshold. To keep this honest, the calibrator for fold *k* is fitted on
    the out-of-fold scores of the *other* folds of the same repeat ("cross-
    fitting"), so fold *k*'s labels never influence its own calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
import sklearn

from spam_detector.config import PROJECT_ROOT, PipelineConfig
from spam_detector.data import LoadReport, load_messages
from spam_detector.training.candidates import AnalysisCache, CandidateSpec, FittedLinearModel, fit_platt, sigmoid
from spam_detector.training.dedup import DedupReport, deduplicate_and_group
from spam_detector.training.metrics import bayes_threshold, classification_metrics
from spam_detector.training.splits import DatasetSplit, cross_validation_folds, split_development_test


@dataclass
class PreparedData:
    split: DatasetSplit
    load_report: LoadReport
    dedup_report: DedupReport

    def report(self) -> dict[str, Any]:
        return {
            "source": self.load_report.to_dict(),
            "deduplication": self.dedup_report.to_dict(),
            "split": self.split.summary(),
        }


def prepare_data(config: PipelineConfig) -> PreparedData:
    messages, load_report = load_messages(config.resolve(config.data["path"]), config.data.get("sha256"))
    unique, groups, dedup_report = deduplicate_and_group(messages, config.data["near_duplicate_jaccard"])
    split = split_development_test(unique, groups, config.split["test_fraction"], config.split["seed"])
    return PreparedData(split, load_report, dedup_report)


def environment_info(config: PipelineConfig) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "spam_detector", "configs"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_commit": commit,
        "git_dirty_code_or_config": dirty,
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
    }


def _summarize(per_fold: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key, value in per_fold[0].items():
        if isinstance(value, list):
            continue
        values = np.array([fold[key] for fold in per_fold], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": std, "sem": std / math.sqrt(len(values))}
    return summary


def cross_validate_candidate(
    spec: CandidateSpec,
    texts: Sequence[str],
    labels: np.ndarray,
    folds: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    threshold: float,
    costs: dict[str, float],
    cache: AnalysisCache,
) -> dict[str, Any]:
    repeats = max(r for r, *_ in folds) + 1
    out_of_fold = np.full((repeats, len(texts)), np.nan)
    default_metrics, feature_counts, fit_seconds = [], [], []
    text_array = np.asarray(texts, dtype=object)

    for repeat, _fold, train, validation in folds:
        start = time.perf_counter()
        model = spec.fit(list(text_array[train]), labels[train], cache)
        fit_seconds.append(time.perf_counter() - start)
        if isinstance(model, FittedLinearModel):
            feature_counts.append(model.space.size)
        scores = model.scores(list(text_array[validation]))
        out_of_fold[repeat, validation] = scores
        default_metrics.append(
            classification_metrics(labels[validation], (scores >= 0).astype(int), scores, **costs)
        )

    calibrated_metrics = []
    for repeat, _fold, _train, validation in folds:
        others = np.ones(len(texts), dtype=bool)
        others[validation] = False
        a, b = fit_platt(out_of_fold[repeat, others], labels[others])
        probability = sigmoid(a * out_of_fold[repeat, validation] + b)
        calibrated_metrics.append(
            classification_metrics(
                labels[validation], (probability >= threshold).astype(int),
                out_of_fold[repeat, validation], probability, **costs,
            )
        )
    return {
        "name": spec.name,
        "classifier": spec.classifier,
        "description": spec.description,
        "exportable": spec.exportable,
        "n_features_mean": float(np.mean(feature_counts)) if feature_counts else None,
        "fit_seconds_mean": float(np.mean(fit_seconds)),
        "at_default": _summarize(default_metrics),
        "calibrated": _summarize(calibrated_metrics),
    }


def select_candidate(results: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """One-standard-error rule over exportable candidates (lower metric is better).

    Picking the single best mean over ~10 candidates rewards noise. Instead,
    every candidate within one standard error of the best is treated as tied,
    and the tie is broken by deployment cost (fewest features). The standard
    error comes from repeated CV folds, which are correlated, so it is an
    optimistic estimate of the true uncertainty: a tie-break, not a test.
    """
    eligible = [r for r in results if r["exportable"]]
    best = min(eligible, key=lambda r: r["calibrated"][metric]["mean"])
    limit = best["calibrated"][metric]["mean"] + best["calibrated"][metric]["sem"]
    tied = [r for r in eligible if r["calibrated"][metric]["mean"] <= limit]
    chosen = min(tied, key=lambda r: (r["n_features_mean"], r["calibrated"][metric]["mean"]))
    return {
        "metric": metric,
        "rule": "one_standard_error",
        "best_mean": best["name"],
        "tolerance_upper_bound": limit,
        "tied_candidates": [r["name"] for r in tied],
        "selected": chosen["name"],
    }


def run_experiments(config: PipelineConfig, prepared: PreparedData, only: Sequence[str] | None = None) -> dict[str, Any]:
    development = prepared.split.development
    texts = [m.text for m in development]
    labels = np.array([m.is_spam for m in development])
    cv = config.cross_validation
    folds = list(cross_validation_folds(labels, prepared.split.development_groups, cv["folds"], cv["repeats"], cv["seed"]))
    costs = {
        "false_positive_cost": config.decision["false_positive_cost"],
        "false_negative_cost": config.decision["false_negative_cost"],
    }
    threshold = bayes_threshold(**costs)
    cache = AnalysisCache()
    results = []
    for raw in config.candidates:
        spec = CandidateSpec.from_dict(raw)
        if only and spec.name not in only:
            continue
        start = time.perf_counter()
        result = cross_validate_candidate(spec, texts, labels, folds, threshold, costs, cache)
        print(
            f"{spec.name:32s} cost={result['calibrated']['expected_cost_per_message']['mean']:.4f} "
            f"F1={result['calibrated']['spam_f1']['mean']:.4f} PR-AUC={result['calibrated']['pr_auc']['mean']:.4f} "
            f"({time.perf_counter() - start:.0f}s)",
            flush=True,
        )
        results.append(result)
    return {
        "protocol": {
            "evaluated_on": "development split only (locked test set untouched)",
            "folds": cv["folds"],
            "repeats": cv["repeats"],
            "group_aware": True,
            "decision_threshold": threshold,
            **costs,
        },
        "data": prepared.report(),
        "environment": environment_info(config),
        "candidates": results,
        "selection": select_candidate(results, config.selection["metric"]),
    }


def write_json(path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
