"""Evaluation metrics, calibration diagnostics, decision costs, and bootstrap intervals.

Spam is the positive class throughout. With ~13% spam, accuracy is a weak
signal (always answering "ham" scores ~87%), so the reports lead with spam
precision/recall/F1, balanced accuracy, PR-AUC (average precision), and the
expected decision cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Weighted mean |observed spam rate - mean predicted probability| over equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(error)


def classification_metrics(
    y_true: Sequence[int],
    predicted: Sequence[int],
    ranking_score: Sequence[float],
    probability: Sequence[float] | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> dict[str, float | list[list[int]]]:
    """Metrics for one set of predictions.

    ``ranking_score`` may be any monotone score (margin, log-odds, probability)
    and drives the threshold-free metrics. ``probability`` must be a calibrated
    P(spam) and drives Brier score, log loss and ECE; omit it for raw margins.
    """
    y = np.asarray(y_true)
    y_hat = np.asarray(predicted)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
    result: dict[str, float | list[list[int]]] = {
        "accuracy": float((tp + tn) / len(y)),
        "spam_precision": float(precision_score(y, y_hat, zero_division=0)),
        "spam_recall": float(recall_score(y, y_hat, zero_division=0)),
        "spam_f1": float(f1_score(y, y_hat, zero_division=0)),
        "macro_f1": float(f1_score(y, y_hat, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
        "mcc": float(matthews_corrcoef(y, y_hat)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "pr_auc": float(average_precision_score(y, ranking_score)),
        "roc_auc": float(roc_auc_score(y, ranking_score)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "expected_cost_per_message": float((false_positive_cost * fp + false_negative_cost * fn) / len(y)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if probability is not None:
        p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
        result["brier"] = float(brier_score_loss(y, p))
        result["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
        result["ece"] = expected_calibration_error(y, p)
    return result


def bootstrap_intervals(
    y_true: Sequence[int],
    predicted: Sequence[int],
    ranking_score: Sequence[float],
    metrics: Sequence[str] = ("accuracy", "spam_precision", "spam_recall", "spam_f1", "pr_auc"),
    resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile bootstrap confidence intervals (resampling messages with replacement).

    The test set has only ~130 spam messages, so a single point estimate hides
    a lot of uncertainty: one extra missed spam moves recall by ~0.8 points.
    """
    y, y_hat, s = np.asarray(y_true), np.asarray(predicted), np.asarray(ranking_score)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(resamples):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue  # a resample with one class has undefined recall/AUC
        values = classification_metrics(y[index], y_hat[index], s[index])
        for name in metrics:
            samples[name].append(float(values[name]))  # type: ignore[arg-type]
    tail = (1 - confidence) / 2 * 100
    return {
        name: [float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail))]
        for name, v in samples.items()
    }


def bayes_threshold(false_positive_cost: float, false_negative_cost: float) -> float:
    """Cost-minimizing threshold for a *calibrated* probability.

    Flag as spam when ``p * C_FN > (1 - p) * C_FP``, i.e. ``p > C_FP / (C_FP + C_FN)``.
    """
    return false_positive_cost / (false_positive_cost + false_negative_cost)
