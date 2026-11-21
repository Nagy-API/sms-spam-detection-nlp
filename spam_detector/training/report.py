"""Render experiment and evaluation JSON as Markdown tables for humans and pull requests."""

from __future__ import annotations

from typing import Any


def _pm(stat: dict[str, float], digits: int = 4) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def cv_markdown(results: dict[str, Any]) -> str:
    protocol = results["protocol"]
    selection = results["selection"]
    lines = [
        "# Cross-validation results (development split only)",
        "",
        f"{protocol['folds']}-fold stratified **group** CV × {protocol['repeats']} repeats "
        f"({protocol['folds'] * protocol['repeats']} folds). Mean ± standard deviation across folds.",
        f"Calibrated view: cross-fitted Platt scaling, threshold {protocol['decision_threshold']:.2f} "
        f"(C_FP={protocol['false_positive_cost']:g}, C_FN={protocol['false_negative_cost']:g}).",
        "",
        "## Deployment operating point (calibrated, cost threshold)",
        "",
        "| Candidate | Expected cost / msg | Spam precision | Spam recall | Spam F1 | FPR | PR-AUC | Brier | ECE | Features |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(results["candidates"], key=lambda r: r["calibrated"]["expected_cost_per_message"]["mean"])
    for r in ordered:
        c = r["calibrated"]
        marker = " **(selected)**" if r["name"] == selection["selected"] else ""
        features = f"{r['n_features_mean']:,.0f}" if r["n_features_mean"] else "n/a"
        lines.append(
            f"| `{r['name']}`{marker} | {_pm(c['expected_cost_per_message'])} | {_pm(c['spam_precision'], 3)} | "
            f"{_pm(c['spam_recall'], 3)} | {_pm(c['spam_f1'], 3)} | {_pm(c['false_positive_rate'], 4)} | "
            f"{_pm(c['pr_auc'], 3)} | {_pm(c['brier'])} | {_pm(c['ece'])} | {features} |"
        )
    lines += [
        "",
        "## Each classifier's own decision rule (score ≥ 0, uncalibrated)",
        "",
        "| Candidate | Accuracy | Spam precision | Spam recall | Spam F1 | Macro F1 | Balanced acc. | MCC | Fit s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ordered:
        d = r["at_default"]
        lines.append(
            f"| `{r['name']}` | {_pm(d['accuracy'], 3)} | {_pm(d['spam_precision'], 3)} | {_pm(d['spam_recall'], 3)} | "
            f"{_pm(d['spam_f1'], 3)} | {_pm(d['macro_f1'], 3)} | {_pm(d['balanced_accuracy'], 3)} | "
            f"{_pm(d['mcc'], 3)} | {r['fit_seconds_mean']:.2f} |"
        )
    lines += [
        "",
        "## Selection",
        "",
        f"Rule: {selection['rule']} on `{selection['metric']}` (lower is better), exportable candidates only.",
        f"Best mean: `{selection['best_mean']}`. Within one standard error: "
        + ", ".join(f"`{n}`" for n in selection["tied_candidates"])
        + f". Selected (fewest features among tied): **`{selection['selected']}`**.",
        "",
    ]
    return "\n".join(lines)


def _metric_row(name: str, m: dict[str, Any], ci: dict[str, list[float]]) -> str:
    def cell(key: str) -> str:
        value = f"{m[key]:.4f}"
        if key in ci:
            value += f" [{ci[key][0]:.3f}, {ci[key][1]:.3f}]"
        return value

    tn, fp = m["confusion_matrix"][0]
    fn, tp = m["confusion_matrix"][1]
    return (
        f"| {name} | {cell('accuracy')} | {cell('spam_precision')} | {cell('spam_recall')} | {cell('spam_f1')} | "
        f"{m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} | {cell('pr_auc')} | {fp} | {fn} | "
        f"TN={tn} FP={fp} FN={fn} TP={tp} |"
    )


def locked_test_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_model"]
    lines = [
        "# Locked test-set evaluation",
        "",
        f"Artifact `{report['artifact']['directory']}` (v{report['artifact']['model_version']}, "
        f"sha256 `{report['artifact']['sha256'][:12]}…`), evaluated {report['evaluated_at']}.",
        f"Test set: {report['test_set']['messages']} messages, {report['test_set']['spam']} spam, "
        f"{report['test_set']['groups']} near-duplicate groups (none shared with development).",
        "",
        f"> {report['caveat']}",
        "",
        "Brackets: 95% percentile bootstrap intervals (2,000 resamples).",
        "",
        "| System | Accuracy | Spam precision | Spam recall | Spam F1 | Macro F1 | Balanced acc. | PR-AUC | FP | FN | Confusion |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        _metric_row(f"v2 selected (threshold {selected['decision_threshold']:.2f})", selected["metrics"],
                    selected["bootstrap_95ci"]),
    ]
    if "legacy_baseline" in report:
        legacy = report["legacy_baseline"]
        lines.append(_metric_row("v1 recipe (retrained, same split)", legacy["metrics"], legacy["bootstrap_95ci"]))
    m = selected["metrics"]
    lines += [
        "",
        f"Calibration of the selected model: Brier {m['brier']:.4f}, log loss {m['log_loss']:.4f}, "
        f"ECE {m['ece']:.4f}. Expected cost per message {m['expected_cost_per_message']:.4f}.",
        f"Coverage: {report['coverage']['predicted']} predicted, {report['coverage']['inconclusive']} inconclusive "
        f"({report['coverage']['inconclusive_reasons']}); inconclusive answers count as not-spam above.",
        "",
    ]
    return "\n".join(lines)
