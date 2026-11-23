"""Copy headline CV and locked-test metrics into models/<release>/metadata.json.

metadata.json is evidence, not model state: model.json (and its checksum) is unchanged.
The summary is what GET /model returns to clients.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("accuracy", "spam_precision", "spam_recall", "spam_f1", "false_positive_rate", "pr_auc", "brier")


def main(release: str = "models/spam-v2") -> None:
    meta_path = ROOT / release / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cv = json.loads((ROOT / "reports/experiments/cv_results.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "reports/evaluation/test_report.json").read_text(encoding="utf-8"))
    if test["artifact"]["sha256"] != meta["model_sha256"]:
        raise SystemExit("test report was produced for a different artifact")
    calibrated = next(c for c in cv["candidates"] if c["name"] == meta["candidate"]["name"])["calibrated"]
    metrics, ci = test["selected_model"]["metrics"], test["selected_model"]["bootstrap_95ci"]
    meta["evaluation_summary"] = {
        "cross_validation": {
            "protocol": "5-fold stratified group CV x 3 repeats on the development split (4,125 messages)",
            **{k: {"mean": round(calibrated[k]["mean"], 4), "std": round(calibrated[k]["std"], 4)}
               for k in KEYS if k != "accuracy"},
        },
        "locked_test": {
            "protocol": "held-out 1,031 messages (128 spam); no near-duplicate group shared with training",
            **{k: round(metrics[k], 4) for k in KEYS},
            "bootstrap_95ci": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in ci.items()},
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "caveat": test["caveat"],
        "reports": ["reports/experiments/cv_results.md", "reports/evaluation/test_report.md"],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
