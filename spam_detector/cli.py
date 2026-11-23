"""Command-line interface: ``python -m spam_detector <command>``.

Offline commands (need ``requirements-train.txt`` and ``data/spam.csv``)::

    data            validate, repair, deduplicate, group and split; write the dataset report
    audit-legacy    reproduce the v1 split and quantify its near-duplicate leakage
    experiment      repeated group-aware CV of every candidate on the development split
    release         build the selected model's artifact (refuses to overwrite)
    evaluate        one-time scoring of an artifact on the locked test set
    evaluate-csv    score an artifact on any external labeled CSV (label,text columns)
    reproduce       rebuild the release in a temporary directory and compare checksums

Runtime commands (standard library only)::

    verify          check an artifact's checksum, schema, and golden predictions
    predict         classify a message
    explain         show the features that drove a prediction
    benchmark       measure load time, latency and memory of an artifact
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from spam_detector.config import DEFAULT_CONFIG, PROJECT_ROOT, PipelineConfig

REPORTS = PROJECT_ROOT / "reports"
DEFAULT_ARTIFACT = PROJECT_ROOT / "models" / "spam-v2"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_predictions.json"


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_data(args: argparse.Namespace) -> None:
    from spam_detector.training.experiment import prepare_data, write_json

    prepared = prepare_data(PipelineConfig.load(args.config))
    write_json(REPORTS / "data" / "dataset_report.json", prepared.report())
    _print(prepared.report())


def cmd_audit_legacy(args: argparse.Namespace) -> None:
    from spam_detector.training.experiment import write_json
    from spam_detector.training.legacy_audit import audit_legacy_protocol

    config = PipelineConfig.load(args.config)
    result = audit_legacy_protocol(
        config.resolve(config.data["path"]), PROJECT_ROOT / "models" / "legacy-v1",
        config.data["near_duplicate_jaccard"],
    )
    write_json(REPORTS / "experiments" / "legacy_protocol_audit.json", result)
    _print(result)


def cmd_experiment(args: argparse.Namespace) -> None:
    from spam_detector.training.experiment import prepare_data, run_experiments, write_json
    from spam_detector.training.report import cv_markdown

    config = PipelineConfig.load(args.config)
    results = run_experiments(config, prepare_data(config), args.only)
    suffix = "" if not args.only else "_partial"
    write_json(REPORTS / "experiments" / f"cv_results{suffix}.json", results)
    (REPORTS / "experiments" / f"cv_results{suffix}.md").write_text(cv_markdown(results), encoding="utf-8")
    _print(results["selection"])


def cmd_release(args: argparse.Namespace) -> None:
    from spam_detector.training.experiment import prepare_data
    from spam_detector.training.release import build_release, resolve_candidate

    config = PipelineConfig.load(args.config)
    spec = resolve_candidate(config, REPORTS / "experiments" / "cv_results.json")
    output = Path(args.output_dir) if args.output_dir else config.resolve(config.release["output_dir"])
    build_release(config, prepare_data(config), spec, output)
    print(f"wrote {output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    if not args.i_understand_this_uses_the_locked_test_set:
        sys.exit(
            "Refusing: the locked test set should be scored once, after model selection is final.\n"
            "Re-run with --i-understand-this-uses-the-locked-test-set to proceed (the run is logged)."
        )
    from spam_detector.training.experiment import prepare_data, write_json
    from spam_detector.training.release import evaluate_locked_test
    from spam_detector.training.report import locked_test_markdown

    config = PipelineConfig.load(args.config)
    report = evaluate_locked_test(
        config, prepare_data(config), Path(args.artifact), REPORTS / "evaluation" / "test_access_log.jsonl"
    )
    write_json(REPORTS / "evaluation" / "test_report.json", report)
    (REPORTS / "evaluation" / "test_report.md").write_text(locked_test_markdown(report), encoding="utf-8")
    _print(report["selected_model"]["metrics"])


def cmd_evaluate_csv(args: argparse.Namespace) -> None:
    import csv

    import numpy as np

    from spam_detector.inference import SpamClassifier
    from spam_detector.training.metrics import bootstrap_intervals, classification_metrics

    classifier, loaded = SpamClassifier.load(Path(args.artifact))
    with open(args.data, encoding=args.encoding, newline="") as handle:
        rows = [(r[args.label_column].strip().lower(), r[args.text_column]) for r in csv.DictReader(handle)]
    y = np.array([int(label in {"spam", "1", "smishing"}) for label, _ in rows])
    predictions = [classifier.predict(text) for _, text in rows]
    predicted = np.array([int(p.label == "spam") for p in predictions])
    probability = np.array([p.spam_probability or 0.0 for p in predictions])
    _print({
        "artifact_sha256": loaded.sha256,
        "messages": len(rows),
        "inconclusive": sum(p.label == "inconclusive" for p in predictions),
        "metrics": classification_metrics(y, predicted, probability, probability),
        "bootstrap_95ci": bootstrap_intervals(y, predicted, probability),
    })


def cmd_reproduce(args: argparse.Namespace) -> None:
    from spam_detector.artifact import sha256_file
    from spam_detector.training.experiment import prepare_data
    from spam_detector.training.release import build_release, resolve_candidate

    config = PipelineConfig.load(args.config)
    committed = Path(args.artifact)
    spec = resolve_candidate(config, REPORTS / "experiments" / "cv_results.json")
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = build_release(config, prepare_data(config), spec, Path(directory) / "artifact")
        new, old = sha256_file(rebuilt / "model.json"), sha256_file(committed / "model.json")
    print(f"committed model.json sha256: {old}\nrebuilt   model.json sha256: {new}")
    if new != old:
        sys.exit("NOT reproducible: the rebuilt artifact differs from the committed one")
    print("Reproduced byte-for-byte.")


def cmd_verify(args: argparse.Namespace) -> None:
    from spam_detector.inference import SpamClassifier

    classifier, loaded = SpamClassifier.load(Path(args.artifact))
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if golden["model_sha256"] != loaded.sha256:
        sys.exit(f"golden file is for {golden['model_sha256']}, artifact is {loaded.sha256}")
    failures = []
    for case in golden["cases"]:
        result = classifier.predict(case["text"])
        probability_ok = (result.spam_probability is None) == (case["spam_probability"] is None) and (
            result.spam_probability is None or abs(result.spam_probability - case["spam_probability"]) < 1e-9
        )
        if result.label != case["label"] or result.reason != case["reason"] or not probability_ok:
            failures.append({"text": case["text"], "expected": case, "got": result.__dict__})
    if failures:
        _print(failures)
        sys.exit(f"{len(failures)} golden predictions changed")
    print(f"OK: checksum {loaded.sha256[:12]}…, {len(golden['cases'])} golden predictions match")


def cmd_predict(args: argparse.Namespace) -> None:
    from spam_detector.inference import SpamClassifier

    classifier, _ = SpamClassifier.load(Path(args.artifact))
    _print(classifier.predict(args.text).__dict__)


def cmd_explain(args: argparse.Namespace) -> None:
    from spam_detector.inference import SpamClassifier
    from spam_detector.text import normalize

    classifier, _ = SpamClassifier.load(Path(args.artifact))
    _print({
        "normalized": normalize(args.text),
        "prediction": classifier.predict(args.text).__dict__,
        "intercept": classifier.artifact.intercept,
        "top_contributions": [c.__dict__ for c in classifier.explain(args.text, args.top)],
    })


def cmd_benchmark(args: argparse.Namespace) -> None:
    from spam_detector.benchmark import benchmark_artifact

    _print(benchmark_artifact(Path(args.artifact), args.repeats))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m spam_detector", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    add("data", cmd_data, "prepare and report on the dataset")
    add("audit-legacy", cmd_audit_legacy, "quantify leakage in the v1 protocol")
    add("experiment", cmd_experiment, "cross-validate candidates").add_argument("--only", nargs="*")
    add("release", cmd_release, "build the release artifact").add_argument("--output-dir")
    evaluate = add("evaluate", cmd_evaluate, "score the locked test set once")
    evaluate.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    evaluate.add_argument("--i-understand-this-uses-the-locked-test-set", action="store_true")
    external = add("evaluate-csv", cmd_evaluate_csv, "score an external labeled CSV")
    external.add_argument("data")
    external.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    external.add_argument("--label-column", default="label")
    external.add_argument("--text-column", default="text")
    external.add_argument("--encoding", default="utf-8")
    add("reproduce", cmd_reproduce, "rebuild and compare checksums").add_argument(
        "--artifact", default=str(DEFAULT_ARTIFACT))
    add("verify", cmd_verify, "verify artifact integrity").add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    for name, handler in (("predict", cmd_predict), ("explain", cmd_explain)):
        sub = add(name, handler, f"{name} one message")
        sub.add_argument("text")
        sub.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
        if name == "explain":
            sub.add_argument("--top", type=int, default=10)
    bench = add("benchmark", cmd_benchmark, "measure latency and memory")
    bench.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    bench.add_argument("--repeats", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
