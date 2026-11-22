"""Offline TRACE v3 acquisition, source audit, and development benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from spam_detector.config import PipelineConfig
from spam_detector.data import Message
from spam_detector.inference import SpamClassifier
from spam_detector.text import URL_TOKEN, letter_profile, normalize
from spam_detector.training.dedup import canonical_key, near_duplicate_groups
from spam_detector.training.experiment import prepare_data

ROOT = Path("data/trace-v3")
LABELS = ("ham", "spam", "phishing", "inconclusive")
SOURCES = {
    "ncsu": (
        "https://raw.githubusercontent.com/wspr-ncsu/sms-phishing/a4f18cedc9909cbf8234eea3c2028080a1e95cd4/phishing_messages.csv",
        "d125c394af792faeb2b71b3f7100cad3b4cece02aff78cb4ec1fb9e7db90f230",
        "MIT (repository)",
    ),
    "imc": (
        "https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/a6175560b57387199871e51fbef6bc523d2516b4/dataset/final_dataset_output.csv",
        "1bbd1e9e82c3ea023112207b80da268a5c4a07d2353c2b0898360ab037fa9a64",
        "CC BY 4.0",
    ),
}
MAX_BYTES = 50 * 1024 * 1024


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")


def acquire() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum, _) in SOURCES.items():
        path = ROOT / f"{name}.csv"
        if path.exists():
            verify(path, checksum)
            continue
        with NamedTemporaryFile(dir=ROOT, delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            with urlopen(url, timeout=30) as response, staged.open("wb") as output:
                if response.geturl() != url:
                    raise ValueError(f"Unexpected redirect for {name}")
                size = 0
                while chunk := response.read(65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"Download exceeds size limit for {name}")
                    output.write(chunk)
            verify(staged, checksum)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


@dataclass(frozen=True)
class Record:
    source: str
    label: str
    text: str


def read_source(name: str) -> tuple[list[Record], dict[str, object]]:
    url, checksum, license_name = SOURCES[name]
    path = ROOT / f"{name}.csv"
    verify(path, checksum)
    retained = []
    counts: Counter[str] = Counter()
    column = "message" if name == "ncsu" else "text"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        # The pinned NCSU export appends one unnamed field to every row.
        rows = csv.DictReader(stream, restkey="extra", strict=True)
        required = {column} if name == "ncsu" else {column, "language", "scam_type"}
        if not required.issubset(rows.fieldnames or ()):
            raise ValueError(f"Missing expected columns in {name}")
        for row in rows:
            counts["raw"] += 1
            extra = row.get("extra", [])
            if len(extra) != (1 if name == "ncsu" else 0) or any(row[key] is None for key in required):
                raise ValueError(f"Malformed CSV in {name} at row {rows.line_num}")
            text = row[column].strip()
            if name == "imc" and (row["language"] != "English" or row["scam_type"] in {"", "spam"}):
                counts["excluded_source_label_or_language"] += 1
                continue
            letters, latin = letter_profile(text)
            if len(text) > 5000 or not letters or latin / letters < 0.5:
                counts["excluded_length_or_script"] += 1
                continue
            retained.append(Record(name, "phishing", text))
    counts["eligible"] = len(retained)
    return retained, {"url": url, "sha256": checksum, "license": license_name, "counts": dict(counts)}


def cluster(records: list[Record]) -> tuple[list[tuple[Record, int]], dict[str, int]]:
    by_key: dict[str, list[Record]] = defaultdict(list)
    for row in records:
        by_key[canonical_key(normalize(row.text))].append(row)

    # Group every distinct key before removing cross-source duplicates, so they can
    # still form transitive links that would otherwise cross a split.
    buckets = list(by_key.values())
    messages = [Message(i, rows[0].label, normalize(rows[0].text)) for i, rows in enumerate(buckets)]
    groups, _, _ = near_duplicate_groups(messages, 0.5, batch_size=256)
    source_sets: dict[int, set[str]] = defaultdict(set)
    for group, rows in zip(groups, buckets, strict=True):
        source_sets[group].update(row.source for row in rows)

    stats = Counter(
        normalized_duplicates=len(records) - len(buckets),
        cross_source_groups=sum(len(sources) > 1 for sources in source_sets.values()),
        groups=len(source_sets),
    )
    output = []
    for group, rows in zip(groups, buckets, strict=True):
        if len({row.label for row in rows}) > 1:
            stats["conflicting_rows"] += len(rows)
        elif any(source.endswith("_test") for source in source_sets[group]):
            stats["quarantined_keys"] += 1
        else:
            output.append((rows[0], group))
    return output, dict(stats)


def development_benchmark() -> dict[str, object]:
    prepared = prepare_data(PipelineConfig.load())
    historical = [Record("uci_dev", m.label, m.text) for m in prepared.split.development]
    historical_test = [Record("uci_test", m.label, m.text) for m in prepared.split.test]
    phishing, ncsu = read_source("ncsu")
    imc, imc_manifest = read_source("imc")
    independent = [Record("imc_test", row.label, row.text) for row in imc]
    rows, audit = cluster(historical + historical_test + phishing + independent)
    texts = np.asarray([normalize(row.text) for row, _ in rows])
    labels = np.asarray([row.label for row, _ in rows])
    groups = np.asarray([group for _, group in rows])
    if set(labels) != set(LABELS[:3]):
        raise ValueError("Development data lacks one or more classes after overlap quarantine")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261122)
    train, validation = next(splitter.split(texts, labels, groups))
    if set(labels[train]) != set(LABELS[:3]) or set(labels[validation]) != set(LABELS[:3]):
        raise ValueError("Grouped development split lacks one or more classes")
    if set(groups[train]) & set(groups[validation]):
        raise AssertionError("A campaign group crosses development partitions")

    v2, _ = SpamClassifier.load(Path("models/spam-v2"))
    v2_predictions = [v2.predict(rows[i][0].text).label for i in validation]
    scores = {"v2": confusion_matrix(labels[validation], v2_predictions, labels=LABELS).tolist()}
    url_mask = np.asarray([URL_TOKEN in value.split() for value in texts])
    without_urls = np.asarray([" ".join(token for token in value.split() if token != URL_TOKEN) for value in texts])
    for name, analyzer, feature_range, inputs in (
        ("word", "word", (1, 2), texts),
        ("char_wb", "char_wb", (3, 5), texts),
        ("word_without_url_token", "word", (1, 2), without_urls),
    ):
        estimator = make_pipeline(
            TfidfVectorizer(analyzer=analyzer, ngram_range=feature_range, min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=300, class_weight="balanced", random_state=20261122),
        )
        estimator.fit(inputs[train], labels[train])
        predictions = estimator.predict(inputs[validation])
        scores[name] = confusion_matrix(labels[validation], predictions, labels=LABELS).tolist()

    return {
        "protocol": "development-only; source and class are confounded; not a release evaluation",
        "sources": {"ncsu": ncsu, "imc": imc_manifest, "historical_split": prepared.split.fingerprint()},
        "overlap": audit,
        "train_counts": dict(Counter(labels[train])),
        "validation_counts": dict(Counter(labels[validation])),
        "validation_group_count": len(set(groups[validation])),
        "validation_url_counts": dict(Counter(labels[validation[url_mask[validation]]])),
        "labels": LABELS,
        "development_confusion_matrices": scores,
        "release_gate": "blocked: independent, verified modern English benign cohort missing",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "benchmark"))
    args = parser.parse_args()
    if args.command == "acquire":
        acquire()
    else:
        print(json.dumps(development_benchmark(), indent=2))


if __name__ == "__main__":
    main()
