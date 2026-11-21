from pathlib import Path

import pytest

from spam_detector.data import DataValidationError, LoadReport, load_messages, repair_encoding

# "£" double-encoded as in the Kaggle file: UTF-8 C2 A3 -> cp1252 "Â£" -> Mac Roman E5 A3.
POUND = b"\xe5\xa3"
# Right single quote: UTF-8 E2 80 99 -> cp1252 "â€™" -> Mac Roman 89 DB AA.
APOSTROPHE = b"\x89\xdb\xaa"


def _csv(tmp_path: Path, body: bytes) -> Path:
    path = tmp_path / "spam.csv"
    path.write_bytes(b"v1,v2,,,\n" + body)
    return path


def test_repair_reverses_double_encoding() -> None:
    report = LoadReport(sha256="x")
    assert repair_encoding(b"Win " + POUND + b"100, it" + APOSTROPHE + b"s yours", report) == "Win £100, it’s yours"
    assert report.mojibake_runs_repaired == 2 and report.mojibake_runs_unrepaired == 0


def test_unrepairable_bytes_are_kept_and_counted() -> None:
    report = LoadReport(sha256="x")
    assert repair_encoding(b"a\xccb", report) == "a" + b"\xcc".decode("mac_roman") + "b"
    assert report.mojibake_runs_unrepaired == 1


def test_load_rejoins_split_rows_and_unescapes_html(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        b'ham,"Hi there",,,\n'
        b'spam,Call now, PO Box 123, London,\n'
        b"ham,I'm late &lt;#&gt; mins &amp; sorry,,,\n",
    )
    messages, report = load_messages(path)
    assert [m.text for m in messages] == ["Hi there", "Call now, PO Box 123, London", "I'm late <#> mins & sorry"]
    assert report.rejoined_split_rows == 1
    assert report.rows_with_html_entities == 1
    assert dict(report.labels) == {"ham": 2, "spam": 1}
    assert [m.row_id for m in messages] == [0, 1, 2]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"maybe,hello,,,\nspam,win,,,\n", "invalid label"),
        (b"ham,   ,,,\nspam,win,,,\n", "empty message"),
        (b"ham,hello,,,\nham,hi,,,\n", "both classes"),
    ],
)
def test_schema_violations_are_rejected(tmp_path: Path, body: bytes, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_messages(_csv(tmp_path, body))


def test_bad_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "spam.csv"
    path.write_bytes(b"label,text\nham,hi\n")
    with pytest.raises(DataValidationError, match="expected header"):
        load_messages(path)


def test_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path, b"ham,hi,,,\nspam,win,,,\n")
    with pytest.raises(DataValidationError, match="fingerprint"):
        load_messages(path, expected_sha256="0" * 64)


def test_missing_dataset_points_to_instructions(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match=r"docs/DATASET.md"):
        load_messages(tmp_path / "nope.csv")
