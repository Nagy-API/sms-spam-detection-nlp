import hashlib
from pathlib import Path

import pytest

from spam_detector.data import Message
from spam_detector.training import v3
from spam_detector.training.dedup import near_duplicate_groups
from spam_detector.training.v3 import Record, cluster, verify


def test_snapshot_integrity(tmp_path: Path) -> None:
    snapshot = tmp_path / "dataset.csv"
    snapshot.write_bytes(b"abc")
    verify(snapshot, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(snapshot, "0" * 64)


def test_ncsu_parser_ignores_contact_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / "ncsu.csv"
    data = b"messageID,objectID,destination number,message,time,error in time\n1,2,123456789,Hello there,0,0,extra\n"
    snapshot.write_bytes(data)
    monkeypatch.setattr(v3, "ROOT", tmp_path)
    monkeypatch.setitem(v3.SOURCES, "ncsu", ("https://example.com", hashlib.sha256(data).hexdigest(), "MIT"))
    rows, manifest = v3.read_source("ncsu")
    assert rows == [Record("ncsu", "phishing", "Hello there")]
    assert manifest["counts"] == {"raw": 1, "eligible": 1}
    assert "123456789" not in str(manifest)


def test_blocked_grouping_matches_original() -> None:
    messages = [
        Message(0, "spam", "A long promotion with coupon code 999"),
        Message(1, "spam", "A long promotion with coupon code 555"),
        Message(2, "spam", "A long promotion with a coupon code today"),
        Message(3, "ham", "Lunch tomorrow?"),
    ]
    expected = near_duplicate_groups(messages, 0.5)
    assert near_duplicate_groups(messages, 0.5, batch_size=1) == expected
    assert expected[0] == [0, 0, 0, 1]


def test_cross_source_overlap_is_quarantined_even_with_changed_urls() -> None:
    rows, audit = cluster([
        Record("uci_dev", "ham", "Lunch tomorrow?"),
        Record("ncsu", "phishing", "Visit https://first.example/ immediately for more information"),
        Record("imc_test", "phishing", "VISIT https://second.example/ immediately for more information"),
        Record("uci_dev", "ham", "Coffee sometime?"),
        Record("ncsu", "phishing", "coffee sometime?"),
    ])
    assert [row.text for row, _ in rows] == ["Lunch tomorrow?"]
    assert audit["cross_source_groups"] == 2
    assert audit["quarantined_keys"] == 1
    assert audit["conflicting_rows"] == 2
