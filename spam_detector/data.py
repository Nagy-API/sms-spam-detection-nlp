"""Load, repair, and validate the SMS Spam Collection CSV (Kaggle representation).

The raw file has three defects that the legacy pipeline silently inherited:

1. **Split messages.** 50 rows contain commas that broke CSV quoting, so the
   end of the message spilled into the unnamed 3rd-5th columns. The legacy code
   kept only ``v2`` and therefore trained on truncated text. We rejoin them.
2. **Double-encoded characters (mojibake).** The text was UTF-8, decoded as
   Windows-1252, and re-encoded as Mac Roman. ``£`` is stored as the bytes
   ``E5 A3`` and ``'`` as ``89 DB F7``. A model trained on that never sees a
   real ``£``, which is what users type. Each run of non-ASCII bytes is
   reversed with ``mac_roman -> cp1252 -> utf-8`` when that round-trip is
   valid; runs that cannot be reversed unambiguously are kept as Mac Roman
   text and counted in the report.
3. **HTML entities.** ``&amp;``, ``&lt;`` and ``&gt;`` appear escaped; the
   ``&lt;#&gt;`` redaction marker appears only in legitimate messages.

Nothing here mutates the file on disk; repairs happen in memory and are counted.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

VALID_LABELS = frozenset({"ham", "spam"})
EXPECTED_HEADER = ["v1", "v2"]
_NON_ASCII_RUN_RE = re.compile(rb"[\x80-\xff]+")


class DataValidationError(ValueError):
    """The dataset does not match the expected schema or fingerprint."""


@dataclass(frozen=True)
class Message:
    """One labeled SMS. ``row_id`` is the 0-based data row in the source file."""

    row_id: int
    label: str
    text: str

    @property
    def is_spam(self) -> int:
        return int(self.label == "spam")


@dataclass
class LoadReport:
    sha256: str
    rows: int = 0
    labels: Counter[str] = field(default_factory=Counter)
    rejoined_split_rows: int = 0
    mojibake_runs_repaired: int = 0
    mojibake_runs_unrepaired: int = 0
    rows_with_html_entities: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "rows": self.rows,
            "labels": dict(sorted(self.labels.items())),
            "rejoined_split_rows": self.rejoined_split_rows,
            "mojibake_runs_repaired": self.mojibake_runs_repaired,
            "mojibake_runs_unrepaired": self.mojibake_runs_unrepaired,
            "rows_with_html_entities": self.rows_with_html_entities,
        }


def repair_encoding(raw: bytes, report: LoadReport | None = None) -> str:
    """Decode the file, reversing the UTF-8 -> cp1252 -> Mac Roman double encoding."""

    def fix(match: re.Match[bytes]) -> str:
        as_mac_roman = match.group().decode("mac_roman")
        try:
            repaired = as_mac_roman.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            if report is not None:
                report.mojibake_runs_unrepaired += 1
            return as_mac_roman
        if report is not None:
            report.mojibake_runs_repaired += 1
        # Some originals were themselves cp1252 bytes stored as C1 control
        # characters (U+0080-U+009F); map those to the characters they denote.
        return "".join(
            bytes([ord(c)]).decode("cp1252", errors="replace") if 0x80 <= ord(c) <= 0x9F else c
            for c in repaired
        )

    pieces, last = [], 0
    for match in _NON_ASCII_RUN_RE.finditer(raw):
        pieces.append(raw[last : match.start()].decode("ascii"))
        pieces.append(fix(match))
        last = match.end()
    pieces.append(raw[last:].decode("ascii"))
    return "".join(pieces)


def load_messages(path: Path, expected_sha256: str | None = None) -> tuple[list[Message], LoadReport]:
    """Read, fingerprint, repair, and schema-check the dataset."""
    if not path.is_file():
        raise DataValidationError(
            f"dataset not found at {path}. Download spam.csv as described in docs/DATASET.md."
        )
    raw = path.read_bytes()
    report = LoadReport(sha256=hashlib.sha256(raw).hexdigest())
    if expected_sha256 and report.sha256 != expected_sha256:
        raise DataValidationError(
            f"dataset fingerprint {report.sha256} differs from the expected {expected_sha256}; "
            "results would not be comparable with the recorded experiments"
        )
    reader = csv.reader(io.StringIO(repair_encoding(raw, report), newline=""))
    header = next(reader, None)
    if header is None or [h.strip() for h in header[:2]] != EXPECTED_HEADER:
        raise DataValidationError(f"expected header starting with {EXPECTED_HEADER}, got {header!r}")

    messages: list[Message] = []
    for row_id, row in enumerate(reader):
        if not row:
            continue
        label = row[0].strip()
        if label not in VALID_LABELS:
            raise DataValidationError(f"row {row_id}: invalid label {label!r}")
        # Rejoin text that spilled into the trailing unnamed columns.
        trailing = row[2:]
        while trailing and not trailing[-1]:
            trailing.pop()
        if trailing:
            report.rejoined_split_rows += 1
        text = ",".join([row[1], *trailing]) if len(row) > 1 else ""
        if re.search(r"&(?:amp|lt|gt|quot);", text):
            report.rows_with_html_entities += 1
            text = html.unescape(text)
        if not text.strip():
            raise DataValidationError(f"row {row_id}: empty message text")
        messages.append(Message(row_id=row_id, label=label, text=text))
        report.labels[label] += 1
    report.rows = len(messages)
    if set(report.labels) != VALID_LABELS:
        raise DataValidationError(f"both classes must be present, found {dict(report.labels)}")
    return messages, report
