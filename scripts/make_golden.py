"""Regenerate tests/golden/golden_predictions.json for the current release artifact.

Run only when a new model version is released on purpose; the diff of the golden
file then documents exactly which behaviours changed.
"""

import json
from pathlib import Path

from spam_detector.inference import SpamClassifier

ROOT = Path(__file__).resolve().parents[1]
# Handwritten examples (not taken from the dataset), covering normal, Unicode and abstention paths.
CASES = [
    "See you at lunch tomorrow",
    "Congratulations! You won a free prize. Call now to claim.",
    "WINNER!! Claim your £900 prize, call 09061701461 before midnight",
    "Can you pick up milk on the way home? Thanks x",
    "Your account is suspended. Verify now at www.example-secure-login.com",
    "Free entry to win a holiday! Txt WIN to 87121",
    "Mum says dinner is at 7, don’t be late 🙂",
    "ＦＲＥＥ ＰＲＩＺＥ ｃａｌｌ ０９０６１７０１４６１",
    "ok",
    "12345",
    "🙂🙂🙂",
    "https://example.com",
    "مرحبا كيف حالك",
    "Привет, как дела?",
    "qzxwvj",
    "the and is",
]


def main() -> None:
    directory = ROOT / "models" / "spam-v2"
    classifier, loaded = SpamClassifier.load(directory)
    cases = []
    for text in CASES:
        p = classifier.predict(text)
        cases.append({"text": text, "label": p.label, "reason": p.reason, "spam_probability": p.spam_probability})
    payload = {"model_sha256": loaded.sha256, "cases": cases}
    out = ROOT / "tests" / "golden" / "golden_predictions.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for case in cases:
        print(f"{case['label']:13s} {case['reason'] or '':24s} {case['spam_probability']!s:22s} {case['text']}")


if __name__ == "__main__":
    main()
