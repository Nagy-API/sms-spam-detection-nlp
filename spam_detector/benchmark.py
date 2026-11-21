"""Measure what the deployment pays for an artifact: size, load time, memory, latency.

Standard library only, so it measures exactly what the API process does.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

from spam_detector.artifact import MODEL_FILENAME
from spam_detector.inference import SpamClassifier

SAMPLES = (
    "See you at lunch tomorrow",
    "WINNER!! Claim your £900 prize now, call 09061701461 before midnight",
    "Can you pick up milk on the way home? Thanks x",
    "URGENT: your account has been suspended, verify at www.example.com/login",
)


def benchmark_artifact(directory: Path, repeats: int = 2000) -> dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    classifier, loaded = SpamClassifier.load(directory)
    load_seconds = time.perf_counter() - start
    _, peak_load = tracemalloc.get_traced_memory()
    resident = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    latencies = {}
    for label, text in (("typical_sms", SAMPLES[1]), ("max_length_5000_chars", (SAMPLES[1] + " ") * 70)):
        text = text[:5000]
        timings = []
        for _ in range(repeats if label == "typical_sms" else max(repeats // 20, 20)):
            begin = time.perf_counter()
            classifier.predict(text)
            timings.append((time.perf_counter() - begin) * 1000)
        timings.sort()
        latencies[label] = {
            "p50_ms": round(statistics.median(timings), 4),
            "p95_ms": round(timings[int(len(timings) * 0.95) - 1], 4),
            "max_ms": round(timings[-1], 4),
        }
    return {
        "model_version": loaded.artifact.model_version,
        "model_json_bytes": (directory / MODEL_FILENAME).stat().st_size,
        "n_features": loaded.artifact.feature_space.size,
        "load_seconds": round(load_seconds, 4),
        "python_heap_after_load_mib": round(resident / 2**20, 2),
        "python_heap_peak_during_load_mib": round(peak_load / 2**20, 2),
        "latency": latencies,
    }
