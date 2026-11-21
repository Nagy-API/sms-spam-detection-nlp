"""English SMS spam detection: reproducible training pipeline and dependency-light inference.

Package layout
--------------
Runtime (standard library only, imported by the API):
    text        single authoritative text normalization
    features    n-gram analyzers and TF-IDF/count weighting
    artifact    pickle-free, checksummed JSON model format
    inference   calibrated prediction with abstention

Offline (requires ``requirements-train.txt``):
    data        loading, repairing, and validating the SMS corpus
    training/   deduplication, group-aware splits, candidates, metrics,
                experiments, release building, and the legacy baseline
    cli         ``python -m spam_detector ...`` commands
"""

__version__ = "2.0.0"
