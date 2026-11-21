"""Typed access to ``configs/pipeline.toml`` (parsed with the standard-library ``tomllib``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.toml"


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> PipelineConfig:
        with path.open("rb") as handle:
            return cls(tomllib.load(handle), path)

    def resolve(self, relative: str) -> Path:
        """Paths in the config are relative to the project root."""
        return (PROJECT_ROOT / relative).resolve()

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def split(self) -> dict[str, Any]:
        return self.raw["split"]

    @property
    def cross_validation(self) -> dict[str, Any]:
        return self.raw["cross_validation"]

    @property
    def decision(self) -> dict[str, float]:
        return self.raw["decision"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.raw["selection"]

    @property
    def release(self) -> dict[str, Any]:
        return self.raw["release"]

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return self.raw["candidates"]
