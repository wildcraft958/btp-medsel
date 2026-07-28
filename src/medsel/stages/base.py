"""Stage interface and run bookkeeping."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from medsel import __version__
from medsel.config import ExperimentConfig, to_dict
from medsel.utils.device import describe_device

__all__ = ["Stage", "StageResult"]


@dataclass
class StageResult:
    """What a finished stage hands back."""

    stage: str
    output_dir: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None


class Stage(ABC):
    """One step of the CPT -> SFT -> alignment pipeline.

    The split between :meth:`prepare` and :meth:`run` is deliberate: data assembly is the
    expensive, interesting part of this project and needs to be inspectable on its own, without
    loading a model or touching an optimiser.
    """

    name: ClassVar[str] = ""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.train.output_dir)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.config.name!r}, out={self.output_dir})"

    @abstractmethod
    def prepare(self) -> Any:
        """Assemble and return this stage's training data. No model, no optimiser."""

    @abstractmethod
    def run(self) -> StageResult:
        """Execute the stage end to end."""

    def write_manifest(self, metrics: dict[str, Any]) -> Path:
        """Record config, hardware and metrics next to the checkpoints.

        A checkpoint whose provenance is unrecorded is not a result anybody can defend later.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "stage": self.name,
            "experiment": self.config.name,
            "medsel_version": __version__,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hardware": describe_device(),
            "config": to_dict(self.config),
            "metrics": metrics,
        }
        path = self.output_dir / "run.json"
        path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        return path
