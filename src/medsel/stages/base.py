"""Stage interface and run bookkeeping."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from medsel import __version__
from medsel.config import ExperimentConfig, to_dict
from medsel.utils.device import describe_device

__all__ = ["Stage", "StageResult", "read_selection", "apply_selection"]


def read_selection(path: str | Path) -> dict[str, Any]:
    """Load a selection manifest written by ``medsel select``."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"selection manifest {manifest_path} does not exist. Produce one with "
            "`medsel select --output <path>` before training on a subset."
        )
    payload = json.loads(manifest_path.read_text())
    if "selected_uids" not in payload:
        raise ValueError(
            f"{manifest_path} has no selected_uids, so it is not a selection manifest. A run "
            "manifest from a training stage is a different file."
        )
    return payload


def apply_selection(
    records: Iterable[Any], path: str | Path, source: str, split: str
) -> Iterator[Any]:
    """Yield only the records a selection manifest chose, in pool order.

    Pool order rather than manifest order, because a training stream should not inherit the
    scorer's ranking: packing consecutive high-scoring documents together would correlate the
    ordering of the training data with the score, which is a confound nobody would want and
    nobody would see.

    Raises if any selected uid is absent from the pool. That means the manifest was built against
    a different pool than the one being trained on, and continuing would train on a silently
    smaller subset while reporting the budget the manifest claims.
    """
    manifest = read_selection(path)

    if manifest.get("source") != source:
        raise ValueError(
            f"selection manifest was built from {manifest.get('source')!r} but this run trains on "
            f"{source!r}. Selections are not transferable between sources."
        )
    if manifest.get("split") != split:
        raise ValueError(
            f"selection manifest was built from split {manifest.get('split')!r} but this run "
            f"trains on {split!r}."
        )

    wanted = set(manifest["selected_uids"])
    if not wanted:
        raise ValueError(f"{path} selected no records, so there is nothing to train on")

    found = 0
    for record in records:
        if record.uid in wanted:
            found += 1
            yield record

    if found != len(wanted):
        raise ValueError(
            f"found {found} of {len(wanted)} selected records in {source}:{split}. The manifest "
            f"was built against a different pool, so the loader arguments and limit have to match "
            f"the ones the selection used ({manifest.get('n_candidates')} candidates)."
        )


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
