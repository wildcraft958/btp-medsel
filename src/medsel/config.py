"""Experiment configuration.

Runs are described by YAML rather than command-line flags so that an experiment is a reviewable
artefact a collaborator can read, diff, and re-run months later.

Unknown keys are a hard error. A silently ignored ``learning_rare`` typo would produce a run that
looks fine and trains at the wrong rate, which is far more expensive than a startup failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "load_experiment",
]


@dataclass
class DataConfig:
    """Which records to train on."""

    source: str
    split: str = "train"
    limit: int | None = None
    loader: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Which model to train and how to hold it in memory."""

    name_or_path: str
    dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    gradient_checkpointing: bool = False

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


@dataclass
class TrainConfig:
    """Optimisation and bookkeeping."""

    output_dir: str = "runs/cpt"
    block_size: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    drop_remainder: bool = True
    report_to: str = "none"

    # SFT-stage only. block_size above governs CPT packing; max_length governs how long a single
    # rendered instruction-response example may be before it is truncated.
    max_length: int = 1024
    # Train on the rationale as well as the answer letter, where the dataset ships one (MedMCQA
    # `exp`, PubMedQA `long_answer`). Off by default because rationale coverage is partial: many
    # MedMCQA rows have no explanation, so turning this on changes what fraction of the dataset
    # is usable and that has to be a deliberate choice.
    include_rationale: bool = False


@dataclass
class ExperimentConfig:
    """A complete, self-describing run."""

    name: str
    stage: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""


def _build(cls: type, data: dict[str, Any], where: str):
    """Instantiate a config dataclass, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"{where} must be a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {where}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )
    return cls(**data)


def _resolve(section: Any, base_dir: Path, where: str) -> dict[str, Any]:
    """Allow a section to be either inline or a path to a shared YAML fragment.

    ``model: configs/model/gemma3_1b.yaml`` keeps one model definition reusable across experiments
    without a templating system.
    """
    if isinstance(section, str):
        path = (base_dir / section) if not Path(section).is_absolute() else Path(section)
        if not path.exists():
            raise FileNotFoundError(f"{where} references {path}, which does not exist")
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"{path} must contain a mapping")
        return loaded
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"{where} must be a mapping or a path, got {type(section).__name__}")
    return section


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Read an experiment YAML into a validated :class:`ExperimentConfig`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a mapping at the top level")

    base_dir = path.parent
    raw = dict(raw)

    try:
        data = _build(DataConfig, _resolve(raw.pop("data", {}), base_dir, "data"), "data")
        model = _build(ModelConfig, _resolve(raw.pop("model", {}), base_dir, "model"), "model")
        train = _build(TrainConfig, _resolve(raw.pop("train", {}), base_dir, "train"), "train")
    except TypeError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    try:
        return _build(
            ExperimentConfig,
            {**raw, "data": data, "model": model, "train": train},
            str(path),
        )
    except TypeError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def to_dict(config: Any) -> Any:
    """Recursively convert config dataclasses to plain dicts for logging."""
    if is_dataclass(config) and not isinstance(config, type):
        return {f.name: to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, (list, tuple)):
        return [to_dict(v) for v in config]
    return config
