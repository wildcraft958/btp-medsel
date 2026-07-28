"""Device and dtype selection.

Compute for this project is not settled - a laptop CPU today, a lab GPU or Colab later. Configs
therefore say ``dtype: auto`` and this module resolves it against whatever hardware is actually
present, so the same experiment file runs everywhere without edits.

``torch`` is imported lazily throughout: the data loaders and the CLI must work with only the core
dependencies installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["pick_device", "supports_bf16", "resolve_dtype", "describe_device"]

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "full": "float32",
}


def pick_device() -> str:
    """Best available device: ``cuda``, ``mps``, or ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16(device: str | None = None) -> bool:
    """Whether bf16 is usable here. CPU bf16 exists but is slow enough to be a trap."""
    import torch

    device = device or pick_device()
    if device == "cuda":
        return torch.cuda.is_bf16_supported()
    return False


def resolve_dtype(requested: str = "auto", device: str | None = None) -> Any:
    """Map a config dtype string onto a concrete ``torch.dtype``.

    ``auto`` prefers bf16 on capable GPUs, falls back to fp16 on older ones, and always uses fp32
    on CPU - fp16 on CPU is both slow and numerically fragile.
    """
    import torch

    device = device or pick_device()

    if requested == "auto":
        if device == "cuda":
            return torch.bfloat16 if supports_bf16(device) else torch.float16
        return torch.float32

    key = requested.lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"unknown dtype {requested!r}; use one of: auto, {', '.join(sorted(_DTYPE_ALIASES))}"
        )
    return getattr(torch, _DTYPE_ALIASES[key])


def describe_device() -> dict[str, Any]:
    """Human-readable hardware summary, recorded alongside every run."""
    try:
        import torch
    except ImportError:
        return {"device": "unknown", "torch": "not installed (pip install -e '.[train]')"}

    device = pick_device()
    info: dict[str, Any] = {
        "device": device,
        "torch": torch.__version__,
        "bf16": supports_bf16(device),
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info
