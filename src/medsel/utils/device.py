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
    """Whether bf16 runs on real silicon here.

    ``torch.cuda.is_bf16_supported()`` answers a different question: it counts an emulation path
    and returns True on a Quadro P5000 (compute capability 6.1), where bf16 is slower and less
    accurate than plain fp32. Native bf16 arrives with the Ampere tensor cores at 8.0.

    CPU bf16 exists too and is slow enough to be a trap, so only CUDA is ever considered.
    """
    import torch

    device = device or pick_device()
    if device != "cuda":
        return False
    return torch.cuda.get_device_capability() >= (8, 0)


def resolve_dtype(requested: str = "auto", device: str | None = None) -> Any:
    """Map a config dtype string onto a concrete ``torch.dtype``.

    ``auto`` follows the GPU generation: bf16 from Ampere (8.0), fp16 back to Volta (7.0) where
    half precision still has tensor cores behind it, and fp32 below that. It is always fp32 on CPU,
    where fp16 is both slow and numerically fragile.

    The pre-Volta case is not hypothetical. On a Pascal card, fp16 throughput is a fraction of
    fp32, so choosing half precision there costs speed as well as accuracy.
    """
    import torch

    device = device or pick_device()

    if requested == "auto":
        if device != "cuda":
            return torch.float32
        if supports_bf16(device):
            return torch.bfloat16
        return torch.float16 if torch.cuda.get_device_capability() >= (7, 0) else torch.float32

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
