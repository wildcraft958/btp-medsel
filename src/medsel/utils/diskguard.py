"""Free-space checks for anything that materialises data to disk.

The full ``MedRAG/pubmed`` repository is roughly 279 GB. Development machines on this project have
far less headroom than that, and a half-written corpus that fills the root filesystem is a much
worse failure than an upfront refusal. Every writing path calls :func:`require_free` first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["InsufficientDiskSpace", "free_bytes", "human_bytes", "require_free"]

_UNITS = ("B", "KB", "MB", "GB", "TB")


class InsufficientDiskSpace(RuntimeError):
    """Raised before writing when the target filesystem lacks room."""


def human_bytes(n: float) -> str:
    for unit in _UNITS:
        if abs(n) < 1024 or unit == _UNITS[-1]:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} {_UNITS[-1]}"


def free_bytes(path: str | Path = ".") -> int:
    """Free bytes on the filesystem holding ``path``, walking up to the nearest existing parent."""
    target = Path(path).resolve()
    while not target.exists():
        if target.parent == target:
            break
        target = target.parent
    return shutil.disk_usage(target).free


def require_free(needed: int, path: str | Path = ".", margin: float = 0.20) -> None:
    """Raise unless ``needed`` bytes plus a ``margin`` safety factor are available.

    ``margin`` is a fraction of ``needed``, not of the disk: a 1 GB write with the default margin
    requires 1.2 GB free. Sizing the buffer to the write keeps the guard usable on nearly-full
    disks, where a whole-disk percentage would reject everything.
    """
    if needed < 0:
        raise ValueError(f"needed must be non-negative, got {needed}")

    required = int(needed * (1 + margin))
    available = free_bytes(path)
    if available < required:
        raise InsufficientDiskSpace(
            f"need {human_bytes(required)} free at {path} "
            f"({human_bytes(needed)} + {margin:.0%} margin) but only "
            f"{human_bytes(available)} available. "
            f"Free space, lower the shard budget, or point MEDSEL_CACHE at a larger volume."
        )
