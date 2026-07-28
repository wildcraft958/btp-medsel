"""Seeding.

Data selection is the object of study here, so which samples a scorer picks has to be
reproducible independently of whether a trainer has seeded anything yet.
"""

from __future__ import annotations

import os
import random

__all__ = ["set_seed"]


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and (if installed) torch. Returns the seed for logging."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed
