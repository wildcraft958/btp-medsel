"""Baseline scorers: the controls every learned scorer must beat.

Neither is clever. That is the point - a selection method that cannot outperform "pick at random"
or "prefer longer documents" at equal token budget has not demonstrated anything.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from medsel.selection.base import Scorer, record_text, register_scorer

__all__ = ["RandomScorer", "LengthScorer"]


@register_scorer("random")
class RandomScorer(Scorer):
    """Uniform random scores. The control condition.

    Seeded from a dedicated generator rather than the global one, so a selection stays
    reproducible regardless of what else has consumed randomness in the process.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def score(self, records: Sequence[Any]) -> list[float]:
        rng = random.Random(self.seed)
        return [rng.random() for _ in records]

    def __repr__(self) -> str:
        return f"RandomScorer(seed={self.seed})"


@register_scorer("length")
class LengthScorer(Scorer):
    """Prefers longer text, as a crude stand-in for information content.

    Known to be a weak proxy: length correlates with substance in PubMed abstracts but also
    rewards verbosity and boilerplate. Included because it is the simplest non-random signal, so
    it separates "selection helps" from "any non-uniform ordering helps".
    """

    def __init__(self, unit: str = "chars") -> None:
        if unit not in {"chars", "words"}:
            raise ValueError(f"unit must be 'chars' or 'words', got {unit!r}")
        self.unit = unit

    def score(self, records: Sequence[Any]) -> list[float]:
        if self.unit == "words":
            return [float(len(record_text(r).split())) for r in records]
        return [float(len(record_text(r))) for r in records]

    def __repr__(self) -> str:
        return f"LengthScorer(unit={self.unit!r})"
