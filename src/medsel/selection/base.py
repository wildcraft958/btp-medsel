"""Scorer interface and registry.

This is the project's main extension point. LESS, 3DS, TRAK, GLISTER and the learned diversity
work all arrive as scorers behind this one interface, so they can be swapped and ablated without
touching the loaders, the stages, or each other.

Two trivial scorers ship now - not as placeholders, but because CPT over 23.9M PubMed documents
is impossible without *some* subset step, and a random baseline is the control that every
sophisticated scorer has to beat.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeVar

__all__ = ["Scorer", "register_scorer", "get_scorer", "available_scorers"]

_SCORERS: dict[str, type] = {}
_discovered = False

T = TypeVar("T", bound=type)


class Scorer(ABC):
    """Assigns a value to each candidate record. Higher means more worth training on.

    Scores carry no fixed scale: only their ordering within one call is meaningful. Selectors
    rank, they never threshold on an absolute value.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def score(self, records: Sequence[Any]) -> list[float]:
        """Return one score per record, in input order."""

    def __call__(self, records: Sequence[Any]) -> list[float]:
        scores = self.score(records)
        if len(scores) != len(records):
            raise ValueError(
                f"{type(self).__name__} returned {len(scores)} scores for {len(records)} records"
            )
        return scores

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def register_scorer(name: str) -> Callable[[T], T]:
    """Class decorator binding a scorer to a lookup name."""

    def decorator(cls: T) -> T:
        existing = _SCORERS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"scorer name {name!r} is already registered by {existing.__name__}")
        cls.name = name  # type: ignore[attr-defined]
        _SCORERS[name] = cls
        return cls

    return decorator


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True

    import medsel.selection.scorers as pkg

    for module in pkgutil.iter_modules(pkg.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"medsel.selection.scorers.{module.name}")


def available_scorers() -> list[str]:
    _discover()
    return sorted(_SCORERS)


def get_scorer(name: str, **kwargs: Any) -> Scorer:
    _discover()
    try:
        cls = _SCORERS[name]
    except KeyError:
        raise KeyError(
            f"unknown scorer {name!r}; available: {', '.join(available_scorers()) or '(none)'}"
        ) from None
    return cls(**kwargs)
