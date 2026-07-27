"""Loader registry.

Adding a dataset must not require editing a central dispatch table. A new module in
``medsel.data`` carrying ``@register_loader("name")`` is discovered automatically, so four people
can add loaders in parallel without ever touching the same file.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from medsel.data.base import BaseLoader

__all__ = ["register_loader", "get_loader", "get_loader_class", "available_loaders"]

_LOADERS: dict[str, type] = {}
_discovered = False

T = TypeVar("T", bound=type)


def register_loader(name: str) -> Callable[[T], T]:
    """Class decorator binding a loader class to a lookup name."""

    def decorator(cls: T) -> T:
        existing = _LOADERS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"loader name {name!r} is already registered by {existing.__name__}")
        cls.name = name  # type: ignore[attr-defined]
        _LOADERS[name] = cls
        return cls

    return decorator


def _discover() -> None:
    """Import every ``medsel.data`` submodule so decorators run. Idempotent."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    import medsel.data as data_pkg

    for module in pkgutil.iter_modules(data_pkg.__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"medsel.data.{module.name}")


def available_loaders() -> list[str]:
    _discover()
    return sorted(_LOADERS)


def get_loader_class(name: str) -> type[BaseLoader]:
    _discover()
    try:
        return _LOADERS[name]
    except KeyError:
        raise KeyError(
            f"unknown loader {name!r}; available: {', '.join(available_loaders()) or '(none)'}"
        ) from None


def get_loader(name: str, **kwargs) -> BaseLoader:
    """Instantiate a registered loader, e.g. ``get_loader("medmcqa")``."""
    return get_loader_class(name)(**kwargs)
