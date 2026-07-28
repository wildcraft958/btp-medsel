"""Data selection: scoring candidates and assembling subsets under a budget."""

from medsel.selection.base import Scorer, available_scorers, get_scorer, register_scorer
from medsel.selection.selector import resolve_budget, select_stratified, select_top_k

__all__ = [
    "Scorer",
    "register_scorer",
    "get_scorer",
    "available_scorers",
    "resolve_budget",
    "select_top_k",
    "select_stratified",
]
