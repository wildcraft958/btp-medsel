"""Turning scores into a chosen subset under a budget.

Scoring and selecting are kept apart deliberately. The proposal's capability-aware constraints -
keeping rare diseases and underrepresented specialties present - are a property of *how* a subset
is assembled from scores, not of the scores themselves, and stratified selection is where they
will live.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

__all__ = ["resolve_budget", "select_top_k", "select_stratified"]


def resolve_budget(budget: int | float, n: int) -> int:
    """Interpret a budget as a count (``int``) or a fraction of the pool (``float``).

    ``1`` selects one record; ``1.0`` selects all of them. The int/float distinction carries the
    meaning, so both readings stay available without a second argument.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if isinstance(budget, float):
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"fractional budget must be in [0.0, 1.0], got {budget}")
        return min(n, int(round(budget * n)))

    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    return min(n, budget)


def select_top_k(scores: Sequence[float], budget: int | float) -> list[int]:
    """Indices of the highest-scoring records, best first.

    Ties break by original index so that a run is reproducible and two scorers that agree on
    values also agree on output.
    """
    k = resolve_budget(budget, len(scores))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def select_stratified(
    scores: Sequence[float],
    groups: Sequence[str],
    budget: int | float,
    min_per_group: int = 0,
) -> list[int]:
    """Top-scoring records within each group, allocated proportionally to group size.

    ``min_per_group`` is the capability-retention lever: it guarantees a floor of representation
    for small groups - rare specialties, uncommon MeSH terms - that a global top-k would drop
    entirely at aggressive budgets.

    Returns indices in ascending order, since a stratified subset has no meaningful global rank.

    Raises:
        ValueError: If the budget is too small to give every group ``min_per_group`` records.
            Silently returning fewer would break the guarantee this argument exists to provide,
            and a capability-retention experiment would then be measuring something other than
            what it claims to.
    """
    if len(scores) != len(groups):
        raise ValueError(f"got {len(scores)} scores but {len(groups)} group labels")

    total = resolve_budget(budget, len(scores))
    if total == 0:
        return []

    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)

    if min_per_group:
        floor = min_per_group * len(members)
        if floor > total:
            raise ValueError(
                f"min_per_group={min_per_group} across {len(members)} group(s) needs {floor} "
                f"records but the budget resolves to {total}. Raise the budget or lower the floor."
            )

    for indices in members.values():
        indices.sort(key=lambda i: (-scores[i], i))

    # Proportional allocation, then hand out the rounding shortfall to the largest groups so the
    # budget is met exactly rather than approximately.
    n = len(scores)
    quotas = {
        group: max(min_per_group, int(len(indices) * total / n))
        for group, indices in members.items()
    }
    for group, indices in members.items():
        quotas[group] = min(quotas[group], len(indices))

    order = sorted(members, key=lambda g: (-len(members[g]), g))
    while sum(quotas.values()) > total:
        for group in reversed(order):
            if sum(quotas.values()) <= total:
                break
            if quotas[group] > min_per_group:
                quotas[group] -= 1
            elif quotas[group] > 0 and sum(quotas.values()) > total:
                quotas[group] -= 1
    while sum(quotas.values()) < total:
        progressed = False
        for group in order:
            if sum(quotas.values()) >= total:
                break
            if quotas[group] < len(members[group]):
                quotas[group] += 1
                progressed = True
        if not progressed:  # every group exhausted
            break

    chosen: list[int] = []
    for group, indices in members.items():
        chosen.extend(indices[: quotas[group]])
    return sorted(chosen)
