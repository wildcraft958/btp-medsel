"""Turning scores into a chosen subset under a budget.

Scoring and selecting are kept apart deliberately. The proposal's capability-aware constraints -
keeping rare diseases and underrepresented specialties present - are a property of *how* a subset
is assembled from scores, not of the scores themselves, and stratified selection is where they
will live.

Both selectors take an optional ``costs`` list, one non-negative cost per record, and then spend a
budget in those units rather than in records. In practice the cost is a token count, and that is
the only budget a comparison between scorers can honestly hold fixed: training consumes tokens, and
a scorer that prefers long documents gets more of them for the same record count. Measured on
PubMed, ``length`` spends 2.06 times the tokens of ``random`` at an equal number of records, so a
record-matched comparison would hand it twice the training data and report that as a difference
between selection methods.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

__all__ = ["resolve_budget", "allocate_quotas", "select_top_k", "select_stratified"]


def _validate_costs(costs: Sequence[float] | None, n: int) -> list[float] | None:
    """Check a cost vector, or pass ``None`` through for a plain record budget."""
    if costs is None:
        return None
    if len(costs) != n:
        raise ValueError(f"got {n} scores but {len(costs)} costs")
    for index, cost in enumerate(costs):
        if cost <= 0:
            raise ValueError(
                f"costs must be positive, got {cost} at index {index}. A zero-cost record would "
                "fit any budget an unlimited number of times."
            )
    return list(costs)


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


def select_top_k(
    scores: Sequence[float], budget: int | float, costs: Sequence[float] | None = None
) -> list[int]:
    """Indices of the highest-scoring records, best first.

    Ties break by original index so that a run is reproducible and two scorers that agree on
    values also agree on output.

    With ``costs``, the budget is spent in cost units and a record that does not fit is skipped
    rather than ending the selection. Skipping fills the budget more completely, which is what
    matching a token budget across scorers requires; the alternative would leave a different
    amount unspent for each scorer and reintroduce the disparity the cost budget removes.
    """
    unit_costs = _validate_costs(costs, len(scores))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    if unit_costs is None:
        return order[: resolve_budget(budget, len(scores))]

    total = resolve_budget(budget, sum(unit_costs))
    chosen: list[int] = []
    spent = 0.0
    for index in order:
        if spent + unit_costs[index] > total:
            continue
        chosen.append(index)
        spent += unit_costs[index]
    return chosen


def allocate_quotas(
    group_sizes: dict[str, int], total: int, min_per_group: int = 0
) -> dict[str, int]:
    """How many records each group may contribute, given how large each group is.

    Proportional to group size, then the rounding shortfall goes to the largest groups so the
    budget is met exactly rather than approximately.

    Separated from :func:`select_stratified` because the streaming selector has to allocate from
    counts it accumulated rather than from lists of indices. Two implementations of this would
    eventually disagree, and the disagreement would look like a difference between selection
    methods rather than a bug.

    Raises:
        ValueError: If the budget cannot give every group ``min_per_group`` records.
    """
    if min_per_group:
        floor = min_per_group * len(group_sizes)
        if floor > total:
            raise ValueError(
                f"min_per_group={min_per_group} across {len(group_sizes)} group(s) needs {floor} "
                f"records but the budget resolves to {total}. Raise the budget or lower the floor."
            )

    n = sum(group_sizes.values())
    if not n:
        return {}

    quotas = {
        group: min(size, max(min_per_group, int(size * total / n)))
        for group, size in group_sizes.items()
    }

    order = sorted(group_sizes, key=lambda g: (-group_sizes[g], g))
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
            if quotas[group] < group_sizes[group]:
                quotas[group] += 1
                progressed = True
        if not progressed:  # every group exhausted
            break
    return quotas


def select_stratified(
    scores: Sequence[float],
    groups: Sequence[str],
    budget: int | float,
    min_per_group: int = 0,
    costs: Sequence[float] | None = None,
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
    unit_costs = _validate_costs(costs, len(scores))

    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    for indices in members.values():
        indices.sort(key=lambda i: (-scores[i], i))

    if unit_costs is None:
        total = resolve_budget(budget, len(scores))
        if total == 0:
            return []
        quotas = allocate_quotas(
            {group: len(indices) for group, indices in members.items()}, total, min_per_group
        )
        chosen: list[int] = []
        for group, indices in members.items():
            chosen.extend(indices[: quotas[group]])
        return sorted(chosen)

    return _select_stratified_by_cost(scores, members, budget, min_per_group, unit_costs)


def _select_stratified_by_cost(
    scores: Sequence[float],
    members: dict[str, list[int]],
    budget: int | float,
    min_per_group: int,
    unit_costs: list[float],
) -> list[int]:
    """Stratified selection where the budget is spent in cost units.

    Each group receives a share of the cost budget proportional to its share of the pool's cost,
    then fills that share greedily by score. ``min_per_group`` stays a floor in *records*, not in
    cost: it exists so a small group survives at all, and a group of short documents would lose
    that guarantee if the floor were denominated in tokens.
    """
    total = resolve_budget(budget, sum(unit_costs))
    if total == 0:
        return []

    chosen: list[int] = []
    if min_per_group:
        floor_cost = sum(
            sum(unit_costs[i] for i in indices[:min_per_group]) for indices in members.values()
        )
        if floor_cost > total:
            raise ValueError(
                f"min_per_group={min_per_group} across {len(members)} group(s) costs "
                f"{floor_cost:g} but the budget resolves to {total:g}. Raise the budget or "
                f"lower the floor."
            )
        for indices in members.values():
            chosen.extend(indices[:min_per_group])

    spent = sum(unit_costs[i] for i in chosen)
    group_cost = {group: sum(unit_costs[i] for i in indices) for group, indices in members.items()}
    pool_cost = sum(group_cost.values())

    for group, indices in members.items():
        share = group_cost[group] * total / pool_cost
        used = sum(unit_costs[i] for i in indices[:min_per_group])
        for index in indices[min_per_group:]:
            cost = unit_costs[index]
            if used + cost > share or spent + cost > total:
                continue
            chosen.append(index)
            used += cost
            spent += cost
    return sorted(chosen)
