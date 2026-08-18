"""Selecting from a pool too large to hold in memory.

`select_top_k` and `select_stratified` take every score at once, which is the right shape when the
pool fits in RAM and the wrong one for the full 23.9M document PubMed corpus.

Nothing here approximates. The results are identical to the in-memory path on the same data, which
is what the equivalence tests in `tests/test_selection_stream.py` check. Only two things change:

- **Selection** keeps a bounded heap instead of every score. To find the best `k` you only ever
  need the best `k` seen so far, so memory is proportional to the budget rather than to the pool.
- **Scoring** happens in chunks, which is why `Scorer.fit` exists. A scorer whose score is relative
  to the pool fits once over a bounded prefix, then every chunk is measured against that same
  reference frame.

The fit sample is the first `fit_size` records rather than a uniform sample, since drawing a
uniform sample would mean reading the whole stream before scoring any of it. That is only sound if
the stream order is unrelated to content. It holds for shard-sampled PubMed; it would not hold for
a corpus sorted by date or by journal, and anyone streaming such a source should shuffle first.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Any

from medsel.selection.base import Scorer
from medsel.selection.selector import allocate_quotas, resolve_budget

__all__ = ["TopK", "StreamSelection", "select_stream"]


class TopK:
    """The highest-scoring `capacity` items from a stream, in bounded memory.

    Ties keep the earlier record, matching :func:`select_top_k`, so a stream and an in-memory run
    over the same data agree down to tie ordering rather than only on the score.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(0, capacity)
        self._heap: list[tuple[float, int, str]] = []
        self._seen = 0

    def offer(self, score: float, uid: str) -> None:
        self._seen += 1
        if self._capacity == 0:
            return
        # Negated position so that among equal scores the heap root, and therefore the first
        # eviction, is the record that arrived latest.
        entry = (score, -self._seen, uid)
        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def entries(self) -> list[tuple[float, int, str]]:
        """Kept items as ``(score, arrival position, uid)``, best first."""
        return [
            (score, -position, uid)
            for score, position, uid in sorted(self._heap, key=lambda e: (-e[0], -e[1]))
        ]

    def ranked(self) -> list[tuple[float, str]]:
        """Kept items as ``(score, uid)``, best first."""
        return [(score, uid) for score, _, uid in self.entries()]

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class StreamSelection:
    """The outcome of a streaming selection, without the pool it came from."""

    uids: list[str]
    n_candidates: int
    score_summary: dict[str, float] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.uids)


class _RunningStats:
    """Score distribution summary that never holds the scores."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.lowest = float("inf")
        self.highest = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.lowest = min(self.lowest, value)
        self.highest = max(self.highest, value)

    def summary(self) -> dict[str, float]:
        if not self.count:
            return {}
        return {"min": self.lowest, "max": self.highest, "mean": self.total / self.count}


def _summary_with_selection(stats: _RunningStats, selected: list[float]) -> dict[str, float]:
    """Pool summary plus the same three figures over what was kept.

    The pair is the point: a selected mean sitting on top of the pool mean means the scorer did
    not discriminate, and no amount of downstream training will recover from that.
    """
    summary = stats.summary()
    if summary and selected:
        summary["selected_min"] = min(selected)
        summary["selected_max"] = max(selected)
        summary["selected_mean"] = sum(selected) / len(selected)
    return summary


def _chunks(records: Iterator[Any], size: int) -> Iterator[list[Any]]:
    while True:
        chunk = list(islice(records, size))
        if not chunk:
            return
        yield chunk


def _group_of(record: Any, key: str) -> str:
    return str(getattr(record, "labels", {}).get(key, "unknown"))


def select_stream(
    records: Iterable[Any],
    scorer: Scorer,
    budget: int,
    *,
    fit_size: int = 10_000,
    chunk_size: int = 1_000,
    stratify_by: str | None = None,
    min_per_group: int = 0,
    progress: bool = True,
) -> StreamSelection:
    """Select from `records` without materialising them.

    `budget` must be a count. A fraction of the pool is not defined until the stream ends, and by
    then the records it would have kept are gone, so guessing would silently return the wrong
    subset. Pass an integer, or use the in-memory path where the pool size is known up front.

    Memory is bounded by `budget` for a plain top-k, and by `budget` times the number of groups
    when stratifying, since no group can be allocated more than the whole budget.
    """
    if isinstance(budget, float):
        raise ValueError(
            f"streaming needs a count, not the fraction {budget}. The pool size is unknown until "
            "the stream ends, by which point the discarded records cannot be recovered. Pass an "
            "integer budget, or select in memory where the pool size is known up front."
        )
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    from tqdm.auto import tqdm

    stream = iter(records)
    warmup = list(islice(stream, max(0, fit_size)))
    scorer.fit(warmup)

    stats = _RunningStats()
    heaps: dict[str, TopK] = {}
    counts: dict[str, int] = {}
    single = TopK(budget)
    n_candidates = 0

    bar = tqdm(
        desc=f"select:{scorer.name or type(scorer).__name__}", unit="doc", disable=not progress
    )
    for chunk in _chunks(chain(warmup, stream), max(1, chunk_size)):
        scores = scorer(chunk)
        for record, score in zip(chunk, scores, strict=True):
            n_candidates += 1
            stats.add(score)
            if stratify_by is None:
                single.offer(score, record.uid)
            else:
                group = _group_of(record, stratify_by)
                counts[group] = counts.get(group, 0) + 1
                heaps.setdefault(group, TopK(budget)).offer(score, record.uid)
        bar.update(len(chunk))
    bar.close()

    total = resolve_budget(budget, n_candidates)

    if stratify_by is None:
        chosen = single.ranked()[:total]
        return StreamSelection(
            uids=[uid for _, uid in chosen],
            n_candidates=n_candidates,
            score_summary=_summary_with_selection(stats, [score for score, _ in chosen]),
        )

    quotas = allocate_quotas(counts, total, min_per_group)
    kept: list[tuple[int, str, float]] = []
    for group, heap in heaps.items():
        for score, position, uid in heap.entries()[: quotas.get(group, 0)]:
            kept.append((position, uid, score))
    # Arrival order, matching select_stratified returning ascending indices: a stratified subset
    # has no meaningful global rank, so pool order is the honest presentation.
    kept.sort()
    return StreamSelection(
        uids=[uid for _, uid, _ in kept],
        n_candidates=n_candidates,
        score_summary=_summary_with_selection(stats, [score for _, _, score in kept]),
        group_counts=counts,
    )
