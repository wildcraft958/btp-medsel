"""DSIR: importance resampling over hashed n-gram features (Xie et al., 2023).

The first scorer here that is aimed at a *target*. Both baseline scorers judge a document on its
own; this one judges it against the distribution you actually want the model to be good at, which
is the whole premise of multi-target selection. The target is normally one of the project's QA
loaders, so "select PubMed text that looks like MedMCQA" is a one-line configuration.

Ported from the parallel CPT selection work on the lab machine. Two details from that
implementation are load bearing and are kept:

- hashing with ``zlib.crc32`` rather than ``hash``, which is salted per process and would make
  scores irreproducible across runs;
- Gumbel noise on the log importance weight, which makes a plain top-k equivalent to sampling
  without replacement from the importance weights, rather than deterministically taking the most
  target-like documents and losing all diversity.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from medsel.selection.base import record_text, register_scorer
from medsel.selection.targets import TargetedScorer

__all__ = ["DSIRScorer", "hashed_ngram_counts"]

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def hashed_ngram_counts(text: str, buckets: int, n: int = 2) -> Counter:
    """Count n-grams of order 1 to ``n`` into ``buckets`` hash buckets.

    Uses ``zlib.crc32`` rather than the builtin ``hash``, which is salted per process: the same
    document would land in different buckets on different runs and no score would reproduce.
    """
    tokens = _normalize(text).split()
    counts: Counter = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            gram = " ".join(tokens[i : i + k]).encode()
            counts[zlib.crc32(gram) % buckets] += 1
    return counts


class _BagOfNgrams:
    """Smoothed categorical distribution over hash buckets."""

    def __init__(self, buckets: int, alpha: float = 1.0) -> None:
        self._counts = np.full(buckets, alpha, dtype=np.float64)
        self._logprobs: np.ndarray | None = None

    def update(self, counts: Counter) -> None:
        for bucket, count in counts.items():
            self._counts[bucket] += count

    def finalize(self) -> None:
        self._logprobs = np.log(self._counts / self._counts.sum())

    def loglik(self, counts: Counter) -> float:
        assert self._logprobs is not None, "finalize() before loglik()"
        return float(sum(count * self._logprobs[bucket] for bucket, count in counts.items()))


@register_scorer("dsir")
class DSIRScorer(TargetedScorer):
    """Rank by how much more likely a document is under the target than under the pool.

    ``target`` is either a loader name (``"medmcqa"``, ``"medqa"``, ``"pubmedqa"``) or an explicit
    iterable of strings. The loader form is what experiments use; the explicit form keeps this
    testable without network access.

    Set ``gumbel=False`` for a deterministic ranking by raw importance weight. That is useful for
    inspection and for tests, but it gives up the sampling interpretation described in the module
    docstring, so experiments should leave it on.
    """

    def __init__(
        self,
        target: str | Iterable[str] = "medmcqa",
        *,
        target_split: str | None = None,
        target_limit: int = 2000,
        buckets: int = 10_000,
        ngram: int = 2,
        alpha: float = 1.0,
        gumbel: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(target, target_split=target_split, target_limit=target_limit)
        if buckets < 1:
            raise ValueError(f"buckets must be positive, got {buckets}")
        if ngram < 1:
            raise ValueError(f"ngram must be at least 1, got {ngram}")
        self.buckets = buckets
        self.ngram = ngram
        self.alpha = alpha
        self.gumbel = gumbel
        self.seed = seed
        self._pool_model: _BagOfNgrams | None = None
        self._target_model: _BagOfNgrams | None = None
        self._rng: random.Random | None = None

    def _fit(self, texts: Sequence[str]) -> _BagOfNgrams:
        model = _BagOfNgrams(self.buckets, self.alpha)
        for text in texts:
            model.update(hashed_ngram_counts(text, self.buckets, self.ngram))
        model.finalize()
        return model

    def fit(self, records: Sequence[Any]) -> None:
        """Fit the target and background distributions over a sample of the pool.

        Both are reference frames for every later score, so they are fitted once. Calling
        :meth:`score` on a chunk after this reuses them rather than rebuilding a background from
        the chunk, which would make scores from different chunks incomparable.
        """
        self._target_model = self._fit(self.target_texts())
        self._pool_model = self._fit([record_text(record) for record in records])
        # One generator for the scorer's whole life. A fresh one per chunk would replay the same
        # noise sequence on every chunk, which correlates the noise with position in the pool.
        self._rng = random.Random(self.seed)

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []

        if self._pool_model is None:
            self.fit(records)
        assert self._target_model is not None and self._pool_model is not None
        rng = self._rng or random.Random(self.seed)

        scores: list[float] = []
        for record in records:
            counts = hashed_ngram_counts(record_text(record), self.buckets, self.ngram)
            total = sum(counts.values()) or 1
            weight = (self._target_model.loglik(counts) - self._pool_model.loglik(counts)) / total
            if self.gumbel:
                weight += -math.log(-math.log(rng.random() + 1e-12) + 1e-12)
            scores.append(weight)
        return scores

    def __repr__(self) -> str:
        target = (
            self.target if isinstance(self.target, str) else f"<{len(list(self.target))} texts>"
        )
        return f"DSIRScorer(target={target!r}, buckets={self.buckets}, ngram={self.ngram})"
