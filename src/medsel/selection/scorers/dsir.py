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

from medsel.selection.base import Scorer, record_text, register_scorer

__all__ = ["DSIRScorer", "hashed_ngram_counts"]

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

# Where each loader's target text comes from by default. Always a training split: the point of a
# target distribution is to describe the task, and drawing it from the split the model is scored on
# would tune selection to the test items themselves. PubMedQA has only one split and its evaluation
# slice is the first 500 rows, so its target starts after them.
_DEFAULT_TARGET_SPLITS = {
    "medqa": "train",
    "medmcqa": "train",
    "pubmedqa": "train[500:]",
}


def _parse_slice(spec: str) -> tuple[int, float]:
    start, _, stop = spec.partition(":")
    return int(start or 0), float(stop or "inf")


def _overlaps(target_split: str, eval_split: str) -> bool:
    """Whether two split specifications can share a row."""
    target_base, _, target_slice = target_split.partition("[")
    eval_base, _, eval_slice = eval_split.partition("[")
    if target_base != eval_base:
        return False
    if not target_slice or not eval_slice:
        return True
    target_start, target_stop = _parse_slice(target_slice.rstrip("]"))
    eval_start, eval_stop = _parse_slice(eval_slice.rstrip("]"))
    return target_start < eval_stop and eval_start < target_stop


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
class DSIRScorer(Scorer):
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
        if buckets < 1:
            raise ValueError(f"buckets must be positive, got {buckets}")
        if ngram < 1:
            raise ValueError(f"ngram must be at least 1, got {ngram}")
        self.target = target
        self.target_split = target_split
        self.target_limit = target_limit
        self.buckets = buckets
        self.ngram = ngram
        self.alpha = alpha
        self.gumbel = gumbel
        self.seed = seed

    def resolve_target_split(self) -> str | None:
        """The split the target text is drawn from, or ``None`` for explicit texts.

        Raises if that split can share rows with the one the model is evaluated on. Fitting the
        selection distribution to the evaluation items would inflate every downstream number
        without any of it being real, and it is an easy mistake to make from a config file.
        """
        if not isinstance(self.target, str):
            return None

        from medsel.registry import get_loader

        loader = get_loader(self.target)
        split = self.target_split or _DEFAULT_TARGET_SPLITS.get(self.target, "train")
        eval_split = str(getattr(loader, "eval_split", "") or "")

        if eval_split and _overlaps(split, eval_split):
            raise ValueError(
                f"DSIR target {self.target}:{split} overlaps {self.target}:{eval_split}, which is "
                f"the split this task is scored on. Selecting towards the evaluation set inflates "
                f"the result without improving the model. Use a training split instead."
            )
        return split

    def _target_texts(self) -> list[str]:
        if not isinstance(self.target, str):
            return [str(text) for text in self.target]

        from medsel.prompts import render_prompt
        from medsel.registry import get_loader
        from medsel.schema import QAExample

        loader = get_loader(self.target)
        split = self.resolve_target_split()
        assert split is not None
        texts: list[str] = []
        for example in loader.load(split, limit=self.target_limit):
            texts.append(
                render_prompt(example) if isinstance(example, QAExample) else record_text(example)
            )
        return texts

    def _fit(self, texts: Sequence[str]) -> _BagOfNgrams:
        model = _BagOfNgrams(self.buckets, self.alpha)
        for text in texts:
            model.update(hashed_ngram_counts(text, self.buckets, self.ngram))
        model.finalize()
        return model

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []

        target_texts = self._target_texts()
        if not target_texts:
            raise ValueError(
                f"target {self.target!r} produced no text, so there is no distribution to select "
                "towards"
            )

        pool_texts = [record_text(record) for record in records]
        target_model = self._fit(target_texts)
        pool_model = self._fit(pool_texts)

        rng = random.Random(self.seed)
        scores: list[float] = []
        for text in pool_texts:
            counts = hashed_ngram_counts(text, self.buckets, self.ngram)
            total = sum(counts.values()) or 1
            weight = (target_model.loglik(counts) - pool_model.loglik(counts)) / total
            if self.gumbel:
                weight += -math.log(-math.log(rng.random() + 1e-12) + 1e-12)
            scores.append(weight)
        return scores

    def __repr__(self) -> str:
        target = (
            self.target if isinstance(self.target, str) else f"<{len(list(self.target))} texts>"
        )
        return f"DSIRScorer(target={target!r}, buckets={self.buckets}, ngram={self.ngram})"
