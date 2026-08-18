"""Perplexity scoring against a reference model.

The standard CPT selection baseline, and the one most likely to be asked about in review, so it is
worth being precise about what each mode assumes.

``low`` keeps text the model already finds easy. That is the usual reading of "high quality", but
it also rewards boilerplate and text the model has effectively already learned. ``high`` keeps what
surprises the model, which finds genuinely new material and also finds noise, OCR damage and
non-prose. ``mid`` keeps the middle band and is the compromise most published recipes settle on.

None of the three is obviously correct, which is why the mode is a parameter rather than a
decision baked into the code. Ported from the parallel CPT selection work on the lab machine.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from medsel.selection.base import Scorer, record_text, register_scorer

__all__ = ["PerplexityScorer", "rank_by_mode", "sequence_nll"]

MODES = ("low", "mid", "high")


def rank_by_mode(nll: Sequence[float], mode: str, median: float | None = None) -> list[float]:
    """Turn mean token negative log likelihood into a score where higher means keep.

    ``median`` only affects ``mid``. Pass one fitted over the pool when scoring in chunks; without
    it each chunk becomes its own reference frame and "middle of the distribution" means something
    different in every chunk.
    """
    if not nll:
        return []
    if mode == "low":
        return [-value for value in nll]
    if mode == "high":
        return list(nll)
    centre = statistics.median(nll) if median is None else median
    return [-abs(value - centre) for value in nll]


def sequence_nll(
    model: Any, tokenizer: Any, texts: Sequence[str], max_length: int, device: Any
) -> list[float]:
    """Mean token negative log likelihood per sequence, with padding excluded."""
    import torch

    with torch.no_grad():
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        logits = model(**encoded).logits.float()

        labels = encoded["input_ids"][:, 1:]
        mask = encoded["attention_mask"][:, 1:].bool()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        token_nll = -logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll.masked_fill(~mask, 0.0)
        # clamp so an empty or single-token sequence divides by one rather than zero.
        return (token_nll.sum(-1) / mask.sum(-1).clamp(min=1)).tolist()


@register_scorer("perplexity")
class PerplexityScorer(Scorer):
    """Score documents by how surprising they are to a reference model.

    The median for ``mid`` is taken over the records being scored, so it describes this pool rather
    than an arbitrary sample of it.
    """

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-135M",
        *,
        mode: str = "mid",
        max_length: int = 512,
        batch_size: int = 8,
        dtype: str = "auto",
        progress: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = model
        self.mode = mode
        self.max_length = max_length
        self.batch_size = batch_size
        self.dtype = dtype
        self.progress = progress
        self._loaded: tuple[Any, Any] | None = None
        self._median: float | None = None

    def _model(self) -> tuple[Any, Any]:
        """Load once per scorer. Reuses the evaluator's loader so padding, dtype and device
        placement cannot drift between scoring and evaluation."""
        if self._loaded is None:
            from medsel.eval.runner import load_model

            self._loaded = load_model(self.model_name, dtype=self.dtype)
        return self._loaded

    def fit(self, records: Sequence[Any]) -> None:
        """Fix the pool median that ``mid`` measures distance from.

        Only ``mid`` needs it. ``low`` and ``high`` are monotone in the raw likelihood, so they
        rank identically whether scored in one call or in chunks.
        """
        if self.mode == "mid" and records:
            self._median = statistics.median(self._negative_log_likelihood(records))

    def _negative_log_likelihood(self, records: Sequence[Any]) -> list[float]:
        from tqdm.auto import tqdm

        model, tokenizer = self._model()
        model.eval()
        device = next(model.parameters()).device

        texts = [record_text(record) for record in records]
        nll: list[float] = []
        batches = range(0, len(texts), self.batch_size)
        for start in tqdm(
            batches, desc=f"perplexity:{self.mode}", unit="batch", disable=not self.progress
        ):
            nll.extend(
                sequence_nll(
                    model,
                    tokenizer,
                    texts[start : start + self.batch_size],
                    self.max_length,
                    device,
                )
            )
        return nll

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []
        return rank_by_mode(self._negative_log_likelihood(records), self.mode, self._median)

    def __repr__(self) -> str:
        return f"PerplexityScorer(model={self.model_name!r}, mode={self.mode!r})"
