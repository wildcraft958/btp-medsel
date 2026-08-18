"""Cosine similarity to the centroid of a target distribution.

A cheap, non-gradient stand-in for influence: rather than asking which documents would move the
model the way the target does, it asks which documents *look* like the target in the reference
model's own representation space. Far cheaper than TracIn, and it gives a gradient method something
to beat before anyone spends days extracting gradients.

Ported from the parallel CPT selection work on the lab machine, which also has a `facility` mode
running greedy facility location for coverage. That one is deliberately not here. Its score for a
record is the record's rank in a greedy ordering over whatever else is in the same call, so the
numbers are relative to the batch and are not comparable between batches. It is a set-selection
algorithm rather than a per-record score, which makes it a *selector* in this codebase's vocabulary,
alongside `select_top_k` and `select_stratified`, not a scorer. Bolting it onto this interface would
silently produce chunk-relative scores under `medsel select --stream`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from medsel.selection.base import record_text, register_scorer
from medsel.selection.targets import TargetedScorer

__all__ = ["EmbedSimilarityScorer", "unit_centroid", "mean_pooled"]


def unit_centroid(embeddings: np.ndarray) -> np.ndarray:
    """The average direction of a set of embeddings, as a unit vector."""
    if len(embeddings) == 0:
        raise ValueError("no embeddings to average, so there is no centroid to compare against")
    centre = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(centre))
    if norm < 1e-12:
        raise ValueError(
            "the target embeddings cancel out, leaving a centroid with no direction. That means "
            "the target set has no shared orientation to select towards."
        )
    return centre / norm


def mean_pooled(model: Any, tokenizer: Any, texts: Sequence[str], max_length: int, device: Any):
    """Mask-aware mean of the last hidden state, L2 normalised so a dot product is a cosine."""
    import torch

    with torch.no_grad():
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        hidden = model(**encoded, output_hidden_states=True).hidden_states[-1].float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        # Padding positions carry real activations, so they have to be masked out of the mean
        # rather than merely divided away.
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()


@register_scorer("embed_similarity")
class EmbedSimilarityScorer(TargetedScorer):
    """Rank documents by cosine similarity to the target centroid.

    The reference model only has to produce representations, not good ones, so the smoke model is a
    usable default. A model already adapted to the domain will separate medical text more sharply.
    """

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-135M",
        *,
        target: str | Iterable[str] = "medmcqa",
        target_split: str | None = None,
        target_limit: int = 2000,
        max_length: int = 512,
        batch_size: int = 8,
        dtype: str = "auto",
        progress: bool = True,
    ) -> None:
        super().__init__(target, target_split=target_split, target_limit=target_limit)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.dtype = dtype
        self.progress = progress
        self._loaded: tuple[Any, Any] | None = None
        self._centroid: np.ndarray | None = None

    def _model(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from medsel.eval.runner import load_model

            self._loaded = load_model(self.model_name, dtype=self.dtype)
        return self._loaded

    def _embed(self, texts: Sequence[str], desc: str) -> np.ndarray:
        from tqdm.auto import tqdm

        model, tokenizer = self._model()
        model.eval()
        device = next(model.parameters()).device

        batches = range(0, len(texts), self.batch_size)
        chunks = [
            mean_pooled(
                model,
                tokenizer,
                texts[start : start + self.batch_size],
                self.max_length,
                device,
            )
            for start in tqdm(batches, desc=desc, unit="batch", disable=not self.progress)
        ]
        return np.concatenate(chunks, axis=0)

    def fit(self, records: Sequence[Any]) -> None:
        """Embed the target and keep its centroid.

        ``records`` is unused: the centroid describes the target, not the pool, so it is the same
        whichever records happen to arrive first. It stays in the signature because that is what
        makes the scorer usable under streaming without a special case.
        """
        self._centroid = unit_centroid(self._embed(self.target_texts(), "embed:target"))

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []
        if self._centroid is None:
            self.fit(records)
        assert self._centroid is not None

        embeddings = self._embed([record_text(record) for record in records], "embed:pool")
        return (embeddings @ self._centroid).tolist()

    def __repr__(self) -> str:
        return f"EmbedSimilarityScorer(model={self.model_name!r}, target={self._target_repr()})"
