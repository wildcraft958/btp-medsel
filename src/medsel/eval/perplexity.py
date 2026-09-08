"""Held-out perplexity, as a sensitive measure of what CPT data did to a model.

Multiple-choice accuracy is a blunt instrument for this project's question. On MedMCQA a 95%
interval is 2.9 points wide, and a small continual-pretraining run does not move accuracy that far,
so a comparison between selection methods scored only on accuracy will report "no difference"
whether or not a difference exists.

Perplexity is continuous, is computed over every token rather than over one choice per question,
and responds to a small amount of training. Two measurements matter here and they answer different
questions:

- **Held-out corpus perplexity** asks whether the model learned the domain at all. It has to be
  measured on documents no selection method chose, or the winner is whichever method happened to
  pick the evaluation text.
- **Target perplexity** asks whether *targeting* worked. If a scorer selects PubMed text that
  resembles MedMCQA, the model trained on it should find MedMCQA text less surprising than a model
  trained on a random subset does. That is the project's thesis measured directly, rather than
  inferred from a downstream score.

A lower number is better in both cases, and neither is comparable across tokenizers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["PerplexityReport", "mean_token_nll", "evaluate_perplexity"]


@dataclass
class PerplexityReport:
    """Mean token loss over one set of documents, and its exponential."""

    name: str
    n_documents: int
    mean_token_nll: float

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_token_nll)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_documents": self.n_documents,
            "mean_token_nll": round(self.mean_token_nll, 6),
            "perplexity": round(self.perplexity, 4),
        }


def mean_token_nll(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int = 512,
    batch_size: int = 8,
    progress: bool = False,
) -> float | None:
    """Mean negative log likelihood per token across every text, or ``None`` for no texts.

    Weighted by token rather than by document, so a long document contributes proportionally more
    than a short one. Averaging per-document losses would let a handful of very short documents
    dominate a corpus number.
    """
    if not texts:
        return None

    import torch
    from tqdm.auto import tqdm

    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    starts = range(0, len(texts), batch_size)

    with torch.no_grad():
        for start in tqdm(starts, desc="perplexity", unit="batch", disable=not progress):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
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
            total_nll += float(token_nll.masked_fill(~mask, 0.0).sum())
            total_tokens += int(mask.sum())

    if total_tokens == 0:
        return None
    return total_nll / total_tokens


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    name: str,
    **kwargs: Any,
) -> PerplexityReport | None:
    """Wrap :func:`mean_token_nll` in a named report, or ``None`` when there is nothing to score."""
    nll = mean_token_nll(model, tokenizer, texts, **kwargs)
    if nll is None:
        return None
    return PerplexityReport(name=name, n_documents=len(texts), mean_token_nll=nll)
