"""What a record costs to train on.

A selection budget can be counted in records or in tokens, and the two are not interchangeable.
Records are the convenient unit and tokens are the meaningful one, because training consumes tokens.
Measured on PubMed, the ``length`` scorer spends 2.06 times the tokens of ``random`` for the same
500 records, so a comparison held fixed in records silently hands one scorer twice the training
data.

Token counting needs a tokenizer and therefore the ``train`` extra, which is why it lives here
rather than in ``selector.py``. That module stays free of heavy imports so the budget arithmetic can
be tested and used without a model present.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from medsel.selection.base import record_text

__all__ = ["BUDGET_UNITS", "token_costs", "load_tokenizer"]

BUDGET_UNITS = ("records", "tokens")


def load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)


def token_costs(
    records: Sequence[Any],
    tokenizer: Any,
    *,
    render: Callable[[Any], str] | None = None,
    batch_size: int = 256,
    progress: bool = True,
) -> list[int]:
    """Token count per record, in input order.

    Counted without special tokens: the cost being modelled is the text's contribution to a packed
    training stream, where per-example separators are a property of packing rather than of the
    document.

    ``render`` maps a record to the text whose tokens should be counted. The default is
    :func:`record_text`, which returns ``full_text`` for corpus documents and ``question`` for QA
    examples. For SFT cost accounting, pass a renderer that produces the full prompt and
    completion so the budget reflects what the trainer actually sees.

    Every count is floored at one. A genuinely empty document would otherwise cost nothing and fit
    any budget an unlimited number of times, which the selectors reject outright.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not records:
        return []

    from tqdm.auto import tqdm

    to_text = render or record_text
    texts = [to_text(record) for record in records]
    counts: list[int] = []
    starts = range(0, len(texts), batch_size)
    for start in tqdm(starts, desc="tokens", unit="batch", disable=not progress):
        encoded = tokenizer(texts[start : start + batch_size], add_special_tokens=False)
        counts.extend(max(1, len(ids)) for ids in encoded["input_ids"])
    return counts
