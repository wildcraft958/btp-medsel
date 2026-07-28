"""Sequence packing for continual pretraining.

CPT trains on fixed-length token blocks. Feeding one document per example instead would waste a
large fraction of every batch on padding, because PubMed abstracts are far shorter than a typical
context window. Packing concatenates the token stream and cuts it into uniform blocks, so almost
every position carries a real training signal.

The tokenizer is taken as a parameter rather than imported, which keeps this module free of any
transformers dependency and lets tests exercise the logic with a trivial stand-in.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from tqdm.auto import tqdm

from medsel.schema import CorpusDoc

__all__ = ["pack_documents", "pack_to_dataset", "TokenizerLike"]


class TokenizerLike(Protocol):
    """The slice of a tokenizer interface packing actually needs."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def pack_documents(
    docs: Iterable[CorpusDoc],
    tokenizer: TokenizerLike,
    block_size: int = 1024,
    add_eos: bool = True,
    drop_remainder: bool = True,
    progress: bool = True,
) -> Iterator[list[int]]:
    """Concatenate tokenised documents and yield blocks of exactly ``block_size`` tokens.

    Args:
        add_eos: Append the tokenizer's EOS after each document. Without a separator the model
            learns to run one abstract straight into the next, which is not a real continuation.
        drop_remainder: Discard the trailing partial block. Turn this off only for very small
            corpora such as smoke tests, where dropping it could leave nothing to train on -
            the short block it yields then breaks uniform batching.

    Raises:
        ValueError: If ``block_size`` is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    buffer: list[int] = []

    with tqdm(desc="pack", unit="block", disable=not progress, leave=False) as bar:
        for doc in docs:
            token_ids = list(tokenizer.encode(doc.full_text, add_special_tokens=False))
            if add_eos and eos_id is not None:
                token_ids.append(eos_id)
            buffer.extend(token_ids)

            while len(buffer) >= block_size:
                yield buffer[:block_size]
                del buffer[:block_size]
                bar.update(1)

        if buffer and not drop_remainder:
            yield buffer
            bar.update(1)


def pack_to_dataset(
    docs: Iterable[CorpusDoc],
    tokenizer: TokenizerLike,
    block_size: int = 1024,
    **kwargs: Any,
):
    """Materialise packed blocks as a ``datasets.Dataset`` ready for ``Trainer``.

    Labels are omitted deliberately: ``DataCollatorForLanguageModeling(mlm=False)`` derives them
    from ``input_ids`` at collation time, so storing a second copy would only waste memory.
    """
    from datasets import Dataset

    blocks = list(pack_documents(docs, tokenizer, block_size, **kwargs))
    if not blocks:
        raise ValueError(
            "packing produced no blocks - the corpus is smaller than one block "
            f"of {block_size} tokens; lower block_size or pass drop_remainder=False"
        )
    return Dataset.from_dict({"input_ids": blocks})
