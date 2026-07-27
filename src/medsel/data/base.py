"""Shared loader plumbing: split validation, Hub access, progress-wrapped iteration, stats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from typing import Any, ClassVar

from tqdm.auto import tqdm

from medsel.schema import CorpusDoc, QAExample

__all__ = ["BaseLoader", "letter_key"]

_LETTERS = "ABCDEFGHIJ"


def letter_key(index: int) -> str:
    """Map a 0-based option index to its letter, e.g. ``2 -> "C"``."""
    try:
        return _LETTERS[index]
    except IndexError:
        raise ValueError(f"option index {index} exceeds supported width {len(_LETTERS)}") from None


class BaseLoader(ABC):
    """Base for every dataset loader.

    Subclasses declare where the data lives and how one raw row becomes a normalised record.
    Everything else - split checking, streaming, progress bars, cache keys - is inherited.

    Bump :attr:`normalizer_version` whenever :meth:`normalize` changes shape or semantics; the
    parquet cache keys on it and will rebuild rather than serve stale records.
    """

    name: ClassVar[str] = ""
    hf_id: str = ""  # class-level default, shadowed per instance by __init__
    default_config: ClassVar[str | None] = None
    splits: ClassVar[tuple[str, ...]] = ()
    unlabeled_splits: ClassVar[frozenset[str]] = frozenset()
    record_type: ClassVar[type] = QAExample
    normalizer_version: ClassVar[int] = 1

    def __init__(
        self,
        config: str | None = None,
        revision: str | None = None,
        hf_id: str | None = None,
    ) -> None:
        self.config = config if config is not None else self.default_config
        self.revision = revision
        self.hf_id = hf_id or type(self).hf_id

    def __repr__(self) -> str:
        cfg = f", config={self.config!r}" if self.config else ""
        return f"{type(self).__name__}(hf_id={self.hf_id!r}{cfg})"

    @property
    def cache_key(self) -> str:
        """Identity of this loader's output, used to name and validate cache entries."""
        return f"{self.name}-{self.config}-v{self.normalizer_version}"

    def check_split(self, split: str) -> None:
        if self.splits and split not in self.splits:
            raise ValueError(
                f"{self.name} has no split {split!r}; available: {', '.join(self.splits)}"
            )

    def has_labels(self, split: str) -> bool:
        """False for splits whose gold answers the publisher withholds."""
        return split not in self.unlabeled_splits

    def raw(self, split: str, streaming: bool = False) -> Any:
        """The untouched Hub dataset for ``split``."""
        from datasets import load_dataset

        self.check_split(split)
        return load_dataset(
            self.hf_id,
            self.config,
            split=split,
            streaming=streaming,
            revision=self.revision,
        )

    @abstractmethod
    def normalize(self, row: dict[str, Any], idx: int, split: str) -> QAExample | CorpusDoc:
        """Convert one raw row into a normalised record."""

    def load(
        self,
        split: str,
        limit: int | None = None,
        streaming: bool = False,
        progress: bool = True,
    ) -> Iterator[QAExample | CorpusDoc]:
        """Yield normalised records, with a progress bar by default.

        Streaming avoids materialising the dataset locally, at the cost of an unknown total.
        """
        dataset = self.raw(split, streaming=streaming)

        total = limit
        if total is None and not streaming:
            try:
                total = len(dataset)
            except TypeError:
                total = None

        with tqdm(
            total=total,
            desc=f"{self.name}:{split}",
            unit="ex",
            disable=not progress,
            leave=False,
        ) as bar:
            for idx, row in enumerate(dataset):
                if limit is not None and idx >= limit:
                    break
                yield self.normalize(row, idx, split)
                bar.update(1)

    def stats(self, split: str, limit: int | None = None, progress: bool = True) -> dict[str, Any]:
        """Summary counts. Cheap enough to run on any split, useful as a sanity check."""
        n = 0
        labeled = 0
        option_counts: Counter[int] = Counter()
        subjects: Counter[str] = Counter()
        total_chars = 0

        for record in self.load(split, limit=limit, progress=progress):
            n += 1
            if isinstance(record, QAExample):
                labeled += record.is_labeled
                option_counts[record.n_options] += 1
                subject = record.labels.get("subject_name") or ""
                if subject:
                    subjects[subject] += 1
                total_chars += len(record.question)
            else:
                total_chars += len(record.full_text)

        return {
            "source": self.name,
            "split": split,
            "hf_id": self.hf_id,
            "config": self.config,
            "n_examples": n,
            "n_labeled": labeled,
            "labeled_fraction": round(labeled / n, 4) if n else 0.0,
            "option_counts": dict(sorted(option_counts.items())),
            "mean_chars": round(total_chars / n, 1) if n else 0.0,
            "top_subjects": dict(subjects.most_common(10)),
        }
