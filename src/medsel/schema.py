"""Unified record types.

The three QA datasets ship mutually incompatible raw schemas (nested dicts, flat columns,
sequence-of-context objects). Every loader normalises into :class:`QAExample` so that prompt
rendering, selection scoring, and evaluation never branch on which dataset they were handed.
:class:`CorpusDoc` plays the same role for raw CPT text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["QAExample", "CorpusDoc", "make_uid"]


def make_uid(source: str, split: str, idx: int) -> str:
    """Stable per-example identifier. Zero-padded so lexical sort matches numeric order."""
    return f"{source}/{split}/{idx:06d}"


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class QAExample:
    """One question-answering item, normalised across MedQA / MedMCQA / PubMedQA.

    ``answer_key`` is ``None`` when the source withholds labels (MedMCQA's ``test`` split ships
    ``cop = -1``). Consumers must check :attr:`is_labeled` rather than assuming a gold answer.
    """

    uid: str
    source: str
    split: str
    question: str
    options: dict[str, str] | None = None
    answer_key: str | None = None
    answer_text: str | None = None
    contexts: list[str] = field(default_factory=list)
    rationale: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("uid", "source", "split", "question"):
            _require_text(getattr(self, name), name)

        if self.options is not None:
            options = {str(k): str(v) for k, v in self.options.items()}
            if not options:
                raise ValueError("options must be None or a non-empty mapping")
            object.__setattr__(self, "options", options)

        if self.answer_key is not None:
            if self.options is None:
                raise ValueError("answer_key set but options is None")
            if self.answer_key not in self.options:
                raise ValueError(
                    f"answer_key {self.answer_key!r} not in options {sorted(self.options)}"
                )
            if self.answer_text is None:
                object.__setattr__(self, "answer_text", self.options[self.answer_key])

        object.__setattr__(self, "contexts", [str(c) for c in self.contexts])
        object.__setattr__(self, "labels", {str(k): str(v) for k, v in self.labels.items()})

    @property
    def is_labeled(self) -> bool:
        return self.answer_key is not None

    @property
    def n_options(self) -> int:
        return len(self.options) if self.options else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QAExample:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CorpusDoc:
    """One raw text document for continual pretraining."""

    uid: str
    source: str
    text: str
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("uid", "source", "text"):
            _require_text(getattr(self, name), name)

    @property
    def full_text(self) -> str:
        """Title-prefixed body. Titles carry real signal in PubMed, so CPT trains on both."""
        return f"{self.title}\n\n{self.text}" if self.title.strip() else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusDoc:
        return cls(**data)
