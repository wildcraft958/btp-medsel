"""MedMCQA: 194k AIIMS/NEET-PG multiple-choice questions across 21 subjects.

Owner: Animesh. Paper: Pal et al., CHIL 2022.

Raw layout is flat: ``question``, ``opa``..``opd``, and ``cop`` as a 0-based index.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.data.base import BaseLoader, letter_key
from medsel.registry import register_loader
from medsel.schema import QAExample, make_uid

__all__ = ["MedMCQALoader"]

_OPTION_COLUMNS = ("opa", "opb", "opc", "opd")


@register_loader("medmcqa")
class MedMCQALoader(BaseLoader):
    """Loader for ``openlifescienceai/medmcqa``.

    The ``test`` split ships ``cop = -1`` for every row - the publisher withholds gold answers for
    leaderboard integrity. Every MedMCQA number in the literature is therefore measured on
    ``validation`` (the NEET-PG set), and so is ours. Examples from ``test`` normalise with
    ``answer_key=None``; the evaluator refuses them outright rather than scoring against a
    placeholder.
    """

    hf_id = "openlifescienceai/medmcqa"
    splits: ClassVar[tuple[str, ...]] = ("train", "validation", "test")
    unlabeled_splits: ClassVar[frozenset[str]] = frozenset({"test"})
    eval_split: ClassVar[str] = "validation"
    normalizer_version: ClassVar[int] = 1

    def normalize(self, row: dict[str, Any], idx: int, split: str) -> QAExample:
        options = {
            letter_key(position): str(row.get(column) or "")
            for position, column in enumerate(_OPTION_COLUMNS)
        }

        cop = row.get("cop")
        answer_key = (
            letter_key(int(cop))
            if isinstance(cop, int) and 0 <= cop < len(_OPTION_COLUMNS)
            else None
        )

        labels = {"source_id": str(row.get("id", ""))}
        for column in ("subject_name", "topic_name", "choice_type"):
            value = row.get(column)
            if value:
                labels[column] = str(value)

        explanation = (row.get("exp") or "").strip()

        return QAExample(
            uid=make_uid(self.name, split, idx),
            source=self.name,
            split=split,
            question=str(row["question"]).strip(),
            options=options,
            answer_key=answer_key,
            rationale=explanation or None,
            labels=labels,
        )
