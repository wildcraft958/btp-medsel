"""MedQA: USMLE-style clinical vignettes with four options.

Owner: Debmalya. Paper: Jin et al., 2021.

Raw layout nests everything under a ``data`` object with capitalised, space-separated keys.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.data.base import BaseLoader
from medsel.registry import register_loader
from medsel.schema import QAExample, make_uid

__all__ = ["MedQALoader"]


@register_loader("medqa")
class MedQALoader(BaseLoader):
    """Loader for ``openlifescienceai/medqa``.

    Two quirks worth knowing. The split is named ``dev``, not ``validation``, unlike MedMCQA.
    And the top-level ``subject_name`` column is present but empty for every row, so it is
    dropped rather than propagated as a meaningless empty label - MedQA offers no subject
    breakdown for capability-retention analysis.
    """

    hf_id = "openlifescienceai/medqa"
    splits: ClassVar[tuple[str, ...]] = ("train", "dev", "test")
    eval_split: ClassVar[str] = "test"
    normalizer_version: ClassVar[int] = 1

    def normalize(self, row: dict[str, Any], idx: int, split: str) -> QAExample:
        payload = row.get("data") or {}

        options = {str(k): str(v) for k, v in (payload.get("Options") or {}).items()}
        correct_option = (payload.get("Correct Option") or "").strip()
        answer_key = correct_option if correct_option in options else None

        labels = {"source_id": str(row.get("id", ""))}
        subject = (row.get("subject_name") or "").strip()
        if subject:
            labels["subject_name"] = subject

        return QAExample(
            uid=make_uid(self.name, split, idx),
            source=self.name,
            split=split,
            question=str(payload["Question"]).strip(),
            options=options or None,
            answer_key=answer_key,
            answer_text=(payload.get("Correct Answer") or "").strip() or None,
            labels=labels,
        )
