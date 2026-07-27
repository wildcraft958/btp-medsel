"""PubMedQA: yes/no/maybe research questions grounded in PubMed abstracts.

Owner: Arkajyoti. Paper: Jin et al., EMNLP 2019.

Unlike the other two this is not a 4-way MCQ; the label space is a fixed three-way decision, and
each item carries the source abstract as retrieval context.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.data.base import BaseLoader
from medsel.registry import register_loader
from medsel.schema import QAExample, make_uid

__all__ = ["PubMedQALoader", "DECISION_OPTIONS"]

# Fixed label space. Order is load-bearing: it fixes the letter each decision maps to.
DECISION_OPTIONS: dict[str, str] = {"A": "yes", "B": "no", "C": "maybe"}
_DECISION_TO_KEY = {v: k for k, v in DECISION_OPTIONS.items()}

_UNLABELED_CONFIG = "pqa_unlabeled"


@register_loader("pubmedqa")
class PubMedQALoader(BaseLoader):
    """Loader for ``qiaojin/PubMedQA``.

    Three configs, each shipping a single ``train`` split:

    - ``pqa_labeled`` (1k, expert-annotated) - the evaluation set
    - ``pqa_artificial`` (211k, auto-labelled) - SFT-scale training data
    - ``pqa_unlabeled`` (61k, no ``final_decision``) - long answers only

    Labelling is a property of the *config*, not the split, so :meth:`has_labels` keys on config
    rather than inheriting the split-based default.

    Note on comparability: the published 500-question test set is defined by a ground-truth file in
    the authors' repository, which the Hub mirror does not carry. ``pqa_labeled`` arrives as one
    undivided 1000-row split, so slice it deterministically (``train[:500]``) and treat absolute
    numbers as internally consistent rather than directly comparable to the paper.
    """

    hf_id = "qiaojin/PubMedQA"
    default_config: ClassVar[str | None] = "pqa_labeled"
    splits: ClassVar[tuple[str, ...]] = ("train",)
    eval_split: ClassVar[str] = "train[:500]"
    normalizer_version: ClassVar[int] = 1

    def has_labels(self, split: str) -> bool:
        return self.config != _UNLABELED_CONFIG

    def normalize(self, row: dict[str, Any], idx: int, split: str) -> QAExample:
        context = row.get("context") or {}
        contexts = [str(c) for c in (context.get("contexts") or [])]

        decision = str(row.get("final_decision") or "").strip().lower()
        answer_key = _DECISION_TO_KEY.get(decision)

        labels = {"source_id": str(row.get("pubid", "")), "config": str(self.config)}
        # MeSH terms drive rare-disease and specialty coverage in the selection stage, so they are
        # kept verbatim rather than summarised away.
        meshes = [str(m) for m in (context.get("meshes") or [])]
        if meshes:
            labels["meshes"] = "; ".join(meshes)
        sections = [str(s) for s in (context.get("labels") or [])]
        if sections:
            labels["sections"] = "; ".join(sections)

        long_answer = str(row.get("long_answer") or "").strip()

        return QAExample(
            uid=make_uid(self.name, split, idx),
            source=self.name,
            split=split,
            question=str(row["question"]).strip(),
            options=dict(DECISION_OPTIONS),
            answer_key=answer_key,
            contexts=contexts,
            rationale=long_answer or None,
            labels=labels,
        )
