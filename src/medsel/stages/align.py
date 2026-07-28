"""Preference optimisation (DPO/RLHF). Not yet implemented.

Planned shape:

- Trainer: ``trl.DPOTrainer``, which needs no separate reward model and is the practical choice
  at the scale this project can afford.
- Data: still the genuine blocker, but narrower than it was. A 2026 survey of the field found no
  open *medical* preference corpus, so pairs must still be constructed from PubMedQA long answers
  or MedMCQA explanations. What it did find is five usable general-domain corpora (TuluDPO, ORPO,
  UltraFeedback, HelpSteer, Code-Preference-Pairs), which can serve as a replay mixture the way
  general text does in CPT. Two findings shape any design here: a curated 190k-pair mixture beat
  the strongest single 280k source, and only 70 to 80 percent of pairs in these corpora agree with
  a reward-model ordering, so the labels are not ground truth. See docs/literature_review.md
  Section 5.
- Selection: a policy favouring pairs that improve reasoning and safety, distinct from both the
  CPT and SFT policies.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.stages.base import Stage, StageResult

__all__ = ["AlignStage"]

_MESSAGE = (
    "Alignment is not implemented yet. The blocker is the preference dataset rather than the "
    "training code: no open medical preference corpus exists, so pairs must be constructed. "
    "See src/medsel/stages/align.py and docs/literature_review.md Section 5."
)


class AlignStage(Stage):
    name: ClassVar[str] = "align"

    def prepare(self) -> Any:
        raise NotImplementedError(_MESSAGE)

    def run(self) -> StageResult:
        raise NotImplementedError(_MESSAGE)
