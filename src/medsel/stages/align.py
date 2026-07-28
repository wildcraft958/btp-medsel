"""Preference optimisation (DPO/RLHF). Not yet implemented.

Planned shape:

- Trainer: ``trl.DPOTrainer``, which needs no separate reward model and is the practical choice
  at the scale this project can afford.
- Data: no medical preference dataset is chosen yet. This is the genuine blocker, not the
  training code - the proposal assumes preference data "if available", and for the public
  datasets we have there is none. Candidates to evaluate: constructing pairs from PubMedQA long
  answers versus generated distractors, or an existing open medical preference set.
- Selection: a policy favouring pairs that improve reasoning and safety, distinct from both the
  CPT and SFT policies.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.stages.base import Stage, StageResult

__all__ = ["AlignStage"]

_MESSAGE = (
    "Alignment is not implemented yet, and is blocked on choosing a medical preference dataset "
    "rather than on training code. See src/medsel/stages/align.py."
)


class AlignStage(Stage):
    name: ClassVar[str] = "align"

    def prepare(self) -> Any:
        raise NotImplementedError(_MESSAGE)

    def run(self) -> StageResult:
        raise NotImplementedError(_MESSAGE)
