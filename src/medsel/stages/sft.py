"""Supervised fine-tuning. Not yet implemented.

Planned shape, so that filling this in stays an isolated change:

- Source: the three QA loaders, rendered to instruction-response text by ``medsel.prompts``.
- Trainer: ``trl.SFTTrainer``, with loss masked to the response span so the model is not
  rewarded for reproducing the question.
- Selection: an SFT-specific policy scoring instruction quality and task utility, as opposed to
  the CPT policy's emphasis on breadth and long-tail coverage.

One open design question from the project whiteboard, recorded here so it is not lost: whether to
train the instruction stage with a *combined* objective - next-token loss on raw biomedical text
alongside instruction loss on QA pairs - rather than running CPT and SFT strictly in sequence.
The claim to test is that mixing retains domain knowledge that pure SFT erodes.
"""

from __future__ import annotations

from typing import Any, ClassVar

from medsel.stages.base import Stage, StageResult

__all__ = ["SFTStage"]

_MESSAGE = (
    "SFT is not implemented yet. See src/medsel/stages/sft.py for the intended design "
    "(trl.SFTTrainer over the QA loaders with response-only loss)."
)


class SFTStage(Stage):
    name: ClassVar[str] = "sft"

    def prepare(self) -> Any:
        raise NotImplementedError(_MESSAGE)

    def run(self) -> StageResult:
        raise NotImplementedError(_MESSAGE)
