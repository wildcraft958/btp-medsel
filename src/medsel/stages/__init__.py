"""Training stages.

The proposal's whole premise is that CPT, SFT and preference optimisation want *different* data,
so they are separate stages with separate selection policies rather than one parameterised
trainer. CPT is implemented; the other two are declared here so the pipeline shape is visible and
so filling them in is an isolated change.
"""

from __future__ import annotations

from medsel.stages.align import AlignStage
from medsel.stages.base import Stage, StageResult
from medsel.stages.cpt import CPTStage
from medsel.stages.sft import SFTStage

__all__ = ["Stage", "StageResult", "get_stage", "available_stages", "STAGES"]

# Fixed by the proposal's three-stage pipeline, so a plain mapping is honest here - unlike
# loaders, this set is not meant to grow.
STAGES: dict[str, type[Stage]] = {
    "cpt": CPTStage,
    "sft": SFTStage,
    "align": AlignStage,
}


def available_stages() -> list[str]:
    return list(STAGES)


def get_stage(name: str) -> type[Stage]:
    try:
        return STAGES[name]
    except KeyError:
        raise KeyError(
            f"unknown stage {name!r}; available: {', '.join(available_stages())}"
        ) from None
