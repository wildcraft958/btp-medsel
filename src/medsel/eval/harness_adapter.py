"""lm-evaluation-harness adapter. Not yet implemented.

Two evaluators, on purpose, serving different needs:

- :mod:`medsel.eval.mcq` is the inner loop. It shares our loaders and prompt templates, so a
  change to either shows up in evaluation immediately, and it can break accuracy down by clinical
  subject - which is the capability-retention metric the proposal is actually built around.
- lm-evaluation-harness is the outer loop, for the paper. Its numbers are directly comparable to
  published MedQA / MedMCQA / PubMedQA results because it uses the same prompts and few-shot
  conventions everyone else reports under.

Our own numbers are internally consistent but should not be quoted against published figures:
prompt formatting, few-shot count, and PubMedQA's test split all differ. Keeping the harness path
open is what makes the eventual comparison honest.

Intended shape: register each loader as a harness task (multiple_choice output type, ``doc_to_text``
from :func:`medsel.prompts.render_prompt`, ``doc_to_choice`` from the option mapping), then invoke
``lm_eval.simple_evaluate`` against a local checkpoint. Needs ``lm-eval`` as an optional extra.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_harness"]

_MESSAGE = (
    "lm-evaluation-harness integration is not implemented yet. Use medsel.eval.run_evaluation "
    "for the in-repo evaluator. See src/medsel/eval/harness_adapter.py for the intended design."
)


def run_harness(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MESSAGE)
