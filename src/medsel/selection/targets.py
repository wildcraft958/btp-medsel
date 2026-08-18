"""Scorers that rank records against a target distribution.

Most scorers judge a document on its own: how long it is, how surprising it is. A targeted scorer
asks a different question, how much this document looks like the thing we want the model to be good
at, and that is what makes multi-target selection possible at all.

Everything shared between such scorers lives here, above all the guard on which split the target is
drawn from. Fitting a selection distribution to the split the model is scored on inflates every
downstream number without improving the model, it is easy to do from a config file, and it is
invisible in the results. One implementation of that check, inherited, is the only way it stays
consistent as scorers are added.
"""

from __future__ import annotations

from collections.abc import Iterable

from medsel.selection.base import Scorer, record_text

__all__ = ["TargetedScorer", "DEFAULT_TARGET_SPLITS"]

# Always a training split. The point of a target distribution is to describe the task, so drawing
# it from the split the model is scored on would tune selection to the test items themselves.
# PubMedQA has only one split and its evaluation slice is the first 500 rows, so its target starts
# after them.
DEFAULT_TARGET_SPLITS = {
    "medqa": "train",
    "medmcqa": "train",
    "pubmedqa": "train[500:]",
}


def _parse_slice(spec: str) -> tuple[int, float]:
    start, _, stop = spec.partition(":")
    return int(start or 0), float(stop or "inf")


def _overlaps(target_split: str, eval_split: str) -> bool:
    """Whether two split specifications can share a row."""
    target_base, _, target_slice = target_split.partition("[")
    eval_base, _, eval_slice = eval_split.partition("[")
    if target_base != eval_base:
        return False
    if not target_slice or not eval_slice:
        return True
    target_start, target_stop = _parse_slice(target_slice.rstrip("]"))
    eval_start, eval_stop = _parse_slice(eval_slice.rstrip("]"))
    return target_start < eval_stop and eval_start < target_stop


class TargetedScorer(Scorer):
    """Base for scorers that need a target distribution.

    ``target`` is either a loader name (``"medmcqa"``, ``"medqa"``, ``"pubmedqa"``) or an explicit
    iterable of strings. The loader form is what experiments use; the explicit form keeps subclasses
    testable without network access.
    """

    def __init__(
        self,
        target: str | Iterable[str] = "medmcqa",
        *,
        target_split: str | None = None,
        target_limit: int = 2000,
    ) -> None:
        self.target = target
        self.target_split = target_split
        self.target_limit = target_limit

    def resolve_target_split(self) -> str | None:
        """The split the target text is drawn from, or ``None`` for explicit texts.

        Raises if that split can share rows with the one the model is evaluated on.
        """
        if isinstance(self.target, str):
            from medsel.registry import get_loader

            loader = get_loader(self.target)
            split = self.target_split or DEFAULT_TARGET_SPLITS.get(self.target, "train")
            eval_split = str(getattr(loader, "eval_split", "") or "")

            if eval_split and _overlaps(split, eval_split):
                raise ValueError(
                    f"target {self.target}:{split} overlaps {self.target}:{eval_split}, which is "
                    f"the split this task is scored on. Selecting towards the evaluation set "
                    f"inflates the result without improving the model. Use a training split."
                )
            return split
        return None

    def target_texts(self) -> list[str]:
        """The target distribution as plain text, ready to be embedded or counted."""
        if not isinstance(self.target, str):
            texts = [str(text) for text in self.target]
        else:
            from medsel.prompts import render_prompt
            from medsel.registry import get_loader
            from medsel.schema import QAExample

            loader = get_loader(self.target)
            split = self.resolve_target_split()
            assert split is not None
            texts = [
                render_prompt(example) if isinstance(example, QAExample) else record_text(example)
                for example in loader.load(split, limit=self.target_limit)
            ]

        if not texts:
            raise ValueError(
                f"target {self.target!r} produced no text, so there is no distribution to select "
                "towards"
            )
        return texts

    def _target_repr(self) -> str:
        if isinstance(self.target, str):
            return repr(self.target)
        return f"<{len(list(self.target))} texts>"
