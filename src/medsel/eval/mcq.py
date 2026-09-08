"""Multiple-choice evaluation by continuation log-likelihood.

Every option is scored by the total log-probability the model assigns to its continuation, and the
highest-scoring option is the prediction. Nothing is generated, so results do not depend on decoding
settings, and base models can be evaluated at all - which matters here, because the comparison this
project needs is a base model before and after CPT.

Two normalisations are reported, following the lm-evaluation-harness convention:

- ``accuracy``: raw summed log-probability. Biases toward short options.
- ``accuracy_norm``: divided by continuation token count, which removes most of that bias.

Under ``letter`` scoring the two coincide, since every continuation is one token.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tqdm.auto import tqdm

from medsel.eval.stats import accuracy_ci, subject_spread
from medsel.prompts.templates import TEMPLATE_VERSION, render_continuation, render_prompt
from medsel.schema import QAExample

# Subjects thinner than this are dropped from the spread summary. MedMCQA's rarest subjects carry
# a handful of validation items, where an accuracy is mostly sample-size artefact, and including
# them would inflate the variance that the retention claim is read from.
SUBJECT_MIN_N = 10

__all__ = ["OptionScores", "EvalReport", "score_example", "evaluate"]


@dataclass
class OptionScores:
    """Per-option log-probabilities for a single example."""

    uid: str
    gold_key: str | None
    total: dict[str, float]
    normalized: dict[str, float]

    @property
    def predicted_key(self) -> str:
        return max(self.total, key=lambda k: self.total[k])

    @property
    def predicted_key_norm(self) -> str:
        return max(self.normalized, key=lambda k: self.normalized[k])

    @property
    def correct(self) -> bool:
        return self.gold_key is not None and self.predicted_key == self.gold_key

    @property
    def correct_norm(self) -> bool:
        return self.gold_key is not None and self.predicted_key_norm == self.gold_key


@dataclass
class EvalReport:
    """Aggregate result of one evaluation run."""

    source: str
    split: str
    model: str
    mode: str
    n_scored: int
    n_skipped_unlabeled: int
    accuracy: float
    accuracy_norm: float
    template_version: int = TEMPLATE_VERSION
    # A 95% bootstrap interval, so a delta against a control can be told from noise. Measured on
    # the base control it is 5.1 points wide on MedQA and 8.6 on PubMedQA, wider than most
    # differences anyone would want to report.
    accuracy_ci: tuple[float, float] | None = None
    accuracy_norm_ci: tuple[float, float] | None = None
    by_subject: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Spread across subjects: aggregate accuracy can hold steady while a model trades one
    # capability for another, and this is the only per-capability signal these datasets ship.
    subject_spread: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuations: Sequence[str],
    device: Any,
) -> tuple[list[float], list[int]]:
    """Total log-probability of each continuation given a shared prompt.

    All options for one example are scored in a single forward pass - they share a prompt, differ
    only in the tail, and there are rarely more than five of them.
    """
    import torch

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not prompt_ids:
        # Scoring below reads the logits at position start-1, so an empty prompt would index -1
        # and silently sum an empty slice to 0.0 for every option, making the first option win
        # every time. Fail instead of reporting that as an accuracy.
        raise ValueError(
            "prompt tokenised to zero tokens; continuation log-probabilities cannot be scored "
            "without at least one preceding token"
        )

    sequences, cont_lengths = [], []
    for continuation in continuations:
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not cont_ids:  # degenerate option, e.g. an empty MedMCQA distractor
            cont_ids = [tokenizer.eos_token_id]
        sequences.append(prompt_ids + cont_ids)
        cont_lengths.append(len(cont_ids))

    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id or 0

    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, : len(seq)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    log_probs = torch.log_softmax(logits.float(), dim=-1)

    totals = []
    for row, seq in enumerate(sequences):
        # Token at position t is predicted by the logits at position t-1.
        start, end = len(seq) - cont_lengths[row], len(seq)
        target = input_ids[row, start:end]
        chosen = log_probs[row, start - 1 : end - 1].gather(-1, target.unsqueeze(-1))
        totals.append(float(chosen.sum()))

    return totals, cont_lengths


def score_example(
    model: Any,
    tokenizer: Any,
    example: QAExample,
    mode: str = "letter",
    include_contexts: bool = True,
    device: Any = None,
) -> OptionScores:
    """Score every option of one example."""
    if not example.options:
        raise ValueError(f"{example.uid} has no options")

    keys = sorted(example.options)
    prompt = render_prompt(example, include_contexts=include_contexts)
    continuations = [render_continuation(example, key, mode=mode) for key in keys]

    device = device if device is not None else next(model.parameters()).device
    totals, lengths = _sequence_logprob(model, tokenizer, prompt, continuations, device)

    return OptionScores(
        uid=example.uid,
        gold_key=example.answer_key,
        total=dict(zip(keys, totals, strict=True)),
        normalized={
            key: total / length for key, total, length in zip(keys, totals, lengths, strict=True)
        },
    )


def evaluate(
    model: Any,
    tokenizer: Any,
    examples: Iterable[QAExample],
    source: str,
    split: str,
    model_name: str,
    mode: str = "letter",
    include_contexts: bool = True,
    progress: bool = True,
    total: int | None = None,
) -> EvalReport:
    """Score a set of examples and aggregate, including a per-subject breakdown.

    Unlabeled examples are counted and skipped rather than scored against a placeholder.
    """
    model.eval()
    device = next(model.parameters()).device

    n_correct = n_correct_norm = n_scored = n_skipped = 0
    subject_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    with tqdm(total=total, desc=f"eval {source}:{split}", unit="ex", disable=not progress) as bar:
        for example in examples:
            bar.update(1)
            if not example.is_labeled:
                n_skipped += 1
                continue

            scores = score_example(
                model,
                tokenizer,
                example,
                mode=mode,
                include_contexts=include_contexts,
                device=device,
            )
            n_scored += 1
            n_correct += scores.correct
            n_correct_norm += scores.correct_norm

            # Capability retention is the proposal's headline metric, and MedMCQA's subject
            # labels are the only per-capability signal any of these datasets ships.
            subject = example.labels.get("subject_name")
            if subject:
                subject_totals[subject][0] += scores.correct
                subject_totals[subject][1] += 1

    if n_scored == 0:
        raise ValueError(
            f"no labelled examples in {source}:{split} ({n_skipped} unlabeled). "
            f"MedMCQA's test split withholds gold answers - evaluate on validation instead."
        )

    by_subject = {
        subject: {"accuracy": round(hits / count, 4), "n": count}
        for subject, (hits, count) in sorted(subject_totals.items())
    }
    return EvalReport(
        source=source,
        split=split,
        model=model_name,
        mode=mode,
        n_scored=n_scored,
        n_skipped_unlabeled=n_skipped,
        accuracy=round(n_correct / n_scored, 4),
        accuracy_norm=round(n_correct_norm / n_scored, 4),
        accuracy_ci=accuracy_ci(n_correct, n_scored),
        accuracy_norm_ci=accuracy_ci(n_correct_norm, n_scored),
        by_subject=by_subject,
        subject_spread=subject_spread(by_subject, min_n=SUBJECT_MIN_N),
    )
