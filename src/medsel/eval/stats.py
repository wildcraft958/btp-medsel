"""Uncertainty and spread on an accuracy, so a difference can be told from noise.

Measured on this project's own base control, a 95% interval is 5.1 points wide on MedQA (1,273
items), 2.9 on MedMCQA (4,183) and 8.6 on PubMedQA (500). Two runs differing by a point and a half
have therefore not been shown to differ at all, and on PubMedQA almost no plausible CPT delta will
clear its own interval. A CPT result is a delta against a control, and reporting one without its
interval invites a conclusion the data does not support.

The second function here answers a different question. Aggregate accuracy can hold steady while a
model trades one clinical capability for another, and a single number cannot show that. Spread
across MedMCQA's subject labels is the only per-capability signal these datasets ship, and it is
the closest thing available to a direct measurement of the proposal's retention claim.
"""

from __future__ import annotations

from typing import Any

__all__ = ["accuracy_ci", "subject_spread"]


def accuracy_ci(
    n_correct: int,
    n_scored: int,
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval on an accuracy, or ``None`` when nothing was scored.

    Drawn from a binomial rather than by resampling an index vector. For a mean of zero-one values
    the two are the same distribution: resampling ``n`` items with replacement from a set where
    ``k`` are correct gives a count of correct answers that is exactly ``Binomial(n, k/n)``. Taking
    that route keeps the interval exact while making it fast enough to run on every evaluation.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0.0, 1.0), got {confidence}")
    if n_scored < 0 or n_correct < 0:
        raise ValueError(f"counts must be non-negative, got {n_correct} of {n_scored}")
    if n_correct > n_scored:
        raise ValueError(f"{n_correct} correct out of {n_scored} scored is not possible")
    if n_scored == 0:
        return None

    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.binomial(n_scored, n_correct / n_scored, size=resamples) / n_scored
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return round(float(low), 4), round(float(high), 4)


def subject_spread(by_subject: dict[str, dict[str, Any]], *, min_n: int = 1) -> dict[str, Any]:
    """How unevenly accuracy is distributed across subjects.

    ``min_n`` drops subjects with too few items to carry a meaningful accuracy. A subject with two
    questions scores 0.0, 0.5 or 1.0 and nothing else, so including it inflates the variance with
    an artefact of sample size rather than a real capability gap.
    """
    scores = {
        subject: float(entry["accuracy"])
        for subject, entry in by_subject.items()
        if int(entry.get("n", 0)) >= min_n
    }
    if not scores:
        return {}

    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    worst = min(scores, key=lambda subject: (scores[subject], subject))
    best = max(scores, key=lambda subject: (scores[subject], subject))

    return {
        "n_subjects": len(values),
        "mean": round(mean, 4),
        "variance": round(variance, 6),
        "std": round(variance**0.5, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "range": round(max(values) - min(values), 4),
        "worst": worst,
        "best": best,
    }
