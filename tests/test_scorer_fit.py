"""Separating pool fitting from per-record scoring, which streaming depends on."""

import pytest

from medsel.schema import CorpusDoc
from medsel.selection.base import Scorer, get_scorer
from medsel.selection.scorers.perplexity import rank_by_mode

MEDICAL = [
    "Patients with acute myocardial infarction received reperfusion therapy within six hours.",
    "The randomised trial measured serum creatinine and glomerular filtration rate at baseline.",
    "Antibiotic resistance in gram negative bacteria complicated the postoperative course.",
]
UNRELATED = [
    "The midfielder scored twice in the second half of the cup final on Saturday.",
    "Preheat the oven and whisk the butter and sugar until the mixture turns pale.",
    "Trains to the coast were delayed after signalling faults near the junction.",
]


def docs(texts, offset=0):
    return [CorpusDoc(uid=f"d{i + offset}", source="test", text=t) for i, t in enumerate(texts)]


class TestScorerFitHook:
    def test_default_fit_is_a_no_op(self):
        class Trivial(Scorer):
            def score(self, records):
                return [0.0] * len(records)

        Trivial().fit(docs(MEDICAL))  # must not raise

    def test_baseline_scorers_accept_fit(self):
        get_scorer("random").fit(docs(MEDICAL))
        get_scorer("length").fit(docs(MEDICAL))


class TestDSIRFit:
    def test_scoring_without_fit_still_works(self):
        scorer = get_scorer("dsir", target=MEDICAL, gumbel=False)
        assert len(scorer(docs(MEDICAL + UNRELATED))) == 6

    def test_fit_state_is_reused_across_chunks(self):
        """Two chunks scored after one fit must match one call over the concatenation."""
        pool = docs(MEDICAL + UNRELATED)

        whole = get_scorer("dsir", target=MEDICAL, gumbel=False)
        whole.fit(pool)
        expected = whole.score(pool)

        chunked = get_scorer("dsir", target=MEDICAL, gumbel=False)
        chunked.fit(pool)
        actual = chunked.score(pool[:3]) + chunked.score(pool[3:])
        assert actual == pytest.approx(expected)

    def test_fitting_on_a_sample_does_not_refit_per_chunk(self):
        scorer = get_scorer("dsir", target=MEDICAL, gumbel=False)
        scorer.fit(docs(MEDICAL + UNRELATED))
        fitted = scorer._pool_model
        scorer.score(docs(UNRELATED))
        assert scorer._pool_model is fitted

    def test_noise_does_not_repeat_across_successive_chunks(self):
        """A fresh generator per chunk would replay the same noise on every chunk."""
        scorer = get_scorer("dsir", target=MEDICAL, gumbel=True, seed=3)
        scorer.fit(docs(MEDICAL))
        first = scorer.score(docs(MEDICAL))
        second = scorer.score(docs(MEDICAL))
        assert first != second


class TestPerplexityMedian:
    def test_rank_by_mode_defaults_to_the_median_of_its_input(self):
        assert rank_by_mode([1.0, 2.0, 3.0], "mid") == pytest.approx([-1.0, 0.0, -1.0])

    def test_rank_by_mode_accepts_an_external_median(self):
        # Streaming supplies a median fitted over the pool, not over this chunk.
        assert rank_by_mode([1.0, 2.0, 3.0], "mid", median=3.0) == pytest.approx([-2.0, -1.0, 0.0])

    def test_external_median_is_ignored_by_the_other_modes(self):
        assert rank_by_mode([1.0, 2.0], "low", median=99.0) == pytest.approx([-1.0, -2.0])
