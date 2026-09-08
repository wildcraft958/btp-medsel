"""Tests for the 3DS (Decomposed Difficulty Data Selection) scorer."""

import math

import numpy as np
import pytest

from medsel.schema import QAExample
from medsel.selection.base import get_scorer
from medsel.selection.scorers.tds import (
    ThreeDSScorer,
    _percentile_ranks,
    _quality_pass,
)
from medsel.selection.selector import select_top_k


def _example(
    uid: str = "test/train/000001",
    question: str = "What is the mechanism of action of aspirin?",
    options: dict[str, str] | None = None,
    answer_key: str = "A",
    rationale: str | None = None,
) -> QAExample:
    opts = options or {
        "A": "Inhibition of cyclooxygenase",
        "B": "Activation of prostaglandins",
        "C": "Inhibition of lipoxygenase",
        "D": "Activation of thromboxane",
    }
    return QAExample(
        uid=uid,
        source="test",
        split="train",
        question=question,
        options=opts,
        answer_key=answer_key,
        rationale=rationale,
    )


class TestQualityPass:
    def test_accepts_well_formed_example(self):
        assert _quality_pass(_example()) is True

    def test_rejects_short_question(self):
        assert _quality_pass(_example(question="Short?")) is False

    def test_rejects_duplicate_options(self):
        opts = {"A": "Same", "B": "Same", "C": "Same", "D": "Same"}
        assert _quality_pass(_example(options=opts)) is False

    def test_rejects_empty_option(self):
        opts = {"A": "Real answer", "B": "", "C": "Other", "D": "Another"}
        assert _quality_pass(_example(options=opts)) is False

    def test_rejects_non_qa_record(self):
        from medsel.schema import CorpusDoc

        doc = CorpusDoc(uid="d0", source="test", text="some text")
        assert _quality_pass(doc) is False

    def test_rejects_example_without_options(self):
        ex = QAExample(
            uid="test/train/000001",
            source="test",
            split="train",
            question="A valid question that is long enough?",
        )
        assert _quality_pass(ex) is False

    def test_two_distinct_options_pass(self):
        opts = {"A": "Yes", "B": "No", "C": "Yes", "D": "No"}
        assert _quality_pass(_example(options=opts)) is True


class TestPercentileRanks:
    def test_monotone_input(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ranks = _percentile_ranks(values)
        assert ranks[0] == 0.0
        assert ranks[-1] == 100.0
        assert all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1))

    def test_single_element(self):
        ranks = _percentile_ranks(np.array([42.0]))
        assert ranks[0] == 0.0

    def test_ties_get_consistent_ranks(self):
        values = np.array([1.0, 1.0, 3.0])
        ranks = _percentile_ranks(values)
        assert len(ranks) == 3
        assert ranks[2] == 100.0

    def test_empty(self):
        ranks = _percentile_ranks(np.array([]))
        assert len(ranks) == 0


class TestThreeDSScorerInterface:
    def test_registered_under_3ds(self):
        scorer = get_scorer("3ds", judge="ignored")
        assert scorer.name == "3ds"

    def test_empty_pool_returns_empty(self):
        scorer = ThreeDSScorer(judge="ignored")
        assert scorer.score([]) == []

    def test_rejects_invalid_percentile_range(self):
        with pytest.raises(ValueError, match="low_th"):
            ThreeDSScorer(judge="x", low_th=80.0, up_th=20.0)

    def test_rejects_invalid_atten_method(self):
        with pytest.raises(ValueError, match="atten_method"):
            ThreeDSScorer(judge="x", atten_method="invalid")

    def test_repr(self):
        s = ThreeDSScorer(judge="model/name", low_th=20, up_th=80, atten_method="max")
        r = repr(s)
        assert "model/name" in r
        assert "max" in r


class TestGoldilocksScoring:
    """Verify the percentile-based Goldilocks logic using synthetic metrics."""

    @staticmethod
    def _make_scores_from_metrics(
        d1_vals: list[float],
        d2_vals: list[float],
        d3_vals: list[float],
        low_th: float = 25.0,
        up_th: float = 75.0,
    ) -> list[float]:
        """Reproduce the scorer's Goldilocks logic on pre-computed metrics."""
        n = len(d1_vals)
        d1 = np.array(d1_vals)
        d2 = np.array(d2_vals)
        d3 = np.array(d3_vals)

        p1 = _percentile_ranks(d1)
        p2 = _percentile_ranks(d2)
        p3 = _percentile_ranks(d3)

        scores = []
        for i in range(n):
            if not (low_th <= p1[i] <= up_th
                    and low_th <= p2[i] <= up_th
                    and low_th <= p3[i] <= up_th):
                scores.append(float("-inf"))
            else:
                dist = abs(p1[i] - 50.0) + abs(p2[i] - 50.0) + abs(p3[i] - 50.0)
                scores.append(-dist)
        return scores

    def test_extreme_examples_get_negative_infinity(self):
        d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        scores = self._make_scores_from_metrics(d, d, d)
        assert scores[0] == float("-inf")
        assert scores[-1] == float("-inf")

    def test_middle_examples_are_finite(self):
        d = list(range(1, 21))
        scores = self._make_scores_from_metrics(d, d, d)
        finite = [s for s in scores if s != float("-inf")]
        assert len(finite) > 0

    def test_center_example_scores_highest(self):
        d = list(range(1, 11))
        scores = self._make_scores_from_metrics(d, d, d)
        finite_scores = [(s, i) for i, s in enumerate(scores) if s != float("-inf")]
        if finite_scores:
            best_score, best_idx = max(finite_scores)
            assert 2 <= best_idx <= 7

    def test_select_top_k_excludes_neg_inf(self):
        d = list(range(1, 11))
        scores = self._make_scores_from_metrics(d, d, d)
        chosen = select_top_k(scores, 3)
        for idx in chosen:
            assert scores[idx] != float("-inf")

    def test_all_outside_range_gives_all_neg_inf(self):
        d = [1.0, 2.0, 3.0]
        scores = self._make_scores_from_metrics(d, d, d, low_th=51.0, up_th=60.0)
        assert all(s == float("-inf") for s in scores)


class TestBuildAnswerText:
    def test_includes_rationale_for_gold_answer(self):
        ex = _example(rationale="Aspirin blocks COX enzymes.")
        scorer = ThreeDSScorer(judge="x", use_rationale=True)
        text = scorer._build_answer_text(ex, "A")
        assert "COX enzymes" in text
        assert "Explanation:" in text

    def test_no_rationale_for_non_gold(self):
        ex = _example(rationale="Aspirin blocks COX enzymes.")
        scorer = ThreeDSScorer(judge="x", use_rationale=True)
        text = scorer._build_answer_text(ex, "B")
        assert "Explanation:" not in text

    def test_rationale_disabled(self):
        ex = _example(rationale="Aspirin blocks COX enzymes.")
        scorer = ThreeDSScorer(judge="x", use_rationale=False)
        text = scorer._build_answer_text(ex, "A")
        assert "Explanation:" not in text

    def test_answer_text_contains_option_text(self):
        ex = _example()
        scorer = ThreeDSScorer(judge="x")
        text = scorer._build_answer_text(ex, "A")
        assert "cyclooxygenase" in text.lower()
