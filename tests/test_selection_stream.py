"""Streaming selection: bounded memory, identical results to the in-memory path."""

import pytest

from medsel.schema import QAExample
from medsel.selection.base import Scorer, get_scorer
from medsel.selection.selector import select_stratified, select_top_k
from medsel.selection.stream import TopK, select_stream

SUBJECTS = ["Anatomy", "Physiology", "Surgery"]


def examples(n):
    return [
        QAExample(
            uid=f"q{i}",
            source="medmcqa",
            split="train",
            question="word " * (i % 17 + 1),
            labels={"subject_name": SUBJECTS[i % len(SUBJECTS)]},
        )
        for i in range(n)
    ]


class RecordingScorer(Scorer):
    """Scores by question length and remembers the batch sizes it was handed."""

    def __init__(self):
        self.score_sizes = []
        self.fit_sizes = []

    def fit(self, records):
        self.fit_sizes.append(len(records))

    def score(self, records):
        self.score_sizes.append(len(records))
        return [float(len(r.question)) for r in records]


class TestTopK:
    def test_keeps_only_the_highest(self):
        heap = TopK(2)
        for i, score in enumerate([1.0, 5.0, 3.0, 2.0]):
            heap.offer(score, f"u{i}")
        assert [uid for _, uid in heap.ranked()] == ["u1", "u2"]

    def test_ranked_is_highest_first(self):
        heap = TopK(3)
        for i, score in enumerate([1.0, 3.0, 2.0]):
            heap.offer(score, f"u{i}")
        assert [score for score, _ in heap.ranked()] == [3.0, 2.0, 1.0]

    def test_ties_keep_the_earlier_record(self):
        heap = TopK(1)
        heap.offer(1.0, "first")
        heap.offer(1.0, "second")
        assert [uid for _, uid in heap.ranked()] == ["first"]

    def test_zero_capacity_keeps_nothing(self):
        heap = TopK(0)
        heap.offer(9.0, "u0")
        assert heap.ranked() == []

    def test_capacity_above_the_stream_keeps_everything(self):
        heap = TopK(10)
        for i in range(3):
            heap.offer(float(i), f"u{i}")
        assert len(heap.ranked()) == 3


class TestStreamMatchesInMemory:
    def test_top_k_selection_is_identical(self):
        pool = examples(200)
        scorer = get_scorer("length")
        expected = [pool[i].uid for i in select_top_k(scorer(pool), 20)]

        result = select_stream(iter(pool), get_scorer("length"), 20, chunk_size=7, fit_size=13)
        assert result.uids == expected

    def test_stratified_selection_is_identical(self):
        pool = examples(200)
        scorer = get_scorer("length")
        scores = scorer(pool)
        groups = [r.labels["subject_name"] for r in pool]
        expected = sorted(pool[i].uid for i in select_stratified(scores, groups, 30, 2))

        result = select_stream(
            iter(pool),
            get_scorer("length"),
            30,
            chunk_size=7,
            fit_size=13,
            stratify_by="subject_name",
            min_per_group=2,
        )
        assert sorted(result.uids) == expected

    def test_counts_the_whole_stream(self):
        result = select_stream(iter(examples(200)), get_scorer("length"), 20, chunk_size=7)
        assert result.n_candidates == 200

    def test_summary_covers_the_whole_pool_not_just_a_chunk(self):
        pool = examples(200)
        scores = get_scorer("length")(pool)
        result = select_stream(iter(pool), get_scorer("length"), 20, chunk_size=7)
        assert result.score_summary["min"] == pytest.approx(min(scores))
        assert result.score_summary["max"] == pytest.approx(max(scores))
        assert result.score_summary["mean"] == pytest.approx(sum(scores) / len(scores))


class TestBoundedMemory:
    def test_never_scores_more_than_a_chunk_at_once(self):
        scorer = RecordingScorer()
        select_stream(iter(examples(500)), scorer, 10, chunk_size=25, fit_size=50)
        assert max(scorer.score_sizes) <= 25

    def test_fits_once_on_a_bounded_sample(self):
        scorer = RecordingScorer()
        select_stream(iter(examples(500)), scorer, 10, chunk_size=25, fit_size=50)
        assert scorer.fit_sizes == [50]

    def test_fit_sample_is_the_whole_pool_when_it_is_smaller(self):
        scorer = RecordingScorer()
        select_stream(iter(examples(20)), scorer, 5, chunk_size=25, fit_size=50)
        assert scorer.fit_sizes == [20]

    def test_consumes_a_generator_lazily(self):
        consumed = []

        def stream():
            for record in examples(100):
                consumed.append(record.uid)
                yield record

        select_stream(stream(), get_scorer("length"), 5, chunk_size=10, fit_size=10)
        assert len(consumed) == 100


class TestBudget:
    def test_rejects_a_fractional_budget(self):
        with pytest.raises(ValueError, match="fraction"):
            select_stream(iter(examples(50)), get_scorer("length"), 0.1)

    def test_budget_larger_than_the_pool_keeps_everything(self):
        result = select_stream(iter(examples(10)), get_scorer("length"), 999)
        assert result.n_selected == 10

    def test_empty_stream(self):
        result = select_stream(iter([]), get_scorer("length"), 5)
        assert result.n_candidates == 0 and result.uids == [] and result.score_summary == {}
