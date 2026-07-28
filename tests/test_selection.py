import pytest

from medsel.schema import CorpusDoc, QAExample
from medsel.selection import (
    Scorer,
    available_scorers,
    get_scorer,
    resolve_budget,
    select_stratified,
    select_top_k,
)
from medsel.selection.base import register_scorer
from medsel.selection.scorers.baseline import LengthScorer, RandomScorer


def docs(*texts: str) -> list[CorpusDoc]:
    return [
        CorpusDoc(uid=f"pubmed/{i}", source="pubmed", text=t, title="") for i, t in enumerate(texts)
    ]


class TestScorerRegistry:
    def test_baselines_are_discoverable(self):
        assert {"random", "length"} <= set(available_scorers())

    def test_lookup_returns_a_configured_instance(self):
        assert isinstance(get_scorer("random", seed=7), RandomScorer)
        assert get_scorer("random", seed=7).seed == 7

    def test_unknown_scorer_lists_alternatives(self):
        with pytest.raises(KeyError, match="random"):
            get_scorer("less")

    def test_duplicate_registration_is_rejected(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_scorer("random")
            class Clashing(Scorer):
                def score(self, records):
                    return []

    def test_wrong_score_count_is_caught(self):
        class Broken(Scorer):
            def score(self, records):
                return [1.0]

        with pytest.raises(ValueError, match="returned 1 scores for 3 records"):
            Broken()(docs("a", "b", "c"))


class TestRandomScorer:
    def test_is_reproducible_for_a_given_seed(self):
        records = docs(*"abcde")
        assert RandomScorer(seed=1).score(records) == RandomScorer(seed=1).score(records)

    def test_different_seeds_differ(self):
        records = docs(*"abcde")
        assert RandomScorer(seed=1).score(records) != RandomScorer(seed=2).score(records)

    def test_does_not_depend_on_global_random_state(self):
        import random

        records = docs(*"abcde")
        random.seed(0)
        first = RandomScorer(seed=5).score(records)
        random.seed(999)
        assert RandomScorer(seed=5).score(records) == first

    def test_produces_one_score_per_record(self):
        assert len(RandomScorer().score(docs(*"abcdefg"))) == 7


class TestLengthScorer:
    def test_longer_text_scores_higher(self):
        scores = LengthScorer().score(docs("short", "much longer text here"))
        assert scores[1] > scores[0]

    def test_word_unit_counts_words(self):
        assert LengthScorer(unit="words").score(docs("one two three")) == [3.0]

    def test_title_counts_toward_length(self):
        titled = [CorpusDoc(uid="p/1", source="p", title="A title", text="body")]
        assert LengthScorer().score(titled)[0] > len("body")

    def test_scores_qa_examples_by_question(self):
        example = QAExample(
            uid="medmcqa/train/000000",
            source="medmcqa",
            split="train",
            question="A longer clinical vignette here",
            options={"A": "x", "B": "y"},
            answer_key="A",
        )
        assert LengthScorer().score([example])[0] == float(len(example.question))

    def test_rejects_unknown_unit(self):
        with pytest.raises(ValueError, match="chars"):
            LengthScorer(unit="tokens")


class TestResolveBudget:
    @pytest.mark.parametrize(
        ("budget", "n", "expected"),
        [(3, 10, 3), (20, 10, 10), (0.5, 10, 5), (1.0, 10, 10), (0.0, 10, 0), (0.25, 8, 2)],
    )
    def test_counts_and_fractions(self, budget, n, expected):
        assert resolve_budget(budget, n) == expected

    def test_int_one_is_a_count_and_float_one_is_everything(self):
        assert resolve_budget(1, 10) == 1
        assert resolve_budget(1.0, 10) == 10

    @pytest.mark.parametrize("budget", [1.5, -0.1])
    def test_rejects_out_of_range_fraction(self, budget):
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            resolve_budget(budget, 10)

    def test_rejects_negative_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            resolve_budget(-1, 10)


class TestSelectTopK:
    def test_returns_highest_scores_first(self):
        assert select_top_k([0.1, 0.9, 0.5], 2) == [1, 2]

    def test_ties_break_by_original_index(self):
        assert select_top_k([1.0, 1.0, 1.0], 2) == [0, 1]

    def test_fractional_budget(self):
        assert select_top_k([0.1, 0.2, 0.3, 0.4], 0.5) == [3, 2]

    def test_budget_beyond_pool_returns_everything(self):
        assert sorted(select_top_k([0.1, 0.2], 99)) == [0, 1]

    def test_zero_budget_selects_nothing(self):
        assert select_top_k([0.1, 0.2], 0) == []

    def test_empty_pool(self):
        assert select_top_k([], 5) == []


class TestSelectStratified:
    def test_respects_the_total_budget(self):
        scores = [float(i) for i in range(10)]
        groups = ["a"] * 5 + ["b"] * 5
        assert len(select_stratified(scores, groups, 4)) == 4

    def test_allocates_proportionally_to_group_size(self):
        scores = [1.0] * 12
        groups = ["big"] * 9 + ["small"] * 3
        chosen = set(select_stratified(scores, groups, 4))
        assert sum(1 for i in chosen if groups[i] == "big") == 3
        assert sum(1 for i in chosen if groups[i] == "small") == 1

    def test_picks_the_best_within_each_group(self):
        scores = [1.0, 5.0, 2.0, 9.0]
        groups = ["a", "a", "b", "b"]
        assert select_stratified(scores, groups, 2) == [1, 3]

    def test_min_per_group_protects_rare_groups(self):
        # Global top-k would take only 'common'; the floor keeps the rare group represented.
        scores = [10.0] * 20 + [0.1]
        groups = ["common"] * 20 + ["rare"]
        chosen = select_stratified(scores, groups, 5, min_per_group=1)
        assert 20 in chosen

    def test_without_the_floor_a_rare_group_can_vanish(self):
        scores = [10.0] * 20 + [0.1]
        groups = ["common"] * 20 + ["rare"]
        assert 20 not in select_stratified(scores, groups, 5)

    def test_returns_ascending_indices(self):
        scores = [float(i) for i in range(8)]
        groups = ["a", "b"] * 4
        chosen = select_stratified(scores, groups, 4)
        assert chosen == sorted(chosen)

    def test_never_exceeds_group_membership(self):
        scores = [1.0] * 4
        groups = ["a", "a", "b", "b"]
        chosen = select_stratified(scores, groups, 4)
        assert len(chosen) == 4 and len(set(chosen)) == 4

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="group labels"):
            select_stratified([1.0, 2.0], ["a"], 1)

    def test_zero_budget_selects_nothing(self):
        assert select_stratified([1.0, 2.0], ["a", "b"], 0) == []
