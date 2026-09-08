"""Selecting under a cost budget, where a record's cost is its token count."""

import pytest

from medsel.selection.selector import resolve_budget, select_stratified, select_top_k

# Descending score, so the greedy order is simply index order.
SCORES = [9.0, 8.0, 7.0, 6.0, 5.0]
COSTS = [100, 400, 100, 100, 100]


class TestResolveBudgetOnTotals:
    def test_fraction_of_a_cost_total(self):
        assert resolve_budget(0.25, 800) == 200

    def test_count_is_capped_by_the_total(self):
        assert resolve_budget(1000, 800) == 800


class TestTopKUnderCosts:
    def test_record_budget_is_unchanged_without_costs(self):
        assert select_top_k(SCORES, 2) == [0, 1]

    def test_stops_at_the_cost_budget(self):
        # 100 + 400 = 500 exactly.
        assert select_top_k(SCORES, 500, costs=COSTS) == [0, 1]

    def test_skips_an_item_that_does_not_fit_and_keeps_going(self):
        # 100 fits, 400 does not, so the next three 100s are taken instead.
        assert select_top_k(SCORES, 400, costs=COSTS) == [0, 2, 3, 4]

    def test_fractional_budget_is_a_fraction_of_total_cost(self):
        # Total cost 800, so 0.5 is 400.
        assert select_top_k(SCORES, 0.5, costs=COSTS) == [0, 2, 3, 4]

    def test_zero_budget_selects_nothing(self):
        assert select_top_k(SCORES, 0, costs=COSTS) == []

    def test_budget_above_total_cost_takes_everything(self):
        assert select_top_k(SCORES, 10_000, costs=COSTS) == [0, 1, 2, 3, 4]

    def test_rejects_mismatched_cost_length(self):
        with pytest.raises(ValueError, match="costs"):
            select_top_k(SCORES, 100, costs=[1, 2])

    def test_rejects_a_nonpositive_cost(self):
        with pytest.raises(ValueError, match="positive"):
            select_top_k(SCORES, 100, costs=[1, 0, 1, 1, 1])

    def test_ties_break_by_index(self):
        assert select_top_k([1.0, 1.0, 1.0], 2, costs=[1, 1, 1]) == [0, 1]


class TestStratifiedUnderCosts:
    GROUPS = ["a", "a", "a", "b", "b"]

    def test_allocates_cost_in_proportion_to_group_cost(self):
        # Group a costs 600, group b costs 200, total 800. At a 400 budget, a gets 300 and b 100.
        chosen = select_stratified(SCORES, self.GROUPS, 400, costs=COSTS)
        assert set(self.GROUPS[i] for i in chosen) == {"a", "b"}
        assert sum(COSTS[i] for i in chosen) <= 400

    def test_never_exceeds_the_budget(self):
        for budget in (100, 250, 400, 700):
            chosen = select_stratified(SCORES, self.GROUPS, budget, costs=COSTS)
            assert sum(COSTS[i] for i in chosen) <= budget

    def test_min_per_group_is_still_a_record_floor(self):
        chosen = select_stratified(SCORES, self.GROUPS, 700, min_per_group=1, costs=COSTS)
        kept = [self.GROUPS[i] for i in chosen]
        assert kept.count("a") >= 1 and kept.count("b") >= 1

    def test_returns_indices_in_ascending_order(self):
        chosen = select_stratified(SCORES, self.GROUPS, 700, costs=COSTS)
        assert chosen == sorted(chosen)

    def test_raises_when_the_floor_cannot_be_afforded(self):
        with pytest.raises(ValueError, match="min_per_group"):
            select_stratified(SCORES, self.GROUPS, 50, min_per_group=1, costs=COSTS)
