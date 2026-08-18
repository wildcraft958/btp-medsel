import pytest

from medsel.selection.base import available_scorers, get_scorer
from medsel.selection.scorers.perplexity import rank_by_mode


class TestRankByMode:
    """The mode transform is separated from the model so it can be checked without one."""

    NLL = [1.0, 2.0, 3.0, 10.0]

    def test_low_prefers_fluent_text(self):
        scores = rank_by_mode(self.NLL, "low")
        assert scores.index(max(scores)) == 0

    def test_high_prefers_surprising_text(self):
        scores = rank_by_mode(self.NLL, "high")
        assert scores.index(max(scores)) == 3

    def test_mid_prefers_the_middle_band(self):
        scores = rank_by_mode(self.NLL, "mid")
        # Median of the four values is 2.5, so the two central documents win and the outlier loses.
        assert scores.index(max(scores)) in {1, 2}
        assert scores.index(min(scores)) == 3

    def test_returns_one_score_per_input(self):
        assert len(rank_by_mode(self.NLL, "mid")) == len(self.NLL)

    def test_empty_input(self):
        assert rank_by_mode([], "mid") == []


class TestPerplexityScorer:
    def test_registered_under_its_name(self):
        assert "perplexity" in available_scorers()

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            get_scorer("perplexity", mode="sideways")

    @pytest.mark.parametrize("mode", ["low", "mid", "high"])
    def test_accepts_the_documented_modes(self, mode):
        assert get_scorer("perplexity", mode=mode).mode == mode

    def test_rejects_a_nonpositive_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            get_scorer("perplexity", batch_size=0)

    def test_empty_pool_needs_no_model(self):
        # Must not attempt to download a model just to score nothing.
        assert get_scorer("perplexity", model="does-not-exist/nope")([]) == []
