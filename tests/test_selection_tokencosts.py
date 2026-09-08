"""Counting the tokens a record will cost to train on."""

import pytest

from medsel.schema import CorpusDoc
from medsel.selection.costs import BUDGET_UNITS, token_costs


class FakeTokenizer:
    """Counts whitespace words, which is enough to test batching and ordering."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [text.split() for text in texts]}


def docs(texts):
    return [CorpusDoc(uid=f"d{i}", source="test", text=t) for i, t in enumerate(texts)]


class TestTokenCosts:
    TEXTS = ["one", "one two", "one two three", "one two three four"]

    def test_one_cost_per_record_in_order(self):
        assert token_costs(docs(self.TEXTS), FakeTokenizer(), progress=False) == [1, 2, 3, 4]

    def test_batching_does_not_change_the_result(self):
        records = docs(self.TEXTS)
        whole = token_costs(records, FakeTokenizer(), progress=False)
        batched = token_costs(records, FakeTokenizer(), batch_size=1, progress=False)
        assert whole == batched

    def test_empty_input(self):
        assert token_costs([], FakeTokenizer(), progress=False) == []

    def test_floors_a_zero_token_document_at_one(self):
        """The schema forbids empty text, but a tokenizer can still return no ids for it.

        A zero cost would fit any budget an unlimited number of times, which the selectors reject
        outright, so the floor is what keeps an odd document from breaking a selection.
        """

        class DropsEverything:
            def __call__(self, texts, add_special_tokens=False):
                return {"input_ids": [[] for _ in texts]}

        assert token_costs(docs(["real text"]), DropsEverything(), progress=False) == [1]

    def test_rejects_a_nonpositive_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            token_costs(docs(self.TEXTS), FakeTokenizer(), batch_size=0, progress=False)


class TestCustomRender:
    def test_render_overrides_default_text(self):
        records = docs(["ignored"])
        costs = token_costs(
            records, FakeTokenizer(), render=lambda _: "a b c d e", progress=False
        )
        assert costs == [5]

    def test_render_none_uses_default(self):
        records = docs(["one two"])
        assert token_costs(records, FakeTokenizer(), render=None, progress=False) == [2]


class TestBudgetUnits:
    def test_records_and_tokens_are_the_options(self):
        assert set(BUDGET_UNITS) == {"records", "tokens"}
