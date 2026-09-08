"""Held-out perplexity, the sensitive instrument for whether CPT data mattered."""

import math

import pytest

from medsel.eval.perplexity import PerplexityReport, mean_token_nll


class FakeModel:
    """Returns fixed logits so the arithmetic can be checked without a real model."""

    def __init__(self, vocab=4):
        self.vocab = vocab

    def eval(self):
        return self

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        import torch

        class Out:
            pass

        out = Out()
        # Uniform over the vocabulary, so every token's NLL is exactly log(vocab).
        out.logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab)
        return out


class FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(self, texts, **kw):
        import torch

        ids = [[1, 2, 3] for _ in texts]
        return _Batch(
            {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.ones(len(ids), 3, dtype=torch.long),
            }
        )


class _Batch(dict):
    def to(self, device):
        return self


class TestMeanTokenNLL:
    def test_uniform_logits_give_log_vocab(self):
        nll = mean_token_nll(FakeModel(vocab=4), FakeTokenizer(), ["a", "b"], max_length=8)
        assert nll == pytest.approx(math.log(4), rel=1e-4)

    def test_empty_input_is_nan_free(self):
        assert mean_token_nll(FakeModel(), FakeTokenizer(), [], max_length=8) is None


class TestPerplexityReport:
    def test_perplexity_is_the_exponential_of_the_loss(self):
        report = PerplexityReport(name="held_out", n_documents=10, mean_token_nll=math.log(7))
        assert report.perplexity == pytest.approx(7.0)

    def test_serialises_both_numbers(self):
        payload = PerplexityReport(name="held_out", n_documents=3, mean_token_nll=1.5).to_dict()
        assert payload["mean_token_nll"] == 1.5
        # Reported to four decimals, since a perplexity carries no meaning past that.
        assert payload["perplexity"] == pytest.approx(math.exp(1.5), abs=5e-5)
        assert payload["n_documents"] == 3
