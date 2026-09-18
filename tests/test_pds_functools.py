"""Tests for PDS torch.func utilities (HVP, per-example JVP scores)."""

import torch
import torch.nn as nn

from medsel.selection.scorers.pds_functools import (
    compute_grad_and_loss,
    flatten_params,
    hvp,
    make_functional_loss,
    per_example_jvp_scores,
    unflatten_params,
)

VOCAB = 6
HIDDEN = 4


class TinyLM(nn.Module):
    """Minimal language model: embedding + linear head. No attention, no bias."""

    def __init__(self, vocab_size: int = VOCAB, hidden: int = HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        return self.head(h)


def _tiny_model():
    torch.manual_seed(42)
    m = TinyLM()
    m.eval()
    return m


def _loss_inputs(batch_size=1, seq_len=4):
    """Synthetic input_ids, labels, attention_mask for the tiny model."""
    torch.manual_seed(99)
    input_ids = torch.randint(0, VOCAB, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :2] = -100
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, labels, attention_mask


class TestFlattenUnflatten:
    def test_roundtrip(self):
        model = _tiny_model()
        params = dict(model.named_parameters())
        flat = flatten_params(params)
        recovered = unflatten_params(flat, params)
        for name in params:
            assert torch.allclose(params[name], recovered[name])

    def test_flatten_order_is_stable(self):
        model = _tiny_model()
        params = dict(model.named_parameters())
        a = flatten_params(params)
        b = flatten_params(params)
        assert torch.equal(a, b)

    def test_total_elements(self):
        model = _tiny_model()
        params = dict(model.named_parameters())
        flat = flatten_params(params)
        total = sum(p.numel() for p in params.values())
        assert flat.shape == (total,)


class TestMakeFunctionalLoss:
    def test_returns_scalar(self):
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs()
        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach() for n, p in model.named_parameters()}
        loss = loss_fn(params)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_same_params_same_loss(self):
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs()
        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach() for n, p in model.named_parameters()}
        a = loss_fn(params)
        b = loss_fn(params)
        assert torch.allclose(a, b)


class TestComputeGradAndLoss:
    def test_grad_shapes_match_params(self):
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs()
        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach() for n, p in model.named_parameters()}
        grads, loss = compute_grad_and_loss(loss_fn, params)
        for name in params:
            assert grads[name].shape == params[name].shape

    def test_loss_is_scalar(self):
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs()
        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach() for n, p in model.named_parameters()}
        _, loss = compute_grad_and_loss(loss_fn, params)
        assert loss.shape == ()


class TestHVP:
    def test_hvp_matches_explicit_hessian(self):
        """For a tiny model, verify HVP against the explicit Hessian."""
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs(batch_size=2, seq_len=4)

        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach().requires_grad_(True) for n, p in model.named_parameters()}

        torch.manual_seed(7)
        v = {n: torch.randn_like(p) for n, p in params.items()}
        hvp_result = hvp(loss_fn, params, v)

        flat_params = flatten_params(params)
        flat_v = flatten_params(v)

        def flat_loss(fp):
            unflat = unflatten_params(fp, params)
            return loss_fn(unflat)

        hessian = torch.autograd.functional.hessian(flat_loss, flat_params)
        expected = hessian @ flat_v

        actual = flatten_params(hvp_result)
        assert torch.allclose(actual, expected, atol=1e-4), (
            f"max diff: {(actual - expected).abs().max().item()}"
        )

    def test_hvp_linearity_in_v(self):
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs(batch_size=2, seq_len=4)

        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach() for n, p in model.named_parameters()}

        v1 = {n: torch.randn_like(p) for n, p in params.items()}
        v2 = {n: torch.randn_like(p) for n, p in params.items()}
        a, b = 2.0, -0.5

        hv1 = hvp(loss_fn, params, v1)
        hv2 = hvp(loss_fn, params, v2)

        combo_v = {n: a * v1[n] + b * v2[n] for n in params}
        hv_combo = hvp(loss_fn, params, combo_v)

        expected = {n: a * hv1[n] + b * hv2[n] for n in params}
        for n in params:
            assert torch.allclose(hv_combo[n], expected[n], atol=1e-5)

    def test_hvp_zero_v_gives_zero(self):
        model = _tiny_model()
        input_ids, labels, attn_mask = _loss_inputs(batch_size=2, seq_len=4)

        loss_fn = make_functional_loss(model, input_ids, labels, attn_mask)
        params = {n: p.detach() for n, p in model.named_parameters()}

        v = {n: torch.zeros_like(p) for n, p in params.items()}
        result = hvp(loss_fn, params, v)
        for n in params:
            assert torch.allclose(result[n], torch.zeros_like(result[n]), atol=1e-7)


class TestPerExampleScores:
    def test_output_length_matches_batch_size(self):
        torch.manual_seed(0)
        model = _tiny_model()
        params = {n: p.detach() for n, p in model.named_parameters()}
        B = 3
        input_ids = torch.randint(0, 2, (B, 4))
        labels = input_ids.clone()
        labels[:, :2] = -100
        attn_mask = torch.ones(B, 4, dtype=torch.long)

        costate = {n: torch.randn_like(p) for n, p in params.items()}
        scores = per_example_jvp_scores(
            model, params, input_ids, labels, attn_mask, costate, chunk_size=1
        )
        assert scores.shape == (B,)

    def test_zero_costate_gives_zero_scores(self):
        torch.manual_seed(0)
        model = _tiny_model()
        params = {n: p.detach() for n, p in model.named_parameters()}
        B = 2
        input_ids = torch.randint(0, 2, (B, 4))
        labels = input_ids.clone()
        labels[:, :2] = -100
        attn_mask = torch.ones(B, 4, dtype=torch.long)

        costate = {n: torch.zeros_like(p) for n, p in params.items()}
        scores = per_example_jvp_scores(
            model, params, input_ids, labels, attn_mask, costate, chunk_size=1
        )
        assert torch.allclose(scores, torch.zeros(B), atol=1e-7)

    def test_chunk_size_does_not_change_result(self):
        torch.manual_seed(0)
        model = _tiny_model()
        params = {n: p.detach() for n, p in model.named_parameters()}
        B = 4
        input_ids = torch.randint(0, 2, (B, 4))
        labels = input_ids.clone()
        labels[:, :2] = -100
        attn_mask = torch.ones(B, 4, dtype=torch.long)

        costate = {n: torch.randn_like(p) for n, p in params.items()}
        scores_1 = per_example_jvp_scores(
            model, params, input_ids, labels, attn_mask, costate, chunk_size=1
        )
        scores_all = per_example_jvp_scores(
            model, params, input_ids, labels, attn_mask, costate, chunk_size=B
        )
        assert torch.allclose(scores_1, scores_all, atol=1e-6)

    def test_scores_are_finite(self):
        torch.manual_seed(0)
        model = _tiny_model()
        params = {n: p.detach() for n, p in model.named_parameters()}
        B = 3
        input_ids = torch.randint(0, 2, (B, 4))
        labels = input_ids.clone()
        labels[:, :2] = -100
        attn_mask = torch.ones(B, 4, dtype=torch.long)

        costate = {n: torch.randn_like(p) for n, p in params.items()}
        scores = per_example_jvp_scores(
            model, params, input_ids, labels, attn_mask, costate, chunk_size=1
        )
        assert torch.isfinite(scores).all()
