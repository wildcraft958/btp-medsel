"""Functional PyTorch utilities for PDS (PMP-based Data Selection).

Wraps torch.func operations (grad, jvp, vmap, functional_call) into reusable
building blocks for the PMP solver. All functions are pure: no side effects
on the model object.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "flatten_params",
    "unflatten_params",
    "make_functional_loss",
    "compute_grad_and_loss",
    "hvp",
    "per_example_jvp_scores",
]


def flatten_params(params: dict[str, Tensor]) -> Tensor:
    """Flatten a parameter dict into a single 1D tensor (sorted by key)."""
    return torch.cat([params[k].detach().reshape(-1) for k in sorted(params)])


def unflatten_params(flat: Tensor, reference: dict[str, Tensor]) -> dict[str, Tensor]:
    """Reshape a flat vector back into a parameter dict matching reference shapes."""
    result: dict[str, Tensor] = {}
    offset = 0
    for k in sorted(reference):
        numel = reference[k].numel()
        result[k] = flat[offset : offset + numel].reshape(reference[k].shape)
        offset += numel
    return result


def _forward_call(
    model: nn.Module,
    params: dict[str, Tensor],
    input_ids: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    """Run a functional forward pass, returning logits.

    Handles both HuggingFace models (input_ids/attention_mask kwargs returning
    an object with .logits) and plain nn.Module (positional input, raw tensor output).
    """
    try:
        out = torch.func.functional_call(
            model,
            params,
            args=(),
            kwargs={"input_ids": input_ids, "attention_mask": attention_mask},
        )
    except TypeError:
        out = torch.func.functional_call(model, params, args=(input_ids,))
    return out.logits if hasattr(out, "logits") else out


def make_functional_loss(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    attention_mask: Tensor | None,
) -> Callable[[dict[str, Tensor]], Tensor]:
    """Return a closure that computes cross-entropy loss as a function of parameters only.

    The returned function is suitable for torch.func.grad and torch.func.jvp.
    Labels with value -100 are ignored (completion-only masking).
    """

    def loss_fn(params: dict[str, Tensor]) -> Tensor:
        logits = _forward_call(model, params, input_ids, attention_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    return loss_fn


def compute_grad_and_loss(
    loss_fn: Callable[[dict[str, Tensor]], Tensor],
    params: dict[str, Tensor],
) -> tuple[dict[str, Tensor], Tensor]:
    """Gradient of loss w.r.t. params via torch.func.grad_and_value."""
    grad_fn = torch.func.grad_and_value(loss_fn)
    grads, loss_val = grad_fn(params)
    return grads, loss_val


def hvp(
    loss_fn: Callable[[dict[str, Tensor]], Tensor],
    params: dict[str, Tensor],
    v: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Hessian-vector product H @ v via forward-over-reverse.

    Computes jvp(grad(loss_fn), params, v) which equals d^2L/dtheta^2 @ v.
    O(params) memory, not O(params^2).
    """
    grad_fn = torch.func.grad(loss_fn)
    _, hvp_result = torch.func.jvp(grad_fn, (params,), (v,))
    return hvp_result


def _single_example_loss(
    model: Any,
    params: dict[str, Tensor],
    input_ids_single: Tensor,
    labels_single: Tensor,
    attention_mask_single: Tensor,
) -> Tensor:
    """Loss for a single (unbatched) example. Used inside vmap."""
    ids = input_ids_single.unsqueeze(0)
    mask = attention_mask_single.unsqueeze(0)
    lbl = labels_single.unsqueeze(0)
    logits = _forward_call(model, params, ids, mask)
    shift_logits = logits[0, :-1, :]
    shift_labels = lbl[0, 1:]
    loss = nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100
    )
    return loss


def per_example_jvp_scores(
    model: nn.Module,
    params: dict[str, Tensor],
    batch_input_ids: Tensor,
    batch_labels: Tensor,
    batch_attention_mask: Tensor,
    costate: dict[str, Tensor],
    chunk_size: int = 4,
) -> Tensor:
    """Per-example <grad(L_i), lambda> via vmap(jvp(...)).

    Returns a (B,) tensor of per-example contribution scores.
    chunk_size controls memory by processing examples in chunks.
    """
    B = batch_input_ids.shape[0]

    def score_single(ids: Tensor, lbl: Tensor, mask: Tensor) -> Tensor:
        def single_loss(p: dict[str, Tensor]) -> Tensor:
            return _single_example_loss(model, p, ids, lbl, mask)

        _, ct = torch.func.jvp(single_loss, (params,), (costate,))
        return ct

    scores = torch.func.vmap(
        score_single,
        in_dims=(0, 0, 0),
        chunk_size=chunk_size if chunk_size < B else None,
    )(batch_input_ids, batch_labels, batch_attention_mask)

    return scores
