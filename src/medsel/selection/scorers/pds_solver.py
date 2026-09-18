"""PMP Solver engine for PDS (PMP-based Data Selection).

Implements the three-phase PMP algorithm from Gu et al. (ICLR 2025):
  1. Forward pass: SGD training with periodic checkpointing
  2. Backward pass: reverse iteration computing per-example quality scores
     via the discretized adjoint equation
  3. Aggregation: accumulate per-example scores across backward steps

The costate update `lambda = g_dev + lambda - lr * hvp` optimizes the
area under the validation loss curve (integral over training steps),
not just the final validation loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm import tqdm

from medsel.selection.scorers.pds_functools import (
    compute_grad_and_loss,
    hvp,
    make_functional_loss,
    per_example_jvp_scores,
)

__all__ = ["PMPConfig", "CheckpointManager", "PMPSolver"]


@dataclass
class PMPConfig:
    """Configuration for the PMP solver."""

    pmp_steps: int = 100
    pmp_lr: float = 0.01
    checkpoint_interval: int = 0
    compute_ct_interval: int = 10
    chunk_size: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.checkpoint_interval <= 0:
            self.checkpoint_interval = max(1, int(math.sqrt(self.pmp_steps)))


class CheckpointManager:
    """Disk-based checkpoint management for PMP solver."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, step: int) -> Path:
        return self.cache_dir / f"step_{step}.pt"

    def save_checkpoint(
        self,
        step: int,
        params: dict[str, Tensor],
        batch_data: dict[str, Tensor],
    ) -> None:
        torch.save({"params": params, "batch": batch_data}, self._path(step))

    def load_checkpoint(
        self, step: int
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        data = torch.load(self._path(step), weights_only=True)
        return data["params"], data["batch"]

    def checkpoint_steps(self) -> list[int]:
        steps = []
        for p in self.cache_dir.glob("step_*.pt"):
            steps.append(int(p.stem.split("_")[1]))
        return sorted(steps)

    def is_complete(self, total_steps: int, interval: int) -> bool:
        expected = set(range(0, total_steps + 1, interval))
        return expected.issubset(set(self.checkpoint_steps()))


class PMPSolver:
    """PMP-based data quality scorer.

    Implements the forward-then-backward PMP algorithm to compute
    per-example quality scores for data selection.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PMPConfig,
        cache_dir: Path,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.ckpt_mgr = CheckpointManager(cache_dir / "pmp_checkpoints")

    def forward_pass(
        self,
        train_batches: list[dict[str, Tensor]],
        progress: bool = True,
    ) -> None:
        """Phase 1: SGD training with periodic checkpointing."""
        params = {
            n: p.detach().clone() for n, p in self.model.named_parameters()
        }
        cfg = self.config
        n_batches = len(train_batches)

        steps = range(cfg.pmp_steps)
        if progress:
            steps = tqdm(steps, desc="PMP forward", leave=False)

        for step in steps:
            batch = train_batches[step % n_batches]
            batch_dev = {k: v.to(self.device) for k, v in batch.items()}

            if step % cfg.checkpoint_interval == 0:
                self.ckpt_mgr.save_checkpoint(
                    step,
                    {n: p.cpu() for n, p in params.items()},
                    {k: v.cpu() for k, v in batch_dev.items()},
                )

            loss_fn = make_functional_loss(
                self.model,
                batch_dev["input_ids"],
                batch_dev["labels"],
                batch_dev.get("attention_mask"),
            )
            grads, _ = compute_grad_and_loss(loss_fn, params)
            params = {n: params[n] - cfg.pmp_lr * grads[n] for n in params}

        self.ckpt_mgr.save_checkpoint(
            cfg.pmp_steps,
            {n: p.cpu() for n, p in params.items()},
            {k: v.cpu() for k, v in train_batches[(cfg.pmp_steps - 1) % n_batches].items()},
        )

    def _reconstruct_segment_to_cpu(
        self,
        start_step: int,
        end_step: int,
        train_batches: list[dict[str, Tensor]],
    ) -> list[dict[str, Tensor]]:
        """Replay forward from a checkpoint, returning intermediate params on CPU."""
        params_cpu, _ = self.ckpt_mgr.load_checkpoint(start_step)
        params = {n: p.to(self.device) for n, p in params_cpu.items()}

        segment_params = [{n: p.cpu() for n, p in params.items()}]
        n_batches = len(train_batches)

        for step in range(start_step, end_step):
            batch = train_batches[step % n_batches]
            batch_dev = {k: v.to(self.device) for k, v in batch.items()}

            loss_fn = make_functional_loss(
                self.model,
                batch_dev["input_ids"],
                batch_dev["labels"],
                batch_dev.get("attention_mask"),
            )
            grads, _ = compute_grad_and_loss(loss_fn, params)
            params = {n: params[n] - self.config.pmp_lr * grads[n] for n in params}
            segment_params.append({n: p.cpu() for n, p in params.items()})

        del params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segment_params

    def _compute_val_grad(
        self,
        params: dict[str, Tensor],
        val_batches: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Average gradient of validation loss over all val batches."""
        total_grads: dict[str, Tensor] | None = None

        for batch in val_batches:
            batch_dev = {k: v.to(self.device) for k, v in batch.items()}
            loss_fn = make_functional_loss(
                self.model,
                batch_dev["input_ids"],
                batch_dev["labels"],
                batch_dev.get("attention_mask"),
            )
            grads, _ = compute_grad_and_loss(loss_fn, params)
            if total_grads is None:
                total_grads = {n: g.clone() for n, g in grads.items()}
            else:
                for n in total_grads:
                    total_grads[n] += grads[n]

        assert total_grads is not None
        n = len(val_batches)
        return {name: g / n for name, g in total_grads.items()}

    def backward_pass(
        self,
        train_batches: list[dict[str, Tensor]],
        val_batches: list[dict[str, Tensor]],
        score_batches: list[dict[str, Tensor]],
        score_batch_indices: list[list[int]],
        progress: bool = True,
    ) -> dict[int, float]:
        """Phase 2: reverse iteration computing per-example quality scores.

        train_batches drive the HVP costate update (same batches as forward pass).
        score_batches cover the entire pool so every record gets a score at each
        compute_ct_interval step.
        """
        cfg = self.config
        n_batches = len(train_batches)

        ckpt_steps = sorted(self.ckpt_mgr.checkpoint_steps())
        segments = list(zip(ckpt_steps[:-1], ckpt_steps[1:], strict=True))

        lam: dict[str, Tensor] | None = None
        grad_gamma: dict[int, float] = {}

        bar = segments[::-1]
        if progress:
            bar = tqdm(bar, desc="PMP backward (segments)", leave=False)

        for seg_start, seg_end in bar:
            seg_params_cpu = self._reconstruct_segment_to_cpu(
                seg_start, seg_end, train_batches
            )

            for offset in range(seg_end - seg_start - 1, -1, -1):
                step = seg_start + offset
                params_at_step = {
                    n: p.to(self.device)
                    for n, p in seg_params_cpu[offset].items()
                }

                g_dev = self._compute_val_grad(params_at_step, val_batches)

                if lam is None:
                    lam = {n: g.clone() for n, g in g_dev.items()}
                else:
                    for n in lam:
                        lam[n] = lam[n] + g_dev[n]

                if step % cfg.compute_ct_interval == 0:
                    score_iter = enumerate(
                        zip(score_batches, score_batch_indices, strict=True)
                    )
                    if progress:
                        score_iter = tqdm(
                            score_iter,
                            total=len(score_batches),
                            desc=f"  scoring pool (step {step})",
                            leave=False,
                        )
                    for _bi, (sbatch, ridxs) in score_iter:
                        sb_dev = {k: v.to(self.device) for k, v in sbatch.items()}
                        scores = per_example_jvp_scores(
                            self.model,
                            params_at_step,
                            sb_dev["input_ids"],
                            sb_dev["labels"],
                            sb_dev["attention_mask"],
                            lam,
                            chunk_size=cfg.chunk_size,
                        )
                        for i, ridx in enumerate(ridxs):
                            grad_gamma[ridx] = (
                                grad_gamma.get(ridx, 0.0) + scores[i].item()
                            )

                batch = train_batches[step % n_batches]
                batch_dev = {k: v.to(self.device) for k, v in batch.items()}
                loss_fn = make_functional_loss(
                    self.model,
                    batch_dev["input_ids"],
                    batch_dev["labels"],
                    batch_dev.get("attention_mask"),
                )
                hvp_result = hvp(loss_fn, params_at_step, lam)
                for n in lam:
                    lam[n] = lam[n] - cfg.pmp_lr * hvp_result[n]

            del seg_params_cpu, params_at_step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return grad_gamma

    def solve(
        self,
        train_batches: list[dict[str, Tensor]],
        val_batches: list[dict[str, Tensor]],
        score_batches: list[dict[str, Tensor]],
        score_batch_indices: list[list[int]],
        progress: bool = True,
    ) -> dict[int, float]:
        """Full PMP solve: forward pass then backward pass.

        train_batches are used in the forward SGD pass and for HVP in backward.
        score_batches (covering the full pool) are scored at each backward
        compute_ct_interval step.
        """
        self.forward_pass(train_batches, progress=progress)
        return self.backward_pass(
            train_batches,
            val_batches,
            score_batches,
            score_batch_indices,
            progress=progress,
        )
