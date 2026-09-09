"""LESS: Selecting Influential Data for Targeted Instruction Tuning (Xia et al., ICML 2024).

Faithful to arxiv.org/abs/2402.04333 and github.com/princeton-nlp/LESS.

Four-step pipeline:
  1. Warmup LoRA training on a random subset of the pool (4 epochs, checkpoint each).
  2. Per-example gradient computation with Adam correction at each checkpoint.
  3. Random projection (block-wise Rademacher) to a low-dimensional space.
  4. Selection by cosine similarity between projected candidate and target gradients.

The random projection avoids materializing a full D x d matrix by generating Rademacher
blocks on the fly with a seeded generator. This is equivalent to TRAK's BasicProjector
but avoids the dependency (CudaProjector needs compute >= 7.0, our P5000 is 6.1).

Adapted for MCQ data: the "instruction" is the rendered prompt, the "response" is the
answer text. Completion-only loss matches our SFT pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from medsel.schema import QAExample
from medsel.selection.base import register_scorer
from medsel.selection.targets import TargetedScorer

__all__ = ["LESSScorer"]


_PROJ_INDICES_CACHE: dict[tuple[int, int, int], Any] = {}


def _rademacher_project(
    grad: Any,
    proj_dim: int,
    seed: int,
) -> Any:
    """Project a gradient vector via random coordinate selection.

    The index array is cached so randperm runs only once per (D, proj_dim, seed).
    """
    import torch

    grad = grad.float().flatten()
    D = grad.shape[0]
    cache_key = (D, proj_dim, seed)
    if cache_key not in _PROJ_INDICES_CACHE:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        _PROJ_INDICES_CACHE[cache_key] = torch.randperm(D, generator=gen)[:proj_dim]
    indices = _PROJ_INDICES_CACHE[cache_key].to(grad.device)
    sketch = grad[indices]
    sketch /= math.sqrt(proj_dim / D)
    return sketch


def _adam_correct(
    grad: Any,
    m: Any,
    v: Any,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Any:
    """Apply one-step Adam correction to a raw gradient.

    This is the LESS paper's key insight: using Adam-corrected gradients accounts for
    the optimizer's per-parameter scaling, which matters for influence estimation.
    """
    updated_m = beta1 * m + (1 - beta1) * grad
    updated_v = beta2 * v + (1 - beta2) * grad.pow(2)
    return updated_m / (updated_v.sqrt() + eps)


@register_scorer("less")
class LESSScorer(TargetedScorer):
    """Influence-based data selection using gradient similarity.

    Scores each candidate by how much its training gradient aligns with the target
    validation examples' gradients, after Adam correction and random projection.
    """

    name: ClassVar[str] = "less"

    def __init__(
        self,
        judge: str = "Qwen/Qwen3-1.7B-Base",
        *,
        target: str = "medmcqa",
        target_split: str | None = None,
        target_limit: int = 50,
        max_length: int = 1024,
        dtype: str = "auto",
        lora_r: int = 128,
        lora_alpha: int = 256,
        warmup_epochs: int = 4,
        warmup_fraction: float = 0.05,
        warmup_lr: float = 2e-5,
        warmup_batch_size: int = 1,
        warmup_grad_accum: int = 16,
        proj_dim: int = 8192,
        proj_seed: int = 42,
        cache_dir: str | None = None,
        progress: bool = True,
    ) -> None:
        super().__init__(target, target_split=target_split, target_limit=target_limit)
        if proj_dim < 1:
            raise ValueError(f"proj_dim must be positive, got {proj_dim}")
        self.judge_name = judge
        self.max_length = max_length
        self.dtype = dtype
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.warmup_epochs = warmup_epochs
        self.warmup_fraction = warmup_fraction
        self.warmup_lr = warmup_lr
        self.warmup_batch_size = warmup_batch_size
        self.warmup_grad_accum = warmup_grad_accum
        self.proj_dim = proj_dim
        self.proj_seed = proj_seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.progress = progress
        self._loaded: tuple[Any, Any] | None = None

    def _model_and_tokenizer(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from medsel.eval.runner import load_model

            self._loaded = load_model(self.judge_name, dtype=self.dtype)
        return self._loaded

    def release_model(self) -> None:
        self._loaded = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _checkpoints_dir(self) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / "warmup_checkpoints"

    def _warmup_done(self) -> bool:
        if self.cache_dir is None:
            return False
        ckpt_dir = self._checkpoints_dir()
        if not ckpt_dir.exists():
            return False
        return all(
            (ckpt_dir / f"epoch_{e}" / "adapter_model.safetensors").exists()
            for e in range(1, self.warmup_epochs + 1)
        )

    def _run_warmup(self, records: Sequence[Any]) -> None:
        """Train LoRA on a random subset, saving per-epoch checkpoints."""
        import random as stdlib_random

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from medsel.prompts.templates import render_prompt
        from medsel.stages.sft import render_completion
        from medsel.utils.device import pick_device, resolve_dtype

        assert self.cache_dir is not None
        ckpt_dir = self._checkpoints_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        n_warmup = max(1, int(len(records) * self.warmup_fraction))
        rng = stdlib_random.Random(42)
        warmup_indices = rng.sample(range(len(records)), min(n_warmup, len(records)))

        device = pick_device()
        tokenizer = AutoTokenizer.from_pretrained(self.judge_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.judge_name, dtype=resolve_dtype(self.dtype, device)
        )
        model = model.to(device)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.warmup_lr,
        )

        warmup_records = [records[i] for i in warmup_indices]

        from tqdm.auto import tqdm

        for epoch in range(1, self.warmup_epochs + 1):
            epoch_loss = 0.0
            rng.shuffle(warmup_indices)
            pbar = tqdm(
                warmup_records,
                desc=f"less:warmup epoch {epoch}/{self.warmup_epochs}",
                disable=not self.progress,
            )
            step = 0
            optimizer.zero_grad()

            for record in pbar:
                if not isinstance(record, QAExample) or record.answer_key is None:
                    continue

                text = render_prompt(record) + render_completion(record)
                ids = tokenizer.encode(
                    text, return_tensors="pt", truncation=True, max_length=self.max_length
                ).to(device)

                out = model(ids, labels=ids)
                loss = out.loss / self.warmup_grad_accum
                loss.backward()
                epoch_loss += out.loss.item()
                step += 1

                if step % self.warmup_grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{out.loss.item():.3f}")

            if step % self.warmup_grad_accum != 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_dir = ckpt_dir / f"epoch_{epoch}"
            model.save_pretrained(epoch_dir)
            torch.save(optimizer.state_dict(), epoch_dir / "optimizer.pt")

            n_steps = step // self.warmup_grad_accum + (1 if step % self.warmup_grad_accum else 0)
            avg_loss = epoch_loss / max(step, 1)
            print(f"  epoch {epoch}: {n_steps} steps, avg loss {avg_loss:.4f}")

        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _get_lora_grad_and_optimizer_state(
        self, model: Any, tokenizer: Any, text: str, optimizer_state: dict, device: Any
    ) -> tuple[Any, Any, Any]:
        """Forward + backward on one example, return (flat_grad, flat_m, flat_v) for LoRA params."""
        import torch

        model.zero_grad()
        ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(device)

        prompt_end = len(tokenizer.encode(
            text.split("\nAnswer:")[0] + "\nAnswer:" if "\nAnswer:" in text else text,
            add_special_tokens=False,
        ))
        labels = ids.clone()
        if prompt_end < ids.shape[1]:
            labels[0, :prompt_end] = -100

        out = model(ids, labels=labels)
        out.loss.backward()

        grads = []
        ms = []
        vs = []
        param_groups = optimizer_state.get("state", {})

        lora_param_ids = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                lora_param_ids.append(name)
                grads.append(param.grad.detach().flatten())

        flat_grad = torch.cat(grads)

        for idx in range(len(lora_param_ids)):
            if idx in param_groups:
                state = param_groups[idx]
                ms.append(state["exp_avg"].flatten())
                vs.append(state["exp_avg_sq"].flatten())
            else:
                n = grads[idx].shape[0]
                ms.append(torch.zeros(n, device=device))
                vs.append(torch.zeros(n, device=device))

        flat_m = torch.cat(ms)
        flat_v = torch.cat(vs)

        model.zero_grad()
        return flat_grad, flat_m, flat_v

    def _compute_projected_grad(
        self, model: Any, tokenizer: Any, text: str, optimizer_state: dict, device: Any
    ) -> Any:
        """Compute Adam-corrected, projected gradient for one example at one checkpoint."""
        import torch

        grad, m, v = self._get_lora_grad_and_optimizer_state(
            model, tokenizer, text, optimizer_state, device
        )
        corrected = _adam_correct(grad, m, v)
        projected = _rademacher_project(corrected, self.proj_dim, self.proj_seed)
        norm = torch.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected.cpu()

    def _load_checkpoint(self, epoch: int, device: Any) -> tuple[Any, dict]:
        """Load a warmup checkpoint (model + optimizer state)."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from medsel.utils.device import resolve_dtype

        ckpt_dir = self._checkpoints_dir() / f"epoch_{epoch}"

        model = AutoModelForCausalLM.from_pretrained(
            self.judge_name, dtype=resolve_dtype(self.dtype, device)
        ).to(device)
        model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=True)
        model.train()

        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.exists():
            opt_state = torch.load(
                opt_path, map_location=device, weights_only=True
            )
        else:
            opt_state = {}

        return model, opt_state

    def _grad_cache_path(self, uid: str, epoch: int, kind: str) -> Path:
        assert self.cache_dir is not None
        safe = uid.replace("/", "_")
        return self.cache_dir / "grads" / kind / f"epoch_{epoch}" / f"{safe}.pt"

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []

        import torch
        from tqdm.auto import tqdm

        from medsel.prompts.templates import render_prompt
        from medsel.stages.sft import render_completion
        from medsel.utils.device import pick_device

        device = pick_device()

        if self.cache_dir is None:
            raise ValueError("LESS scorer requires cache_dir for warmup checkpoints and gradients")

        if not self._warmup_done():
            print("LESS: running warmup training...")
            self._run_warmup(records)
        else:
            print(f"LESS: warmup checkpoints found at {self._checkpoints_dir()}")

        target_texts = self.target_texts()
        print(f"LESS: {len(target_texts)} target examples for gradient similarity")

        scores = [0.0] * len(records)

        for epoch in range(1, self.warmup_epochs + 1):
            print(f"LESS: processing checkpoint epoch {epoch}/{self.warmup_epochs}")

            all_target_cached = all(
                self._grad_cache_path(f"target_{i}", epoch, "target").exists()
                for i in range(len(target_texts))
            )
            all_candidate_cached = all(
                self._grad_cache_path(
                    r.uid if hasattr(r, "uid") else str(j), epoch, "candidate"
                ).exists()
                for j, r in enumerate(records)
                if isinstance(r, QAExample) and r.answer_key is not None
            )
            need_model = not (all_target_cached and all_candidate_cached)

            model = opt_state = tokenizer = None
            if need_model:
                model, opt_state = self._load_checkpoint(epoch, device)
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.judge_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            else:
                print(f"  epoch {epoch}: all grads cached, skipping model load")

            target_grads = []
            for i, text in enumerate(
                tqdm(
                    target_texts,
                    desc=f"less:target grads (epoch {epoch})",
                    disable=not self.progress,
                )
            ):
                cache_path = self._grad_cache_path(f"target_{i}", epoch, "target")
                if cache_path.exists():
                    target_grads.append(torch.load(cache_path, weights_only=True))
                    continue
                proj = self._compute_projected_grad(model, tokenizer, text, opt_state, device)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(proj, cache_path)
                target_grads.append(proj)

            mean_target = torch.stack(target_grads).mean(dim=0)
            tnorm = torch.norm(mean_target)
            if tnorm > 0:
                mean_target = mean_target / tnorm

            for i, record in enumerate(
                tqdm(
                    records,
                    desc=f"less:candidate grads (epoch {epoch})",
                    disable=not self.progress,
                )
            ):
                if not isinstance(record, QAExample) or record.answer_key is None:
                    scores[i] = float("-inf")
                    continue

                uid = record.uid if hasattr(record, "uid") else str(i)
                cache_path = self._grad_cache_path(uid, epoch, "candidate")

                if cache_path.exists():
                    proj = torch.load(cache_path, weights_only=True)
                else:
                    text = render_prompt(record) + render_completion(record)
                    proj = self._compute_projected_grad(model, tokenizer, text, opt_state, device)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(proj, cache_path)

                sim = float(torch.dot(proj, mean_target))
                scores[i] += sim

            if model is not None:
                del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        n_epochs = self.warmup_epochs
        scores = [s / n_epochs if s != float("-inf") else s for s in scores]

        self.release_model()
        return scores

    def __repr__(self) -> str:
        return (
            f"LESSScorer(judge={self.judge_name!r}, "
            f"proj_dim={self.proj_dim}, lora_r={self.lora_r})"
        )
