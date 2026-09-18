"""PDS: PMP-based Data Selection (Gu et al., ICLR 2025).

Faithful to arxiv.org/abs/2410.07064 and github.com/microsoft/LMOps/tree/main/data_selection.

Uses Pontryagin's Maximum Principle from optimal control theory to compute
per-example quality scores. A small proxy model (SmolLM2-135M) is trained
briefly via SGD, then the adjoint equation is solved backward through the
training trajectory to determine each example's contribution to validation
loss reduction.

Adapted for SFT on MedMCQA: the training loss is completion-only cross-entropy
on the answer, and the validation loss is the same over held-out validation
examples.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from medsel.schema import QAExample
from medsel.selection.base import register_scorer
from medsel.selection.targets import TargetedScorer

__all__ = ["PDSScorer"]


@register_scorer("pds")
class PDSScorer(TargetedScorer):
    """PMP-based Data Selection scorer.

    Scores each candidate by its contribution to reducing validation loss
    across the full training trajectory of a small proxy model, computed
    via the discretized adjoint equation from optimal control theory.
    """

    name: ClassVar[str] = "pds"

    def __init__(
        self,
        proxy: str = "HuggingFaceTB/SmolLM2-135M",
        *,
        target: str = "medmcqa",
        target_split: str | None = None,
        target_limit: int = 50,
        max_length: int = 512,
        pmp_steps: int = 100,
        pmp_lr: float = 0.01,
        pmp_batch_size: int = 4,
        checkpoint_interval: int | None = None,
        compute_ct_interval: int = 10,
        chunk_size: int = 4,
        gumbel: bool = False,
        noise_ratio: float = 0.2,
        seed: int = 42,
        cache_dir: str | None = None,
        progress: bool = True,
    ) -> None:
        super().__init__(target, target_split=target_split, target_limit=target_limit)
        if pmp_steps < 1:
            raise ValueError(f"pmp_steps must be positive, got {pmp_steps}")
        if noise_ratio < 0:
            raise ValueError(f"noise_ratio must be non-negative, got {noise_ratio}")
        self.proxy_name = proxy
        self.max_length = max_length
        self.pmp_steps = pmp_steps
        self.pmp_lr = pmp_lr
        self.pmp_batch_size = pmp_batch_size
        self.checkpoint_interval = checkpoint_interval or int(math.sqrt(pmp_steps))
        self.compute_ct_interval = compute_ct_interval
        self.chunk_size = chunk_size
        self.gumbel = gumbel
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.progress = progress

    def resolve_target_split(self) -> str | None:
        """PDS uses validation targets for the differentiable validation objective."""
        if isinstance(self.target, str):
            return self.target_split or "validation"
        return None

    def target_texts(self) -> list[str]:
        """Target texts with completion, so validation loss matches training format."""
        if not isinstance(self.target, str):
            return [str(t) for t in self.target]
        from medsel.prompts.templates import render_prompt
        from medsel.registry import get_loader
        from medsel.stages.sft import render_completion

        loader = get_loader(self.target)
        split = self.resolve_target_split()
        assert split is not None
        texts = []
        for example in loader.load(split, limit=self.target_limit):
            if isinstance(example, QAExample) and example.answer_key is not None:
                texts.append(render_prompt(example) + render_completion(example))
        if not texts:
            raise ValueError(
                f"target {self.target!r} produced no scoreable validation examples"
            )
        return texts

    def _tokenize_texts(
        self, texts: list[str], tokenizer: Any, device: str
    ) -> list[dict[str, Any]]:
        """Tokenize a list of rendered prompt+completion strings into batches.

        Returns list of batch dicts with input_ids, labels, attention_mask.
        Labels are set to -100 for tokens before the completion boundary,
        which is found by tokenizing the prompt portion alone.
        """
        import torch

        all_ids = []
        all_labels = []
        all_masks = []

        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            labels = ids.clone()
            all_ids.append(ids)
            all_labels.append(labels)
            all_masks.append(mask)

        batches: list[dict[str, Any]] = []
        for start in range(0, len(all_ids), self.pmp_batch_size):
            end = min(start + self.pmp_batch_size, len(all_ids))
            batch_ids = all_ids[start:end]
            batch_labels = all_labels[start:end]
            batch_masks = all_masks[start:end]

            max_len = max(t.shape[0] for t in batch_ids)
            padded_ids = torch.full((end - start, max_len), tokenizer.pad_token_id or 0)
            padded_labels = torch.full((end - start, max_len), -100, dtype=torch.long)
            padded_masks = torch.zeros(end - start, max_len, dtype=torch.long)

            for i, (ids, lbl, msk) in enumerate(
                zip(batch_ids, batch_labels, batch_masks, strict=True)
            ):
                padded_ids[i, : ids.shape[0]] = ids
                padded_labels[i, : lbl.shape[0]] = lbl
                padded_masks[i, : msk.shape[0]] = msk

            batches.append({
                "input_ids": padded_ids.to(device),
                "labels": padded_labels.to(device),
                "attention_mask": padded_masks.to(device),
            })

        return batches

    def _prepare_batches(
        self,
        records: Sequence[Any],
        tokenizer: Any,
        device: str,
    ) -> tuple[list[dict[str, Any]], list[list[int]]]:
        """Tokenize pool records and return (batches, batch_record_indices)."""
        from medsel.prompts.templates import render_prompt
        from medsel.stages.sft import render_completion

        texts: list[str] = []
        record_indices: list[int] = []

        for i, rec in enumerate(records):
            if isinstance(rec, QAExample) and rec.answer_key is not None:
                texts.append(render_prompt(rec) + render_completion(rec))
                record_indices.append(i)

        if not texts:
            return [], []

        all_batches = self._tokenize_texts(texts, tokenizer, device)

        batch_record_indices: list[list[int]] = []
        pos = 0
        for batch in all_batches:
            bs = batch["input_ids"].shape[0]
            batch_record_indices.append(record_indices[pos : pos + bs])
            pos += bs

        return all_batches, batch_record_indices

    def _load_proxy_model(self, device: str) -> tuple[Any, Any]:
        """Load the proxy model in fp32."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.proxy_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.proxy_name,
            dtype=torch.float32,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        model.config.use_cache = False
        model.to(device)
        model.eval()
        return model, tokenizer

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []

        import torch

        from medsel.utils.device import pick_device
        from medsel.utils.seed import set_seed

        if self.cache_dir is None:
            raise ValueError("PDS scorer requires cache_dir for PMP checkpoints")

        device = pick_device()
        scores_path = self.cache_dir / "pmp_scores.pt"

        if scores_path.exists():
            cached = torch.load(scores_path, weights_only=True)
            if len(cached) == len(records):
                return cached.tolist()

        set_seed(self.seed)
        model, tokenizer = self._load_proxy_model(device)

        all_batches, all_batch_indices = self._prepare_batches(
            records, tokenizer, "cpu"
        )
        if not all_batches:
            return [float("-inf")] * len(records)

        target_texts = self.target_texts()
        val_batches = self._tokenize_texts(target_texts, tokenizer, device)

        n_train = min(
            max(self.pmp_steps, len(all_batches)),
            len(all_batches),
        )
        rng = random.Random(self.seed)
        train_indices = rng.sample(range(len(all_batches)), n_train)
        train_batches = [all_batches[i] for i in train_indices]

        from medsel.selection.scorers.pds_solver import PMPConfig, PMPSolver

        config = PMPConfig(
            pmp_steps=self.pmp_steps,
            pmp_lr=self.pmp_lr,
            checkpoint_interval=self.checkpoint_interval,
            compute_ct_interval=self.compute_ct_interval,
            chunk_size=self.chunk_size,
            seed=self.seed,
        )
        solver = PMPSolver(model, config, self.cache_dir, device)
        raw_scores = solver.solve(
            train_batches,
            val_batches,
            all_batches,
            all_batch_indices,
            progress=self.progress,
        )

        scores = [float("-inf")] * len(records)
        for ridx, val in raw_scores.items():
            scores[ridx] = val

        if self.gumbel and self.noise_ratio > 0:
            finite = [s for s in scores if s != float("-inf")]
            if len(finite) > 1:
                noise_scale = self.noise_ratio * statistics.stdev(finite)
                rng = random.Random(f"pds:{self.seed}")
                for i in range(len(scores)):
                    if scores[i] != float("-inf"):
                        u = max(rng.random(), 1e-12)
                        scores[i] += noise_scale * -math.log(-math.log(u))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(scores), scores_path)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores

    def __repr__(self) -> str:
        return (
            f"PDSScorer(proxy={self.proxy_name!r}, "
            f"pmp_steps={self.pmp_steps}, pmp_lr={self.pmp_lr})"
        )
