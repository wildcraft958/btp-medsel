"""3DS: Decomposed Difficulty Data Selection (Zhong et al., EMNLP 2025).

Faithful to arxiv.org/abs/2410.10901 and github.com/PuppyKnightUniversity/3DS.

Two-stage pipeline:
  Stage 1 -- quality filtering via structural checks (adapted for curated MCQ datasets).
  Stage 2 -- three perplexity-based difficulty metrics with attention-based importance weighting,
  followed by percentile-based Goldilocks filtering.

D1 (Instruction Understanding): causal LM perplexity of the prompted instruction.
D2 (Response Confidence): attention-weighted perplexity of the model's predicted answer.
D3 (Response Correctness): attention-weighted perplexity of the reference answer.

Goldilocks selection keeps examples where all three metrics fall within a percentile range
(default 25th-75th). Among those, examples closer to the median of all three distributions
score higher, so select_top_k picks the most central examples first.

MCQ adaptation: the paper targets free-form instruction-response pairs. For MedMCQA the
"response" is the full option text (text mode, not letter mode). When a rationale is
available it is appended to the reference answer for D3, giving attention weighting more
tokens to work with.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from medsel.prompts.templates import render_continuation, render_prompt
from medsel.schema import QAExample
from medsel.selection.base import Scorer, register_scorer

__all__ = ["ThreeDSScorer"]

ATTEN_METHODS = ("mean", "max")


def _quality_pass(example: Any) -> bool:
    """Structural quality check for MCQ examples."""
    if not isinstance(example, QAExample) or not example.options:
        return False
    if len(example.question.strip()) < 10:
        return False
    texts = list(example.options.values())
    if len(set(texts)) < 2:
        return False
    if any(not t.strip() for t in texts):
        return False
    return True


def _instruction_ppl_and_embedding(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_length: int,
    device: Any,
) -> tuple[float, Any]:
    """D1: causal LM perplexity of the instruction, plus mean-pooled embedding.

    Matches calculate_ins_ppl_emb.py::text_perplexity_and_embedding().
    """
    import torch

    ids = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
    ids = ids.to(device)
    with torch.no_grad():
        out = model(ids, labels=ids.contiguous(), output_hidden_states=True)
        ppl = float(torch.exp(out.loss))
        embedding = out.hidden_states[-1].mean(dim=1).squeeze(0).cpu()
    return ppl, embedding


def _answer_ppl_and_attention(
    model: Any,
    tokenizer: Any,
    full_text: str,
    answer_span: str,
    max_length: int,
    device: Any,
    atten_method: str,
) -> tuple[float, float]:
    """Attention-weighted PPL of an answer span, plus raw PPL.

    Matches calculate_attention.py::answer_perplexity_and_attention().
    Returns (attention_weighted_ppl, raw_ppl).
    """
    import torch
    import torch.nn as nn

    ids = tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    ids = ids.to(device)

    start_idx = full_text.rfind(answer_span)
    if start_idx < 0:
        start_idx = 0
    prefix_text = full_text[:start_idx]
    start_token = len(tokenizer.encode(prefix_text, add_special_tokens=False))

    labels = ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        out = model(ids, labels=labels, output_attentions=True)
        raw_ppl = float(torch.exp(out.loss))

        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        losses = per_token_loss.view(-1)

        atten = out.attentions[-1][0].detach().cpu()

        if atten_method == "max":
            agg, _ = torch.max(atten, dim=0)
            importance = torch.mean(agg, dim=0)
        else:
            agg = torch.sum(atten, dim=0)
            agg = torch.mean(agg, dim=0)
            seq_len = agg.shape[0]
            for i in range(seq_len):
                denom = seq_len - i
                if denom > 0:
                    agg[i] /= denom
            importance = agg

    answer_losses = losses[start_token - 1 :].cpu() if start_token > 0 else losses.cpu()
    answer_importance = importance[start_token:] if start_token > 0 else importance

    n = min(len(answer_losses), len(answer_importance))
    if n == 0:
        return raw_ppl, raw_ppl

    al = answer_losses[:n].float()
    ai = answer_importance[:n].float()
    weight_sum = ai.sum()
    if weight_sum > 0:
        weighted_loss = float((ai * al).sum() / weight_sum)
    else:
        weighted_loss = float(al.mean())
    atten_ppl = math.exp(weighted_loss)

    return atten_ppl, raw_ppl


def _predict_option(
    model: Any,
    tokenizer: Any,
    example: QAExample,
    device: Any,
) -> str:
    """Return the option key the model predicts (highest logprob continuation)."""
    from medsel.eval.mcq import score_example

    scores = score_example(model, tokenizer, example, mode="text", device=device)
    return scores.predicted_key


@register_scorer("3ds")
class ThreeDSScorer(Scorer):
    """Decomposed Difficulty Data Selection for MCQ SFT data.

    Computes three perplexity-based difficulty metrics per example, applies attention-based
    importance weighting to D2 and D3, then selects examples whose difficulty falls within a
    configurable percentile range (the Goldilocks zone).
    """

    name: ClassVar[str] = "3ds"

    def __init__(
        self,
        judge: str = "Qwen/Qwen3-1.7B-Base",
        *,
        max_length: int = 1024,
        dtype: str = "auto",
        low_th: float = 25.0,
        up_th: float = 75.0,
        atten_method: str = "mean",
        use_rationale: bool = True,
        cache_dir: str | None = None,
        progress: bool = True,
    ) -> None:
        if not 0.0 <= low_th < up_th <= 100.0:
            raise ValueError(f"need 0 <= low_th < up_th <= 100, got [{low_th}, {up_th}]")
        if atten_method not in ATTEN_METHODS:
            raise ValueError(f"atten_method must be one of {ATTEN_METHODS}, got {atten_method!r}")
        self.judge_name = judge
        self.max_length = max_length
        self.dtype = dtype
        self.low_th = low_th
        self.up_th = up_th
        self.atten_method = atten_method
        self.use_rationale = use_rationale
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.progress = progress
        self._loaded: tuple[Any, Any] | None = None

    def _model(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from medsel.eval.runner import load_model

            self._loaded = load_model(
                self.judge_name, dtype=self.dtype, attn_implementation="eager"
            )
        return self._loaded

    def release_model(self) -> None:
        """Free GPU memory after scoring is complete."""
        self._loaded = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _build_answer_text(self, example: QAExample, key: str) -> str:
        """Render the answer span for D2/D3."""
        text = render_continuation(example, key, mode="text").lstrip()
        if (
            self.use_rationale
            and example.rationale
            and key == example.answer_key
        ):
            text = f"{text}\n\nExplanation: {example.rationale.strip()}"
        return text

    def _score_one(
        self,
        example: QAExample,
        model: Any,
        tokenizer: Any,
        device: Any,
    ) -> dict[str, Any]:
        """Compute all three difficulty metrics for one example."""
        prompt = render_prompt(example)

        d1, embedding = _instruction_ppl_and_embedding(
            model, tokenizer, prompt, self.max_length, device
        )

        pred_key = _predict_option(model, tokenizer, example, device)
        pred_answer_text = self._build_answer_text(example, pred_key)
        full_text_d2 = prompt + " " + pred_answer_text
        d2_atten, d2_raw = _answer_ppl_and_attention(
            model, tokenizer, full_text_d2, pred_answer_text,
            self.max_length, device, self.atten_method,
        )

        if example.answer_key:
            gold_answer_text = self._build_answer_text(example, example.answer_key)
        else:
            gold_answer_text = pred_answer_text
        full_text_d3 = prompt + " " + gold_answer_text
        d3_atten, d3_raw = _answer_ppl_and_attention(
            model, tokenizer, full_text_d3, gold_answer_text,
            self.max_length, device, self.atten_method,
        )

        return {
            "d1": d1,
            "d2": d2_atten,
            "d3": d3_atten,
            "d2_raw": d2_raw,
            "d3_raw": d3_raw,
            "embedding": embedding,
        }

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.cache_dir is None:
            return {}
        cache_file = self.cache_dir / "tds_cache.jsonl"
        if not cache_file.exists():
            return {}
        cached: dict[str, dict[str, Any]] = {}
        for line in cache_file.read_text().splitlines():
            entry = json.loads(line)
            cached[entry["uid"]] = entry
        return cached

    def _save_cache_entry(self, uid: str, entry: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "tds_cache.jsonl"
        row = {"uid": uid, **{k: v for k, v in entry.items() if k != "embedding"}}
        with open(cache_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    def score(self, records: Sequence[Any]) -> list[float]:
        if not records:
            return []

        from tqdm.auto import tqdm

        quality = [_quality_pass(r) for r in records]

        model, tokenizer = self._model()
        model.eval()
        device = next(model.parameters()).device

        cached = self._load_cache()

        metrics: list[dict[str, Any] | None] = [None] * len(records)
        scored_indices = []

        for i, record in enumerate(
            tqdm(records, desc="3ds:scoring", unit="ex", disable=not self.progress)
        ):
            if not quality[i]:
                continue

            uid = record.uid if hasattr(record, "uid") else str(i)
            if uid in cached:
                metrics[i] = cached[uid]
                scored_indices.append(i)
                continue

            m = self._score_one(record, model, tokenizer, device)
            entry = {"d1": m["d1"], "d2": m["d2"], "d3": m["d3"]}
            metrics[i] = entry
            scored_indices.append(i)
            self._save_cache_entry(uid, entry)

        if not scored_indices:
            return [float("-inf")] * len(records)

        d1_vals = np.array([metrics[i]["d1"] for i in scored_indices])
        d2_vals = np.array([metrics[i]["d2"] for i in scored_indices])
        d3_vals = np.array([metrics[i]["d3"] for i in scored_indices])

        d1_pct = _percentile_ranks(d1_vals)
        d2_pct = _percentile_ranks(d2_vals)
        d3_pct = _percentile_ranks(d3_vals)

        pct_map: dict[int, tuple[float, float, float]] = {}
        for j, idx in enumerate(scored_indices):
            pct_map[idx] = (d1_pct[j], d2_pct[j], d3_pct[j])

        scores: list[float] = []
        for i in range(len(records)):
            if not quality[i] or metrics[i] is None:
                scores.append(float("-inf"))
                continue

            p1, p2, p3 = pct_map[i]
            if not (self.low_th <= p1 <= self.up_th
                    and self.low_th <= p2 <= self.up_th
                    and self.low_th <= p3 <= self.up_th):
                scores.append(float("-inf"))
                continue

            dist = abs(p1 - 50.0) + abs(p2 - 50.0) + abs(p3 - 50.0)
            scores.append(-dist)

        return scores

    def __repr__(self) -> str:
        return (
            f"ThreeDSScorer(judge={self.judge_name!r}, "
            f"range=[{self.low_th}, {self.up_th}], atten={self.atten_method!r})"
        )


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Percentile rank of each value within the array (0-100)."""
    n = len(values)
    if n == 0:
        return values
    order = values.argsort().argsort()
    return 100.0 * order / max(n - 1, 1)
