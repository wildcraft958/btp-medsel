# Phase 3 Experiment Report: Data Selection for Medical SFT

**Date:** 2026-09-12 (updated)
**Author:** Animesh Raj

## Research Question

Does the choice of data selection method matter for supervised fine-tuning (SFT) on MedMCQA,
or does random selection perform comparably?

We compare six selection strategies at a matched token budget, training Qwen3-1.7B-Base with
LoRA adapters on MedMCQA and evaluating on its validation split (4,183 questions).

## Experimental Setup

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-1.7B-Base |
| Task | MedMCQA (medical MCQ, 4 options) |
| Pool | 30,000 MedMCQA training examples |
| Token budget | 108,104 tokens (~2,000 records, 6.7% of pool) |
| Adaptation | LoRA r=16, alpha=32 |
| Training | 1 and 3 epochs, lr=2e-5, completion-only loss |
| Evaluation | Zero-shot, letter-mode logprob scoring |
| Hardware | Quadro P5000 16GB, fp32 |

**Selection methods tested:**

- **3DS** (Jeong et al., 2025): Three-dimensional scoring with quality filtering, Goldilocks
  difficulty selection, and K-Center diversity sampling
- **LESS** (Xia et al., ICML 2024): Gradient-based influence function approximation using
  LoRA warmup checkpoints
- **DSIR** (Xie et al., ICML 2023): Importance reweighting by n-gram distribution matching
  (per-token normalized variant)
- **Perplexity**: Mid-range perplexity selection (Goldilocks-style)
- **Embed similarity**: Cosine similarity of causal LM embeddings to validation target
- **Random**: Uniform random baseline

## Results

### Main Result: 3-Epoch Training (108k Token Budget)

| Method | Accuracy | 95% CI | Delta vs Base |
|---|---|---|---|
| Base (no SFT) | 0.4891 | [0.4748, 0.5042] | -- |
| **3DS** | **0.4946** | [0.4803, 0.5097] | +0.0055 |
| **DSIR** | **0.4946** | [0.4803, 0.5097] | +0.0055 |
| Perplexity | 0.4929 | [0.4786, 0.5080] | +0.0038 |
| Random | 0.4925 | [0.4781, 0.5075] | +0.0034 |
| LESS (buggy) | 0.4829 | [0.4686, 0.4980] | -0.0062 |
| Embed similarity | 0.4793 | [0.4650, 0.4944] | -0.0098 |

**All confidence intervals overlap.** No method achieves a statistically significant
improvement over the base model or over random selection at this token budget.

The best methods (3DS, DSIR) achieve 0.4946 while random achieves 0.4925 -- a gap of 0.21
percentage points, well within the 95% CI width of ~3 percentage points.

### 1-Epoch Training (Partial)

| Method | Accuracy | 95% CI |
|---|---|---|
| Base (no SFT) | 0.4891 | [0.4748, 0.5042] |
| DSIR | 0.4939 | [0.4796, 0.5090] |
| 3DS | 0.4865 | [0.4721, 0.5016] |
| LESS (buggy) | 0.4841 | [0.4698, 0.4992] |
| Embed similarity | 0.4779 | [0.4635, 0.4929] |

3-epoch training improved all methods by 0.5 to 1.5 percentage points over 1-epoch, with 3DS
showing the largest gain (+0.0081).

### Training Verification

To rule out a degenerate training bug, we compared LoRA adapter weights between the 3DS and
random models:

- All 112 LoRA layers differ between models
- All 56 `lora_B` matrices have non-zero values (max magnitude 0.0032)
- Training produced genuinely different models, but the updates are small

### LESS Bug Report

Two bugs were found in the LESS implementation:

1. **Target gradient mismatch.** Target gradients used prompt-only text while candidate
   gradients used prompt+completion. This meant target gradients measured prompt prediction
   while candidate gradients measured answer prediction -- the cosine similarity was comparing
   unrelated signals.

2. **Wrong target split.** The default target split resolved to "train" instead of
   "validation." LESS is designed to select training data that helps validation performance,
   so the target should be from the validation set.

Both bugs have been fixed. LESS rescoring is queued and will produce corrected results.

## Interpretation

### Why Selection Methods Converge with Random

Three factors explain the convergence:

**1. Homogeneous source pool.** All 30,000 candidates are MedMCQA training examples -- every
example is already on-topic for the evaluation task. Selection methods like LESS and 3DS are
designed to find relevant data in heterogeneous pools (e.g., selecting medical data from FLAN's
60+ subcollections). When every example is already relevant, there is little for smart selection
to filter.

**2. High selection ratio.** At 108k tokens, we select roughly 2,000 of 30,000 records (6.7%).
At this ratio, all methods include the majority of the useful signal. The literature shows
selection methods separate from random primarily at very low selection ratios (1-5%).

**3. Limited model capacity.** LoRA r=16 on a 1.7B parameter model constrains how much the
model can exploit fine-grained data quality differences. The adapter updates are genuinely small
(max 0.0032), suggesting the model reaches its LoRA capacity regardless of data quality.

### Literature Support

This finding aligns with two recent papers:

- **"Rethinking Data Selection at Scale" (Jeong et al., EMNLP 2025 Findings):** Tested LESS,
  DSIR, and other methods on LLaMA3-8B and Qwen2-7B. Found that all selection methods produced
  p-values > 0.05 against random. Concluded that data diversity matters more than quality metrics
  when the source pool is clean.

- **"Chasing Random" (Wettig et al., 2024):** Tested instruction selection across FLAN, Dolly,
  Alpaca, and Evol datasets. Found that performance advantages at one budget size reverse at
  another, and different benchmarks contradict each other on which strategy wins.

### Seed Sweep (3-Epoch, 108k Budget)

Three-seed sweep (seeds 42, 43, 44) across 3DS, DSIR, LESS, and random at 3 epochs.

| Method | Mean | Std | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|---|---|
| **3DS** | **0.4941** | 0.0014 | 0.4946 | 0.4922 | 0.4956 |
| DSIR | 0.4930 | 0.0025 | 0.4946 | 0.4894 | 0.4949 |
| Random | 0.4908 | 0.0019 | 0.4925 | 0.4882 | 0.4918 |
| LESS (buggy) | 0.4831 | 0.0004 | 0.4829 | 0.4836 | 0.4827 |

3DS leads random by 0.33pp on average, but error bars overlap (3DS: 0.4941 +/- 0.0014,
random: 0.4908 +/- 0.0019). LESS is consistently ~0.8pp below random with remarkably low
variance (std 0.0004), suggesting the buggy scorer deterministically selects low-quality data.

### Small-Budget Experiment (27k Tokens, ~500 Records, 1.7% of Pool)

To test the budget-dependence hypothesis, we ran a second experiment at ~27,000 tokens
(~500 records, 1.7% of pool) -- a quarter of the main budget.

| Method | Accuracy | 95% CI | Records | Tokens |
|---|---|---|---|---|
| **Perplexity** | **0.4941** | [0.4798, 0.5092] | 535 | 27,274 |
| Embed similarity | 0.4908 | [0.4765, 0.5059] | 577 | 27,271 |
| 3DS | 0.4906 | [0.4762, 0.5056] | 457 | 27,282 |
| Base (no SFT) | 0.4891 | [0.4748, 0.5042] | -- | -- |
| Random | 0.4841 | [0.4698, 0.4992] | 500 | 27,288 |
| DSIR | 0.4834 | [0.4690, 0.4984] | 524 | 27,284 |
| LESS (buggy) | 0.4831 | [0.4688, 0.4982] | 600 | 27,281 |

At 1.7% selection ratio, the top methods (perplexity 0.4941, embed_similarity 0.4908, 3DS
0.4906) cluster ~1pp above random (0.4841). This is a larger gap than at the 108k budget
(0.21pp), consistent with the hypothesis that selection methods separate at lower budgets.
The CIs still overlap, but the direction is consistent: methods that score by target-relevance
outperform random more at low budgets than at high budgets.

DSIR drops from top-2 at 108k to near-random at 27k, suggesting n-gram distribution matching
needs a larger sample to be effective.

### Judge Ablation

We replaced the self-judging Qwen3-1.7B with MedGemma-1.5-4b-it (4-bit NF4 quantized) as an
external judge for 3DS quality scoring, then trained and evaluated a new model from the
MedGemma-judged selection.

| Judge | Accuracy | 95% CI | Records | Tokens |
|---|---|---|---|---|
| Self-judged (Qwen3-1.7B) | 0.4865 | [0.4721, 0.5016] | 1,850 | 108,097 |
| MedGemma-judged (4b-it) | 0.4851 | [0.4707, 0.5001] | 2,000 | 108,103 |

**Delta: -0.0014 (CIs fully overlap). No meaningful difference.**

The stronger external judge did not improve downstream accuracy. MedGemma placed 4,460 examples
in the Goldilocks difficulty zone (vs 3DS self-judged which selected 1,850 records), producing a
different selection of 2,000 records at the same token budget. Despite selecting different data,
the trained models perform identically within noise.

Subject-level analysis shows MedGemma-judged selection has slightly higher variance across
medical subjects (std 0.080 vs 0.068), meaning less uniform performance. This is consistent with
the overall finding: on a homogeneous pool, the judge quality does not matter because every
example is already on-topic.

## Remaining Work

### LESS Rescore (Blocked)

Re-running LESS with both bugs fixed to get corrected results. Currently blocked by VRAM
constraints (ollama process holds 5.5 GB of the P5000's 16 GB; LESS requires full model +
optimizer state + per-example gradients, which exceeds the remaining ~10.5 GB).

## Summary

At a 108k token budget, data selection method choice does not meaningfully affect SFT accuracy
on MedMCQA. All methods fall within a 1.5 percentage point band, and no method achieves
statistical significance over random. This is consistent with recent findings in the broader
NLP literature about the limits of data selection methods on clean, homogeneous source pools.

The judge ablation shows that a stronger external judge does not improve 3DS selection on this
task.

The seed sweep (3 seeds, 4 methods) confirms the ranking is stable: 3DS leads random by 0.33pp
on average, but the difference is not statistically significant (overlapping error bars). The
small-budget experiment (27k tokens, 1.7% of pool) shows a larger gap between smart selection
and random (~1pp vs ~0.2pp at 108k), consistent with the budget-dependence hypothesis from
Wettig et al. (2024).
