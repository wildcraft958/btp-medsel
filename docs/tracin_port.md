# Porting TracIn into the scorer interface

A specification for the one selection method from the parallel CPT work that has not been ported.
It is written so that whoever picks it up does not have to rediscover the design.

`dsir` and `perplexity` are already in `src/medsel/selection/scorers/`. TracIn is not, because it
is not a scorer with a different formula. It needs a training trajectory, per-example gradients and
a caching layer, none of which the current interface has.

---

## What it computes

Influence of a candidate document on a probe set, summed over a trajectory of checkpoints:

```
s(z) = sum_t  w_t * mean over z' in T of  <g_t(z), g_t(z')>
```

`g_t` is the gradient at checkpoint `t`, `T` is the probe set, and `w_t` is the learning rate that
checkpoint was saved at. A document scores highly when training on it would move the model in the
same direction as training on the probe examples.

This is the only method under consideration that is *aimed* by gradients rather than by surface
statistics, which is why it is worth the cost.

## The five pieces

| Module | Responsibility |
|---|---|
| `checkpoints.py` | Resolve warmup checkpoints and recover the learning rate each was saved at |
| `grads.py` | Exact per-example LoRA gradients from **one** batched backward pass, via forward and grad hooks |
| `projection.py` | Per-tensor Johnson-Lindenstrauss projection, so gradients fit in memory |
| `adam.py` | Read `optimizer.pt` and turn raw gradients into Adam update directions |
| `target_set.py` | Build the probe set |

`grads.py` is the subtle one. Per-example gradients normally mean one backward pass per example,
which is unaffordable. It hooks the LoRA modules and reconstructs each example's gradient from the
batched pass instead.

`adam.py` exists because TracIn's derivation assumes SGD, while the model was actually trained with
Adam. Without the correction the inner product is between quantities that are not the steps the
optimiser took. It has to reproduce the exact parameter ordering HF Trainer hands the optimiser,
which is why `_trainable_in_optimizer_order` exists.

## Three decisions to preserve

**Probe disjointness.** The probe set must not include the benchmark the result is reported on.
Reporting a gain on a task the probe was drawn from is circular. Our `dsir` scorer already enforces
the equivalent rule in `resolve_target_split`, and TracIn should reuse that check rather than invent
a second one. Note that PubMedQA is the trap here: it has a single `train` split and its evaluation
slice is `train[:500]`.

**Gradient norm bias.** A raw dot product favours high-loss documents, which on PubMed abstracts
means short, noisy or off-distribution ones. Normalising the gradients turns the score into a cosine
and removes most of it. Report the correlation between score and document length either way, so the
bias is measured rather than assumed absent.

**Adam correction is not optional if the numbers are going in a write-up.** It is a real difference
in what is being measured, not a refinement.

## What this repo is missing

1. **A checkpoint trajectory.** Our CPT stage saves checkpoints but does not record the learning
   rate at each save, so the `w_t` weights cannot be recovered afterwards. Recording it during
   training is a small change and should happen before anyone needs it.
2. **A fit phase.** `Scorer.score(records)` receives everything in one call and holds no state
   between calls. That is fine for `dsir` and `perplexity`, which fit in the same call, but TracIn
   has to load a model per checkpoint and that loop must sit outside any per-batch loop. The
   original solves this with `fit(ctx)` plus `score_pool(ctx)` and a `SelectionContext` carrying
   cache paths.
3. **A gradient cache.** Extraction is the expensive step and must survive a crash. The original
   caches projected gradients keyed on checkpoint and shard.

## Suggested order

1. Record the learning rate in the CPT stage's checkpoint metadata. Independently useful, small.
2. Add an optional `fit(records)` hook to `Scorer`, defaulting to a no-op. Existing scorers are
   unaffected; TracIn gets somewhere to put the checkpoint loop.
3. Port `projection.py` and `adam.py` first. Both are self-contained and unit testable against
   known matrices without any model, so they are the cheap half.
4. Port `grads.py` against a tiny LoRA model, and test it the only way that means anything: compare
   the hook-derived per-example gradients against a loop of single-example backward passes. If
   those do not match to tolerance, nothing downstream is worth running.
5. Only then wire up the scorer.

## Cost, which is the real blocker

The original notes gradient extraction taking tens of hours. That is on the same Quadro P5000 this
project uses, where everything runs in fp32 because Pascal has no tensor cores. Budget for it as a
multi-day job, not an afternoon, and check `nvidia-smi` first: the machine is shared and an `ollama`
service holds about 5.5 GB of the 16 GB permanently.

Before committing that time, note the cheaper result from
[research_update_2026-08.md](research_update_2026-08.md): a published critique finds selection may
not beat random at all, and the reversal is budget conditional. Running `random`, `length`, `dsir`
and `perplexity` at a matched token budget first tells you how much room there is for a gradient
method to win. If the cheap scorers already beat random by a wide margin, TracIn has something to
improve on. If they do not, that is the more interesting finding and it cost days less to get.

---

## A limitation this repo has today

Selection is **in memory**: `medsel select` loads the records, scores them all in one call, and
applies the budget. That is correct for the pool sizes used so far and it keeps the interface small,
but it will not hold the full 23.9M document PubMed corpus. The original streams parquet shards for
exactly this reason.

Anyone scaling past what fits in RAM should add streaming rather than raise the machine's memory,
and that change is a precondition for TracIn at full pool size, not a separate nicety.
