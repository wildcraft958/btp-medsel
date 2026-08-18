# Status and Open Items

Where the project stands at the end of stage 1, what to look at in review, and what is
deliberately still open. Read this before the team review; it is the map to everything else.

Last updated 2026-08-19.

**Newest document:** [research_update_2026-08.md](research_update_2026-08.md) is an adversarial
verification audit of the July research pass. It corrects three access-terms errors, records five
misreadings to avoid, and adds four references. Read it before citing the 2025-2026 work.

**Newest result:** the project now has GPU access, and both training stages plus the evaluator have
been run on it. There is a committed base-model control in [../results/](../results/). See
[Running on the lab GPU](#running-on-the-lab-gpu) for the two bugs that only appeared on real
hardware.

---

## Stage 1 is complete and ready for review

| Component | State |
|---|---|
| Core schema, registry, parquet cache, disk guard | done |
| MedQA / MedMCQA / PubMedQA / PubMed loaders | done |
| Sequence packing | done |
| CPT training stage | done, verified end to end on CPU and GPU |
| SFT training stage | done, verified end to end on CPU and GPU |
| MCQ evaluator with per-subject breakdown | done |
| Selection interface, top-k and stratified selectors | done |
| Scorers: `random`, `length`, `dsir`, `perplexity` | done, all four verified on GPU |
| `medsel select` command writing a reproducible manifest | done |
| TracIn scorer | not ported, spec in [tracin_port.md](tracin_port.md) |
| PubMed to PubMedQA contamination filter | done, opt-in via `exclude_pmids` |
| Literature review and dataset reference | done, refreshed with 2025-2026 work |
| Preference optimisation (`stages/align.py`) | stub, see open items |
| lm-evaluation-harness adapter (`eval/harness_adapter.py`) | stub, see open items |

The test suite is fully offline: `uv run pytest` needs no network, no GPU and no credentials.

**Pipeline verification evidence.** Both training stages have been run end to end on CPU and again
on the lab GPU. The checkpoints themselves were deleted afterwards (they are gitignored, 2 GB each,
and teach the model nothing), but they are reproducible in minutes:

| Run | Config | Result | GPU time |
|---|---|---|---|
| CPT | `configs/experiment/smoke_cpu.yaml` | 522 blocks packed from 500 documents, 20 steps, loss 2.954 to 2.833 | 24.6 s |
| SFT | `configs/experiment/sft_smoke_cpu.yaml` | 200 MedMCQA examples, 20 steps, train_loss 0.9493 | 25.3 s |
| Eval | `Qwen/Qwen3-0.6B-Base`, three tasks | base-model control, see [../results/](../results/) | ~25 min |
| Select | `dsir` on 300 PubMed docs, target MedMCQA | pool mean 0.086 against selected mean 3.296 | seconds |
| Select | `perplexity mode=mid` on 300 PubMed docs | selected scores cluster at the pool median, as the mode intends | seconds |
| Select | `length` on 600 MedMCQA, stratified by subject | all 21 subjects retained at `min_per_group=1`, none dropped | seconds |

Neither loss means anything about model quality. They exist to prove the path runs. Both smoke
configs are named `_cpu` but are device agnostic: they picked up the GPU with no edit, which is what
`dtype: auto` and the portable-by-default design were for.

The CPT run packed the same 522 blocks and the SFT run reported the same `train_loss` of 0.9493 on
both devices, which is the reproducibility check worth having.

---

## What to review

Suggested reading order for someone seeing this repo for the first time:

1. [README.md](../README.md) for the five-minute quickstart.
2. [architecture.md](architecture.md) for why the modules are split the way they are.
3. [datasets.md](datasets.md) for the six traps that produce plausible-but-wrong numbers. This is
   the highest-value document in the repo for avoiding a wasted experiment.
4. [literature_review.md](literature_review.md), or the
   [PDF](medsel_literature_review_2026-07.pdf) if you are not cloning the repo.
5. [contributing.md](contributing.md) for how to add a loader or a scorer without touching a
   central file.
6. [tracin_port.md](tracin_port.md) if you are picking up selection work.

**Sections 6.1 to 6.4 of the literature review are owned by their dataset owners** (Debmalya,
Animesh, Arkajyoti, Srinjoy) and were deliberately left for each owner to extend.

---

## Running on the lab GPU

The GPU is a **Quadro P5000, 16 GB, compute capability 6.1**. Pascal, from 2016, and its age is not
a detail. It has no tensor cores, an `ollama` service holds about 5.5 GB of its VRAM
persistently, and the machine has no sudo and no git.

**The working stack, established by running it rather than by reading pins:**

| Package | Version | Why this one |
|---|---|---|
| torch | 2.6.0+cu126 | The cu128 wheels dropped Pascal. This is the newest build that runs here. |
| transformers | 5.15.0 | Coexists with torch 2.6 despite the newer torch it normally ships beside. |
| trl / peft / datasets | 1.10.0 / 0.20.0 / 5.0.1 | No conflicts. |

### Two bugs that only appeared on real hardware

Both had passed the full offline suite and both were silent rather than loud, which is the argument
for running the pipeline on the target machine before trusting any result from it.

**`torch.cuda.is_bf16_supported()` returns `True` on this card.** It counts an emulation path, so
`dtype: auto` was selecting bf16 on hardware with no bf16 in it. The dtype rule now gates on compute
capability: bf16 from Ampere (8.0), fp16 back to Volta (7.0) where half precision still has tensor
cores behind it, fp32 below that. On this GPU `auto` correctly resolves to **fp32**, which is also
the conclusion the parallel project on the same box reached independently.

**Both training stages set `fp16=True` whenever bf16 was unavailable**, duplicating a precision
decision that belongs in one place. On Pascal that selects fp16 on a card where fp16 throughput is a
fraction of fp32. Stages now call `autocast_flags` next to `resolve_dtype`, so a stage cannot
contradict the dtype rule.

Separately, `transformers` 5.15 removed `TrainingArguments.warmup_ratio`, which crashed the stage on
construction. `warmup_steps` takes a float as a ratio and replaces it. On 5.14 the old argument was
still accepted but `get_warmup_steps` never read it, so **configured warmup had been silently doing
nothing** there too.

### Deploying

The box is a run target, not a development machine: one-way `rsync`, no git, and `DEPLOYED_COMMIT`
in the deploy directory records which commit is running. When rsyncing, **anchor the excludes**.
A bare `--exclude='data'` matches at every level and silently drops `configs/data/` and the whole
`src/medsel/data/` loader package. This is the same trap the `.gitignore` comment warns about.

---

## Open items

### 1. Two citations need a human read before the gap analysis is defended

`[He26]` (arXiv:2606.04466) and `[Ya26]` (arXiv:2507.15640) are the two papers that narrowed Gap 1
and Gap 2 in Section 8. **These reshape the project's novelty claim, so read them before
presenting it.**

**Partly discharged, 2026-08-18.** The August verification pass had three independent adversarial
readers fetch each primary source and check every number against the paper body. Both confirmed
3-0, so the gap analysis is no longer resting on an unread source. See
[research_update_2026-08.md](research_update_2026-08.md) section 1. A human read is still worth
doing before the defence, but it is now a confirmation rather than a risk.

### 2. Non-MCQ benchmark integration

**Access terms corrected 2026-08-18.** HealthBench is MIT, not CC BY-4.0, and BRIDGE ships only 55
of its 87 tasks openly. Also, the Medmarks "saturated" line does not apply at this project's model
scale, so MedQA, MedMCQA and PubMedQA remain usable. Details in
[research_update_2026-08.md](research_update_2026-08.md) section 2.

The proposal claims six clinical capabilities; MedQA, MedMCQA and PubMedQA measure roughly one and
a half. Section 6.5 of the review names four open, non-credentialed candidates with access terms
(HealthBench, BRIDGE, ACI-Bench, Medmarks), so the decision input exists and the work is unblocked.

Not started because none of the four matches `QAExample`, so this needs a fifth record shape, which
is a design decision the team should make together rather than one person picking.

### 3. Selection work still open

`dsir` and `perplexity` are ported and working. What remains:

- **TracIn**, the gradient-based method. Full spec in [tracin_port.md](tracin_port.md), including
  the three design decisions worth preserving and the reason to run the cheap baselines first.
- **`embed_similarity`**, cosine to the target-set centroid. Uses base-model hidden states, so it
  needs no new dependency. The cheapest remaining port.
- **Streaming selection.** `medsel select` holds the pool in memory. Correct at current sizes,
  will not hold 23.9M PubMed documents, and is a precondition for TracIn at full scale.

### 4. Three selection and evaluation improvements the 2026 literature now supports

None are implemented. All are proposals, listed highest value first:

- **`medterm_density` scorer.** Huang et al. (arXiv:2606.22079) found a medical-term density filter
  beats a generic educational-quality classifier at matched compute on medical text. Cheap,
  corpus-scale, CPU-friendly, and it gives the CPT stage a real baseline to beat instead of only
  `random` and `length`. Highest value per unit of effort.
- **Compute-aware selection budget.** BETR (arXiv:2507.12466) fits an optimal keep-rate that scales
  with training compute, roughly 3% at 1e20 FLOPs rising to 30% at 1e23. A fixed keep-rate is a
  scale-specific artefact, and a small-compute project should filter harder than published
  large-scale recipes suggest. Would extend `resolve_budget` in `selection/selector.py`.
- **Cross-capability variance in `EvalReport`.** Data Mixing Agent (arXiv:2507.15640) uses variance
  across fields as a retention metric. MedMCQA subject labels are already parsed, so the input
  exists. This is the closest thing to a direct measurement of the proposal's capability-retention
  claim.

### 5. Alignment stage

`stages/align.py` is a stub. The blocker is data, not training code. A 2026 survey found no open
*medical* preference corpus, so pairs must be constructed from PubMedQA long answers or MedMCQA
explanations. Five usable general-domain corpora exist and could serve as a replay mixture; see
Section 5 of the review, including the finding that only 70 to 80 percent of pairs in those corpora
agree with a reward-model ordering, so their labels are not ground truth.

### 6. lm-evaluation-harness adapter

`eval/harness_adapter.py` is a stub with the intended design recorded. Two evaluators exist on
purpose: `eval/mcq.py` is the inner loop, sharing our loaders and prompts so a change to either
shows up immediately, and it is what produces the per-subject capability breakdown. The harness is
the outer loop, for the paper, because its numbers are directly comparable to published MedQA,
MedMCQA and PubMedQA figures under the same prompts and few-shot conventions everyone else reports.

Our current numbers are internally consistent but should **not** be quoted against published
figures. Building this is what makes that comparison honest, and it matters before any result goes
into a write-up.

### 7. Standing experimental practice, not yet done

- **Run the base-model control** for every benchmark before any CPT run and commit it as a
  reference result. Per Jeong et al. and its 2026 follow-ups, a CPT result is a delta and a delta
  needs a control run, not a published number. **Done for `Qwen/Qwen3-0.6B-Base`**, committed in
  [../results/](../results/): MedQA 0.3802 on 1273 items, MedMCQA 0.4117 on 4183, PubMedQA 0.5240
  on 500, all well clear of chance. Still owed for any other base checkpoint the project trains,
  and the control has to be rerun whenever the prompt template version changes.
- **Add bootstrap confidence intervals** to `EvalReport`. MedQA's test split is 1,273 items; a
  1.5-point difference there is inside noise.

---

## Things that are easy to get wrong

Each of these is enforced in code, but they are worth knowing before you write an experiment.

- **MedMCQA `test` has no labels** (`cop = -1`). Evaluate on `validation`. The loader marks the
  split unlabeled and the evaluator raises rather than silently scoring 0%.
- **MedQA calls its validation split `dev`**, not `validation`.
- **PubMedQA labelling depends on the config, not the split.** All three configs use the split name
  `train`, and `pqa_unlabeled` omits the `final_decision` column entirely rather than nulling it.
- **The contamination filter is opt-in.** Set `exclude_pmids: pubmedqa` in the `loader:` block of
  any CPT experiment whose model will be scored on PubMedQA. `cpt_baseline.yaml` already does.
  Without it, that PubMedQA number is not a measurement.
- **Never commit data.** `data/` and `runs/` are gitignored. A single training run writes gigabytes.
- **Do not ask torch whether bf16 works.** `torch.cuda.is_bf16_supported()` returns `True` on the
  lab P5000, which has no bf16 hardware. Go through `resolve_dtype` and `autocast_flags`, never the
  torch flag directly, and never let a stage decide precision for itself.
- **`accuracy_norm` equalling `accuracy` is not a bug in `letter` mode.** Every continuation is one
  option letter, so there is nothing for length normalisation to normalise. The two separate only
  in `text` mode.

Full list with the underlying evidence in [datasets.md](datasets.md).
