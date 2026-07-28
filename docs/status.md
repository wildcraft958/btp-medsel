# Status and Open Items

Where the project stands at the end of stage 1, what to look at in review, and what is
deliberately still open. Read this before the team review; it is the map to everything else.

Last updated 2026-07-29.

---

## Stage 1 is complete and ready for review

| Component | State |
|---|---|
| Core schema, registry, parquet cache, disk guard | done |
| MedQA / MedMCQA / PubMedQA / PubMed loaders | done |
| Sequence packing | done |
| CPT training stage | done, verified end to end on CPU |
| SFT training stage | done, verified end to end on CPU |
| MCQ evaluator with per-subject breakdown | done |
| Selection interface, `random` and `length` scorers, top-k and stratified selectors | done |
| PubMed to PubMedQA contamination filter | done, opt-in via `exclude_pmids` |
| Literature review and dataset reference | done, refreshed with 2025-2026 work |
| Preference optimisation (`stages/align.py`) | stub, see open items |
| lm-evaluation-harness adapter (`eval/harness_adapter.py`) | stub, see open items |

348 tests, all offline. `uv run pytest` needs no network and no GPU.

**Pipeline verification evidence.** Both training stages were run end to end on CPU before this
review. The checkpoints themselves were deleted afterwards (they are gitignored, 2 GB each, and
teach the model nothing), but they are reproducible in minutes:

| Run | Config | Result |
|---|---|---|
| CPT | `configs/experiment/smoke_cpu.yaml` | 522 blocks packed from 500 documents, 20 steps, loss 2.904 to 2.833 |
| SFT | `configs/experiment/sft_smoke_cpu.yaml` | 200 MedMCQA examples, 20 steps, train_loss 0.9493 |

Neither number means anything about model quality. They exist to prove the path runs.

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

**Sections 6.1 to 6.4 of the literature review are owned by their dataset owners** (Debmalya,
Animesh, Arkajyoti, Srinjoy) and were deliberately left for each owner to extend.

---

## Open items

### 1. Two citations need a human read before the gap analysis is defended

`[He26]` (arXiv:2606.04466) and `[Ya26]` (arXiv:2507.15640) are the two papers that narrowed Gap 1
and Gap 2 in Section 8. Their findings are recorded in the review, but nobody on the team has read
either primary source in full. **These reshape the project's novelty claim, so read them before
presenting it.** If either says something other than what Section 8 reports, the gap analysis needs
revising, not the citation.

### 2. Non-MCQ benchmark integration

The proposal claims six clinical capabilities; MedQA, MedMCQA and PubMedQA measure roughly one and
a half. Section 6.5 of the review names four open, non-credentialed candidates with access terms
(HealthBench, BRIDGE, ACI-Bench, Medmarks), so the decision input exists and the work is unblocked.

Not started because none of the four matches `QAExample`, so this needs a fifth record shape, which
is a design decision the team should make together rather than one person picking.

### 3. Three selection and evaluation improvements the 2026 literature now supports

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

### 4. Alignment stage

`stages/align.py` is a stub. The blocker is data, not training code. A 2026 survey found no open
*medical* preference corpus, so pairs must be constructed from PubMedQA long answers or MedMCQA
explanations. Five usable general-domain corpora exist and could serve as a replay mixture; see
Section 5 of the review, including the finding that only 70 to 80 percent of pairs in those corpora
agree with a reward-model ordering, so their labels are not ground truth.

### 5. lm-evaluation-harness adapter

`eval/harness_adapter.py` is a stub with the intended design recorded. Two evaluators exist on
purpose: `eval/mcq.py` is the inner loop, sharing our loaders and prompts so a change to either
shows up immediately, and it is what produces the per-subject capability breakdown. The harness is
the outer loop, for the paper, because its numbers are directly comparable to published MedQA,
MedMCQA and PubMedQA figures under the same prompts and few-shot conventions everyone else reports.

Our current numbers are internally consistent but should **not** be quoted against published
figures. Building this is what makes that comparison honest, and it matters before any result goes
into a write-up.

### 6. Standing experimental practice, not yet done

- **Run the base-model control** for every benchmark before any CPT run and commit it as a
  reference result. Per Jeong et al. and its 2026 follow-ups, a CPT result is a delta and a delta
  needs a control run, not a published number.
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

Full list with the underlying evidence in [datasets.md](datasets.md).
