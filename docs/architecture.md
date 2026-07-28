# Architecture

What the pieces are, why they are separate, and where new work attaches.

## The shape

```
   MedQA ─┐
 MedMCQA ─┼─→ QAExample ─┐
PubMedQA ─┘              │
                         ├─→ selection ─→ stages ─→ eval
  PubMed ──→ CorpusDoc ──┘   (score,       (cpt,     (mcq,
                              select)       sft,      per-subject)
                                            align)
```

Four sources, two record types, one path through the pipeline.

## Why normalise

The three QA datasets ship mutually incompatible raw shapes: MedQA nests under `data.{Question,
Options}`, MedMCQA is flat `opa..opd` with a 0-based `cop` index, PubMedQA is a three-way decision
with abstracts attached. Without a normalisation layer, every scorer, prompt renderer, and
evaluator would branch on which dataset it was handed, and each branch would be a place for a
dataset-specific trap (§ [datasets.md](datasets.md)) to hide.

`QAExample` and `CorpusDoc` (`src/medsel/schema.py`) are frozen dataclasses that validate on
construction, so a malformed record fails where it is built rather than three stages later.

The one thing the schema refuses to paper over is **missing labels**. `answer_key` is `None` when
a source withholds gold answers, and consumers must check `is_labeled`. Filling in a placeholder
would make MedMCQA's `test` split look scoreable.

## Registries, not dispatch tables

Both extension points auto-discover:

- **Loaders** — a module in `src/medsel/data/` with `@register_loader("name")`
- **Scorers** — a module in `src/medsel/selection/scorers/` with `@register_scorer("name")`

Neither requires editing a central file. That is deliberate: four people are adding loaders in
parallel, and the selection scorers are the research contribution, so both should be
add-a-file operations rather than merge-conflict operations.

Stages are the exception — `STAGES` in `src/medsel/stages/__init__.py` is a plain dict, because the
three-stage pipeline is fixed by the proposal and is not meant to grow.

## Separating scoring from selection

`selection/base.py` scores records; `selection/selector.py` assembles subsets under a budget.

They are apart because the proposal's capability-aware constraints — keeping rare diseases and
underrepresented specialties represented — are a property of *how a subset is assembled*, not of
any per-item score. A global top-k drops rare groups by construction, since rare things score low
almost by definition. `select_stratified(min_per_group=…)` is where that constraint lives.

## Separating `prepare()` from `run()`

`Stage.prepare()` assembles training data and touches no model. `Stage.run()` trains.

Data assembly is the interesting, expensive part of this project, and needs to be inspectable on
its own — you should be able to examine what a selection policy chose without loading a 4B model.
It also means configuration errors surface in milliseconds: `CPTStage.prepare()` rejects a QA
source before a tokenizer is downloaded.

## Caching and provenance

`data/cache/{source}/{split}.parquet` plus a manifest recording the loader's cache key, the record
dataclass' field-signature hash, and the row count. Any drift invalidates the entry, so a stale
parquet cannot be served as current data after a normaliser changes. `normalizer_version` on the
loader is the manual lever for that.

Every training run writes `run.json` beside its checkpoints: config, hardware, metrics, versions.
A checkpoint whose provenance is unrecorded is not a result anyone can defend later.

## Configuration

Experiments are YAML (`configs/experiment/*.yaml`), not command-line flags, so a run is a
reviewable artefact that can be diffed and re-run. Sections may be inline or a path to a shared
fragment (`model: ../model/gemma3_1b.yaml`).

**Unknown keys raise.** A silently ignored `learning_rare` typo produces a run that looks correct
and trains at the wrong rate — far more expensive than a startup failure.

## Portability

Compute for this project is unsettled: a CPU laptop now, a lab GPU or Colab later. So:

- `dtype: auto` resolves against actual hardware (`utils/device.py`) — bf16 on capable GPUs, fp16
  on older ones, fp32 on CPU, since fp16 on CPU is slow and numerically fragile.
- `configs/experiment/smoke_cpu.yaml` runs the entire CPT path on a laptop in minutes using an
  **ungated** 135M model, so a fresh clone can validate the pipeline without licence acceptance.
- The PubMed loader streams and takes a shard budget, with a free-space guard sized from real
  remote file sizes.

## Evaluation

Options are scored by continuation log-likelihood in a single forward pass per example. Nothing is
generated, so results do not depend on decoding settings, and **base models are evaluable** —
which is the comparison this project needs, base versus post-CPT.

Two scoring modes exist because base models are badly served by letter-scoring: an
instruction-untuned model has no reason to know that "A" is a valid utterance. `text` mode scores
the option wording instead. Reporting one without the other invites exactly the prompt-fit artefact
that Jeong et al. identify (§ [literature_review.md §3.5](literature_review.md#35-the-contrarian-result-and-why-it-matters-most)).

Per-subject accuracy comes from MedMCQA's subject labels — the only per-capability signal any of
our datasets ships, and the closest thing we have to the proposal's capability-retention metric.

## Dependencies

Core (`datasets`, `huggingface-hub`, `pyarrow`, `pyyaml`, `numpy`, `tqdm`) is enough to load,
inspect and cache every dataset. `torch`/`transformers`/`peft`/`trl` live in the `train` extra and
are imported lazily inside functions, so `medsel data stats` works on a machine with no ML stack
installed.

Written against `transformers` 5.x (`dtype` not `torch_dtype`, `processing_class` not `tokenizer`).

## Where to add things

| To add… | Do this |
|---|---|
| A dataset | New module in `src/medsel/data/`, subclass `BaseLoader`, `@register_loader` |
| A selection method | New module in `src/medsel/selection/scorers/`, subclass `Scorer`, `@register_scorer` |
| An experiment | New YAML in `configs/experiment/` |
| A training stage | Implement `prepare`/`run` in the existing `stages/sft.py` or `stages/align.py` |
| A metric | Extend `EvalReport` in `src/medsel/eval/mcq.py` |
