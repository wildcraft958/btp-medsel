# medsel

**Multi-stage, multi-target data selection harness for medical foundation models.**

BTP project. Medical LLMs are trained in stages: continual pretraining (CPT) → supervised
fine-tuning (SFT) → preference optimisation (DPO/RLHF). Existing data-selection methods optimise a
single stage against a single objective. This harness learns *stage-specific* selection policies
while preserving multiple clinical capabilities across the whole pipeline.

Current focus: **CPT**.

## Quickstart

```bash
git clone git@github.com:wildcraft958/btp-medsel.git && cd btp-medsel
uv sync --extra dev                  # core + test deps
uv run pytest                        # fully offline, no network or GPU needed
uv run medsel info                   # hardware, loaders, stages

uv run medsel data list
uv run medsel data stats --source medmcqa --limit 500
uv run medsel data stats --source pubmed --loader num_shards=1 --limit 1000
```

To train and evaluate, add the `train` extra (see
[docs/contributing.md](docs/contributing.md) if your laptop has no NVIDIA GPU):

```bash
uv sync --extra dev --extra train

uv run medsel train --config configs/experiment/smoke_cpu.yaml    # whole CPT path, CPU, minutes
uv run medsel eval --model runs/smoke-cpu --tasks medmcqa --limit 100
```

Selecting a subset, which is what the project is actually about:

```bash
# Keep the 10% of PubMed that most resembles MedMCQA.
uv run medsel select --source pubmed --scorer dsir --budget 0.1 \
  --limit 300 --loader num_shards=1 --scorer-arg target=medmcqa

# Keep a spread across every MedMCQA subject rather than a global top-k.
uv run medsel select --source medmcqa --split train --scorer length --budget 0.1 \
  --limit 600 --stratify-by subject_name --min-per-group 1

# Same selection, from a pool too large to hold in memory.
uv run medsel select --source pubmed --scorer dsir --budget 5000 --stream \
  --loader num_shards=8 --scorer-arg target=medmcqa
```

Everything long-running shows a `tqdm` progress bar: dataset normalisation, PubMed shard
download, token packing, caching, and evaluation.

## Datasets

| Source | HF id | Stage | Owner |
|---|---|---|---|
| MedQA | `openlifescienceai/medqa` | SFT / eval | Debmalya |
| MedMCQA | `openlifescienceai/medmcqa` | SFT / eval | Animesh |
| PubMedQA | `qiaojin/PubMedQA` | SFT / eval | Arkajyoti |
| PubMed | `MedRAG/pubmed` | CPT | Srinjoy |

All four expose the same interface. The three QA sets normalise to `QAExample`, the PubMed corpus
to `CorpusDoc`, so no stage has to know which dataset it is consuming.

```python
from medsel.registry import get_loader

for ex in get_loader("medmcqa").load("validation", limit=100):
    print(ex.question, ex.options, ex.answer_key)
```

Raw schemas differ substantially between the three QA sets. See `docs/datasets.md` for the field
maps and the known traps (notably: **MedMCQA's `test` split has no labels**, so evaluation uses
`validation`).

## Layout

```
src/medsel/
  schema.py       QAExample, CorpusDoc
  registry.py     name -> loader, so adding a dataset touches no core file
  data/           per-source loaders, parquet cache, token packing
  stages/         cpt, sft (implemented), align (stub)
  selection/      Scorer interface, scorers, budget and stratified selectors
  eval/           MCQ log-prob evaluator, per-capability breakdown
configs/          data / model / experiment YAML
docs/             literature review, architecture, dataset notes, contributing
tests/            offline fixture-based tests
```

## Adding a dataset

One new file in `src/medsel/data/`, one config in `configs/data/`. Subclass `BaseLoader`,
implement `normalize()`, decorate with `@register_loader("yourname")`. No central file to edit.
See `docs/contributing.md`.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/status.md](docs/status.md) | **Start here for review.** What is done, what is open, and what is easy to get wrong |
| [docs/literature_review.md](docs/literature_review.md) | CPT, SFT, alignment, the four datasets, data-selection methods, gap analysis |
| [docs/datasets.md](docs/datasets.md) | Field-level schemas and the traps that produce plausible-but-wrong numbers |
| [docs/architecture.md](docs/architecture.md) | What the pieces are, why they're separate, where new work attaches |
| [docs/contributing.md](docs/contributing.md) | Setup, ownership, how to add a loader or scorer |
| [docs/tracin_port.md](docs/tracin_port.md) | Spec for the one selection method still to port, and why cheap baselines come first |
| [docs/medsel_literature_review_2026-07.pdf](docs/medsel_literature_review_2026-07.pdf) | The review plus the raw research findings, as one shareable 23-page PDF |
| [results/](results/) | Committed reference results, currently the `Qwen3-0.6B-Base` base-model control |

The PDF is the version to hand to someone who is not cloning the repo. Rebuild it after editing
the review with `uv pip install markdown && uv run python scripts/build_pdf.py` (needs Chrome).

## Status

| Component | State |
|---|---|
| Core schema, registry, parquet cache, disk guard | done |
| MedQA / MedMCQA / PubMedQA loaders | done |
| PubMed CPT corpus loader + packing | done |
| CPT training stage | done, verified end to end on CPU and GPU |
| MCQ evaluator with per-subject breakdown | done |
| Selection interface + `random`, `length`, `dsir`, `perplexity` scorers | done |
| `medsel select` with top-k and stratified budgets | done, verified on GPU |
| Streaming selection for pools larger than memory | done, `--stream` |
| Literature review + docs | done |
| SFT stage | done, verified end to end on CPU and GPU |
| Preference optimisation | stub, blocked on constructing medical preference pairs |
| Influence-based scorers (TracIn, LESS, 3DS) | not started, spec in [docs/tracin_port.md](docs/tracin_port.md) |
| PubMed→PubMedQA contamination filter | done, opt-in via `exclude_pmids` |

## Known caveats

- **PubMedQA contamination.** Our CPT corpus is PubMed abstracts and PubMedQA is built from PubMed
  abstracts. The PMID exclusion filter now exists but is opt-in: set `exclude_pmids: pubmedqa` in
  your experiment's `loader:` block. `cpt_baseline.yaml` already does. Without it, a PubMedQA
  number from a CPT'd model is not a measurement.
- **PubMedQA comparability.** The canonical 500-question test set is defined by a file the Hub
  mirror doesn't carry, so our PubMedQA numbers are internally consistent but not directly
  comparable to published figures.
- **Capability coverage.** The proposal names six clinical capabilities; the three QA datasets
  measure roughly one and a half of them. Either the evaluation suite grows or the claims narrow.

## Notes on compute

`data/` and `runs/` are gitignored: regenerate locally, never commit corpora. The full
`MedRAG/pubmed` corpus is **70 GB** across 1166 shards (~60 MB each), so the CPT loader streams,
takes a configurable shard budget, and runs a free-space check sized from real remote file sizes
before downloading anything. `shard_offset` lets collaborators split the corpus without overlap.

A CPU-only smoke profile (`configs/experiment/smoke_cpu.yaml`) runs the whole CPT path on a laptop
in minutes using an ungated 135M model, so a fresh clone can validate the pipeline without GPU
access or licence acceptance.
