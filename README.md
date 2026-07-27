# medsel

**Multi-stage, multi-target data selection harness for medical foundation models.**

BTP project. Medical LLMs are trained in stages — continual pretraining (CPT) → supervised
fine-tuning (SFT) → preference optimisation (DPO/RLHF). Existing data-selection methods optimise a
single stage against a single objective. This harness learns *stage-specific* selection policies
while preserving multiple clinical capabilities across the whole pipeline.

Current focus: **CPT**.

## Quickstart

```bash
git clone <repo-url> && cd BTP
uv sync --extra dev            # core + test deps
uv sync --extra dev --extra train   # add torch/transformers/trl/peft when you need to train

uv run pytest                  # offline, no network needed
uv run medsel data stats --source medmcqa
```

Everything long-running shows a `tqdm` progress bar — dataset normalisation, PubMed shard
download, token packing, and evaluation.

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

Raw schemas differ substantially between the three QA sets — see `docs/datasets.md` for the field
maps and the known traps (notably: **MedMCQA's `test` split has no labels**, so evaluation uses
`validation`).

## Layout

```
src/medsel/
  schema.py       QAExample, CorpusDoc
  registry.py     name -> loader, so adding a dataset touches no core file
  data/           per-source loaders, parquet cache, token packing
  stages/         cpt (implemented), sft / align (stubs)
  selection/      Scorer interface + baseline scorers
  eval/           MCQ log-prob evaluator, per-capability breakdown
configs/          data / model / experiment YAML
docs/             literature review, architecture, dataset notes, contributing
tests/            offline fixture-based tests
```

## Adding a dataset

One new file in `src/medsel/data/`, one config in `configs/data/`. Subclass `BaseLoader`,
implement `normalize()`, decorate with `@register_loader("yourname")`. No central file to edit.
See `docs/contributing.md`.

## Status

| Phase | State |
|---|---|
| Repo + packaging | done |
| Core schema, registry, cache, disk guard | pending |
| MedQA / MedMCQA / PubMedQA loaders | pending |
| PubMed CPT corpus loader | pending |
| CPT training stage | pending |
| MCQ evaluator | pending |
| SFT / alignment stubs | pending |
| Literature review + docs | pending |

## Notes on compute

`data/` and `runs/` are gitignored — regenerate locally, never commit corpora. The full
`MedRAG/pubmed` repo is ~279 GB, so the CPT loader streams and takes a configurable shard budget,
with a disk guard that refuses to write past your free space. A CPU-only smoke profile
(`configs/experiment/smoke_cpu.yaml`) runs the whole CPT path on a laptop for pipeline validation.
