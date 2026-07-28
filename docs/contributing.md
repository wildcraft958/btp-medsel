# Contributing

## Ownership

| Area | Owner | Code | Lit review section |
|---|---|---|---|
| MedQA | Debmalya | `src/medsel/data/medqa.py` | §6.1 |
| MedMCQA | Animesh | `src/medsel/data/medmcqa.py` | §6.2 |
| PubMedQA | Arkajyoti | `src/medsel/data/pubmedqa.py` | §6.3 |
| PubMed / CPT corpus | Srinjoy | `src/medsel/data/pubmed.py` | §6.4 |
| Shared core, stages, eval | everyone | `schema.py`, `stages/`, `eval/` | §3, §7 |

All four loaders are already implemented and tested, so nobody is blocked waiting on anyone else.
Owning an area means extending it, verifying it against the source, and writing its literature
review section — not writing it from scratch.

## Setup

```bash
git clone git@github.com:wildcraft958/btp-medsel.git && cd btp-medsel
uv sync --extra dev
uv run pytest            # offline, no network
uv run medsel info
```

To train or evaluate, add the `train` extra:

```bash
uv sync --extra dev --extra train
```

**Laptops without an NVIDIA GPU:** install CPU-only torch instead, or `uv` will pull ~3 GB of CUDA
wheels you cannot use.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

That index is deliberately *not* in `pyproject.toml` — most of us will use GPUs, and pinning CPU
builds for everyone would be worse.

## Before you push

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv run medsel train --config configs/experiment/smoke_cpu.yaml   # a few minutes on CPU
```

The smoke run exercises the whole CPT path — shard fetch, filtering, packing, training,
checkpointing. It exists to fail loudly when someone breaks the pipeline. Its loss curve means
nothing; do not read it.

## Adding a dataset

One new module, one config. No central file to edit.

```python
# src/medsel/data/mynewset.py
from typing import Any, ClassVar

from medsel.data.base import BaseLoader
from medsel.registry import register_loader
from medsel.schema import QAExample, make_uid


@register_loader("mynewset")
class MyNewSetLoader(BaseLoader):
    hf_id = "org/dataset"
    splits: ClassVar[tuple[str, ...]] = ("train", "test")
    unlabeled_splits: ClassVar[frozenset[str]] = frozenset()

    def normalize(self, row: dict[str, Any], idx: int, split: str) -> QAExample:
        return QAExample(
            uid=make_uid(self.name, split, idx),
            source=self.name,
            split=split,
            question=row["question"],
            options={"A": row["a"], "B": row["b"]},
            answer_key=row["answer"],
        )
```

Then capture a few real rows as a fixture and test `normalize` against them offline:

```bash
uv run python -c "
from datasets import load_dataset; import json
ds = load_dataset('org/dataset', split='train', streaming=True)
rows = [r for i, r in enumerate(ds) if i < 5]
open('tests/fixtures/mynewset_train.jsonl','w').writelines(
    json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
"
```

Bump `normalizer_version` whenever `normalize` changes shape — the parquet cache keys on it and
will rebuild rather than serve stale records.

## Adding a selection method

Same pattern, in `src/medsel/selection/scorers/`:

```python
from medsel.selection.base import Scorer, register_scorer


@register_scorer("less")
class LessScorer(Scorer):
    def score(self, records):
        return [...]  # one float per record; higher = keep
```

Scores are ranked, never thresholded on absolute value, so any monotone scale is fine. Beat
`random` at an equal token budget or the method has not demonstrated anything.

## House rules

**Verify against the data, not the docs.** Every dataset trap in
[datasets.md](datasets.md) was found by comparing actual values; several contradict what a dataset
card or API summary says. If a claim matters, check it and record how you checked.

**Encode traps in code.** A caveat in a markdown file gets forgotten. MedMCQA's unlabeled `test`
split raises rather than scoring; that is the standard to hold.

**Say when a number is uncertain.** The literature review marks figures taken from secondary
sources. Do the same rather than laundering a search result into a citation.

**Commit messages explain why.** The diff already shows what changed.

**Never commit data.** `data/` and `runs/` are gitignored. Corpora are regenerated from configs,
which is what makes runs reproducible.

## Gotchas worth knowing before you start

- MedMCQA `test` has no labels (`cop = -1`). Use `validation`.
- MedQA's validation split is called `dev`.
- PubMedQA labelling depends on **config**, not split.
- PubMed contaminates PubMedQA — the PMID exclusion filter is **not yet built**. Do not report
  PubMedQA numbers from a CPT'd model until it is.
- `transformers` 5.x renamed `torch_dtype` → `dtype` and `Trainer(tokenizer=)` →
  `processing_class=`.
- Gemma weights are licence-gated: `hf auth login` and accept the licence before using
  `configs/model/gemma3_1b.yaml`.

Full list in [datasets.md](datasets.md).
