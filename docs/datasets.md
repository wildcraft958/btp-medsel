# Dataset Reference

Field-level schemas and the traps. Every schema below was read off the Hub, not from a paper or
a README. Where our loader disagrees with a dataset card, this file records what the data
actually contains.

Provenance and "what is this dataset for" live in
[literature_review.md §6](literature_review.md#6-the-datasets).

---

## Summary

| Loader | HF id | Record | Splits | Eval split | Owner |
|---|---|---|---|---|---|
| `medqa` | `openlifescienceai/medqa` | `QAExample` | train, dev, test | `test` | Debmalya |
| `medmcqa` | `openlifescienceai/medmcqa` | `QAExample` | train, validation, test | **`validation`** | Animesh |
| `pubmedqa` | `qiaojin/PubMedQA` | `QAExample` | train (per config) | `train[:500]` | Arkajyoti |
| `pubmed` | `MedRAG/pubmed` | `CorpusDoc` | train | n/a (CPT) | Srinjoy |

```python
from medsel.registry import get_loader

for ex in get_loader("medmcqa").load("validation", limit=10):
    print(ex.question, ex.options, ex.answer_key)
```

---

## The traps

These are the ones that produce *plausible but wrong* numbers rather than crashes. Each is
enforced in code, not just documented here.

### 1. MedMCQA's `test` split has no labels

Every row ships `cop = -1`. The publisher withholds gold answers.

Scoring against `-1` silently yields ~0% accuracy, which looks like a broken model rather than a
broken evaluation. All published MedMCQA numbers use `validation` (the NEET-PG set).

*Enforced:* `MedMCQALoader.unlabeled_splits = {"test"}`; `answer_key` normalises to `None`;
`evaluate_task` raises if you ask for `test`.

### 2. PubMedQA's labelling depends on the config, not the split

`pqa_unlabeled` omits the `final_decision` **column entirely**. It is not null, it is absent. All
three configs use the split name `train`, so split-based label logic gets this wrong.

*Enforced:* `PubMedQALoader.has_labels()` keys on `self.config`.

### 3. PubMed contaminates PubMedQA

PubMedQA is constructed from PubMed abstracts, and our CPT corpus is PubMed abstracts. Continually
pretraining on PubMed and then evaluating on PubMedQA can mean evaluating on text the model was
just trained on.

*Not yet enforced.* This needs a PMID exclusion list built from `pqa_labeled` and wired into the
PubMed loader as a filter. **Until it exists, treat any PubMedQA number from a CPT'd model as
unreliable.** Tracked in [literature_review.md §9](literature_review.md#9-implications-for-our-harness).

### 4. MedQA calls its validation split `dev`

MedMCQA uses `validation`, MedQA uses `dev`. Code that assumes one name breaks silently on the
other by falling back to `train`.

*Enforced:* `check_split` raises with the available names.

### 5. `MedRAG/pubmed` stores its text twice

`contents` is exactly `title + " " + content`. It is not a duplicate of `content` alone. An
earlier assumption here was wrong, and reading both roughly doubles memory and disk for no new
information.

*Enforced:* the loader reads `title` and `content` only; `CorpusDoc.full_text` recomputes the join
(with `\n\n`, which suits CPT better than a single space).

### 6. The Hub's storage figure is not the download size

`MedRAG/pubmed` reports ~279 GB of `usedStorage`, which counts LFS history. The actual sum of file
sizes is **70 GB**. Sizing a disk budget from the former needlessly rules the corpus out.

*Enforced:* `remote_size_bytes()` sums `repo_info(files_metadata=True)` sibling sizes.

---

## Raw schemas

### MedQA: `openlifescienceai/medqa`

Everything nests under `data`, with capitalised, space-separated keys.

```json
{
  "id": "82660913-bdf0-493d-ac69-158c5514142e",
  "data": {
    "Question": "A 23-year-old pregnant woman at 22 weeks gestation presents with...",
    "Options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
    "Correct Option": "D",
    "Correct Answer": "Nitrofurantoin"
  },
  "subject_name": ""
}
```

| Raw | `QAExample` | Note |
|---|---|---|
| `data.Question` | `question` | |
| `data.Options` | `options` | already a letter→text mapping |
| `data.Correct Option` | `answer_key` | `None` if not a valid option key |
| `data.Correct Answer` | `answer_text` | |
| `id` | `labels.source_id` | |
| `subject_name` | *dropped* | empty for every row |

**`subject_name` is `""` on every row**, so MedQA contributes nothing to per-capability analysis.
A blank label is worse than no label, so it is dropped rather than propagated.

### MedMCQA: `openlifescienceai/medmcqa`

Flat columns, 0-based answer index.

```json
{
  "id": "e9ad821a-c438-4965-9f77-760819dfa155",
  "question": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to...",
  "opa": "Hyperplasia", "opb": "Hyperophy", "opc": "Atrophy", "opd": "Dyplasia",
  "cop": 2,
  "choice_type": "single",
  "exp": "Chronic urethral obstruction because of urinary calculi...",
  "subject_name": "Anatomy",
  "topic_name": "Urinary tract"
}
```

| Raw | `QAExample` | Note |
|---|---|---|
| `question` | `question` | |
| `opa`..`opd` | `options` `{A,B,C,D}` | `None` becomes `""`, never the string `"None"` |
| `cop` | `answer_key` | `0→A`; **`-1` → `None`** |
| `exp` | `rationale` | **null on many rows**; blank becomes `None` |
| `subject_name`, `topic_name`, `choice_type` | `labels` | 21 subjects, our only capability signal |

Quality note: MedMCQA is scraped from exam material and contains some typos, ambiguous items, and
occasionally disputed keys. Do not over-interpret small accuracy differences.

### PubMedQA: `qiaojin/PubMedQA`

Three configs, each a single `train` split.

| Config | Rows | Labels | Use |
|---|---|---|---|
| `pqa_labeled` | 1,000 | expert | evaluation (default) |
| `pqa_artificial` | ~211,000 | auto-generated | SFT-scale training |
| `pqa_unlabeled` | ~61,000 | **none** | long answers only |

```json
{
  "pubid": 21645374,
  "question": "Do mitochondria play a role in remodelling lace plant leaves...?",
  "context": {
    "contexts": ["Programmed cell death (PCD) is the regulated death of cells...", "..."],
    "labels": ["BACKGROUND", "RESULTS"],
    "meshes": ["Alismataceae", "Apoptosis", "Cell Differentiation", "Mitochondria"]
  },
  "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses...",
  "final_decision": "yes"
}
```

| Raw | `QAExample` | Note |
|---|---|---|
| `question` | `question` | |
| (none) | `options` | fixed `{A: yes, B: no, C: maybe}` |
| `final_decision` | `answer_key` | absent in `pqa_unlabeled` |
| `context.contexts` | `contexts` | the source abstract |
| `long_answer` | `rationale` | |
| `context.meshes` | `labels.meshes` | `"; "`-joined; drives rare-disease coverage |
| `context.labels` | `labels.sections` | BACKGROUND / METHODS / RESULTS |

**Comparability:** the canonical 500-question test set is defined by a ground-truth file in the
authors' repository that the Hub mirror does not carry. `pqa_labeled` arrives as one undivided
1,000-row split, so we slice `train[:500]` deterministically. Our numbers are internally
consistent but are **not** directly comparable to published PubMedQA results.

### PubMed corpus: `MedRAG/pubmed`

1,166 JSONL shards under `chunk/`, 23.9M snippets, ~296 tokens each.

```json
{
  "id": "pubmed23n0001_33",
  "title": "Constitution and properties of axonal membranes of crustacean nerves.",
  "content": "The purification of axonal membranes of crustaceans was followed by...",
  "contents": "Constitution and properties... The purification of axonal membranes...",
  "PMID": 57
}
```

| Raw | `CorpusDoc` | Note |
|---|---|---|
| `id` | `uid` (prefixed `pubmed/`) | |
| `title` | `title` | |
| `content` | `text` | the abstract |
| `contents` | *dropped* | `= title + " " + content`, redundant |
| `PMID` | `meta.PMID` | |

Measured: **70 GB total**, ~60 MB per shard (max 92 MB). The `pubmed23n` naming indicates a 2023
baseline snapshot; it is abstracts-only, and the inclusion filter (articles having both title and
abstract) is MedRAG's, not ours.

**Loader options**

| Option | Default | Purpose |
|---|---|---|
| `num_shards` | 8 | shard budget, ~480 MB / ~160k docs |
| `shard_offset` | 0 | split the corpus between collaborators without overlap |
| `min_chars` | 200 | drop title-only records that skew packing |
| `dedup` | `True` | drop repeat PMIDs; one int per doc in memory |
| `shard_dir` | `None` | read an already-fetched copy on a shared machine |

```bash
uv run medsel data stats --source pubmed --loader num_shards=1 --limit 1000
```

Downloads are preceded by a free-space check sized from the actual remote file sizes, and abort
rather than fill the filesystem.

---

## Regenerating test fixtures

`tests/fixtures/*.jsonl` are real Hub rows so the test suite runs offline. If a dataset changes
upstream, regenerate them and re-read the assertions in `tests/test_loaders.py`. Several encode
specific values (`cop = -1`, the absence of `final_decision`) that are the point of the fixture.
