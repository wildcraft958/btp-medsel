# Committed reference results

Evaluation reports that other results are measured against. Small JSON, committed on purpose:
a CPT or SFT result is a delta, and a delta needs a control produced by this repo under the same
prompts and the same scoring convention. A published number from a paper is not a substitute,
because prompt and few-shot conventions differ.

Training artifacts still never get committed. `runs/` and `data/` stay gitignored.

| File | What it is |
|---|---|
| `base_control_qwen3-0.6b-base.json` | `Qwen/Qwen3-0.6B-Base`, untrained, on the three labelled QA splits |
| `scorer_comparison.json` | All five scorers over one 5,000 document PubMed pool, at a matched budget |

## The current control

Run on the lab Quadro P5000 in fp32, `letter` scoring mode.

| Task | Split | n | accuracy | chance |
|---|---|---|---|---|
| MedQA | test | 1273 | 0.3802 | 0.20 |
| MedMCQA | validation | 4183 | 0.4117 | 0.25 |
| PubMedQA | train[:500] | 500 | 0.5240 | 0.33 |

`accuracy_norm` equals `accuracy` in all three. That is expected rather than a bug: in `letter`
mode every continuation is a single option letter, so length normalisation has nothing to
normalise. The two diverge only in `text` mode.

The PubMedQA split is `train[:500]`, not the canonical 500-question test set, which is defined by a
file the Hub mirror does not carry. That number is internally consistent and **not** comparable to
published PubMedQA figures. See the caveats in the top-level README.

## The scorer comparison

Five scorers, one pool of 5,000 PubMed abstracts (1,265,291 tokens), budget 500 records.
Regenerate with `uv run python scripts/compare_scorers.py --limit 5000 --budget 500`.

**A record budget is not a token budget.** Training consumes tokens, and the scorers disagree
sharply about how many they spend for the same record count:

| Scorer | Tokens at 500 records | Against random | Records at a matched 122,577 tokens |
|---|---|---|---|
| `length` | 252,183 | **2.06x** | 212 |
| `perplexity` (mid) | 130,398 | 1.06x | 469 |
| `random` | 122,577 | 1.00x | 500 |
| `embed_similarity` | 106,807 | 0.87x | 564 |
| `dsir` | 93,547 | 0.76x | 630 |

Comparing these at 500 records each would hand `length` twice the training data and call the result
a difference between selection methods. Any downstream comparison has to hold tokens fixed, and
report which budget it held.

**The scorers do select different documents.** Pairwise Jaccard overlap at the record budget:

| Pair | Overlap |
|---|---|
| `dsir` and `embed_similarity` | 0.1099 |
| `length` and `perplexity` | 0.0627 |
| everything else | 0.015 to 0.057 |

Near-zero overlap is expected when 500 records are drawn from 5,000: two independent choices would
share about 10%. So these are close to independent, with one exception worth noting. The highest
overlap is between the two scorers aimed at MedMCQA, which is the sanity check that "resembles the
target" is a coherent signal reached by two different routes, n-gram statistics and embedding
geometry.

There is no "best" here. Selection overlap is not model quality, and nothing in this file says which
method trains a better model. It says the choice of method is not cosmetic, and that a downstream
comparison is worth running.

## Regenerating

```bash
medsel eval --model Qwen/Qwen3-0.6B-Base --tasks medqa medmcqa pubmedqa \
  --output results/base_control_qwen3-0.6b-base.json
```

About 25 minutes on a P5000. Rerun the control whenever the prompt template changes:
`template_version` in each task block records which template produced the number, and a control
from an older template cannot be compared against a newer run.
