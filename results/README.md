# Committed reference results

Evaluation reports that other results are measured against. Small JSON, committed on purpose:
a CPT or SFT result is a delta, and a delta needs a control produced by this repo under the same
prompts and the same scoring convention. A published number from a paper is not a substitute,
because prompt and few-shot conventions differ.

Training artifacts still never get committed. `runs/` and `data/` stay gitignored.

| File | What it is |
|---|---|
| `base_control_qwen3-0.6b-base.json` | `Qwen/Qwen3-0.6B-Base`, untrained, on the three labelled QA splits |

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

## Regenerating

```bash
medsel eval --model Qwen/Qwen3-0.6B-Base --tasks medqa medmcqa pubmedqa \
  --output results/base_control_qwen3-0.6b-base.json
```

About 25 minutes on a P5000. Rerun the control whenever the prompt template changes:
`template_version` in each task block records which template produced the number, and a control
from an older template cannot be compared against a newer run.
