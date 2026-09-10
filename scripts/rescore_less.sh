#!/usr/bin/env bash
# Re-score LESS with the bugfix (validation targets + completion text).
# Candidate gradients are cached; only target gradients recompute (~50 examples x 4 epochs).
# Then re-select, re-train at 1 and 3 epochs, and re-evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="runs/sft-selection-experiment"

echo "=== LESS re-scoring with bugfix ==="
echo "Started: $(date)"

echo "--- Step 1: Re-score (recomputes 200 target gradients) ---"
.venv/bin/python scripts/score_less.py --limit 30000

echo ""
echo "--- Step 2: Re-select ---"
rm -f "$RUNS/less/selection.json"
.venv/bin/python scripts/create_selections_from_cache.py --limit 30000 --scorers less --force

echo ""
echo "--- Step 3: Re-train at 1 epoch ---"
rm -rf "$RUNS/less/model"
.venv/bin/python scripts/train_and_eval_selections.py \
    --limit 30000 --scorers less --skip-base \
    --output results/sft_selection_experiment.json

echo ""
echo "--- Step 4: Re-train at 3 epochs ---"
DEST="runs/sft-selection-experiment-3ep"
rm -f "$DEST/less/selection.json" 2>/dev/null
rm -rf "$DEST/less/model" 2>/dev/null
mkdir -p "$DEST/less"
ln -sf "$(realpath "$RUNS/less/selection.json")" "$DEST/less/selection.json"
.venv/bin/python scripts/train_and_eval_selections.py \
    --runs-dir "$DEST" --epochs 3 --skip-base --scorers less \
    --output results/sft_selection_experiment_3ep.json \
    --limit 30000

echo ""
echo "=== LESS re-scoring complete ==="
echo "Finished: $(date)"
