#!/usr/bin/env bash
# Re-score DSIR and embed_similarity after pool/target format mismatch fix.
# Both score inline (no cache), so re-scoring is just re-running selection.
# Then re-train at 1 and 3 epochs and re-evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="runs/sft-selection-experiment"

echo "=== DSIR + embed_similarity re-scoring with format fix ==="
echo "Started: $(date)"

echo "--- Step 1: Re-select (re-scores inline) ---"
rm -f "$RUNS/dsir/selection.json"
rm -f "$RUNS/embed_similarity/selection.json"
.venv/bin/python scripts/create_selections_from_cache.py \
    --limit 30000 --scorers dsir embed_similarity --force

echo ""
echo "--- Step 2: Re-train at 1 epoch ---"
rm -rf "$RUNS/dsir/model"
rm -rf "$RUNS/embed_similarity/model"
.venv/bin/python scripts/train_and_eval_selections.py \
    --limit 30000 --scorers dsir embed_similarity --skip-base \
    --output results/sft_selection_experiment.json

echo ""
echo "--- Step 3: Re-train at 3 epochs ---"
DEST="runs/sft-selection-experiment-3ep"
for scorer in dsir embed_similarity; do
    rm -f "$DEST/$scorer/selection.json" 2>/dev/null
    rm -rf "$DEST/$scorer/model" 2>/dev/null
    mkdir -p "$DEST/$scorer"
    ln -sf "$(realpath "$RUNS/$scorer/selection.json")" "$DEST/$scorer/selection.json"
done
.venv/bin/python scripts/train_and_eval_selections.py \
    --runs-dir "$DEST" --epochs 3 --skip-base --scorers dsir embed_similarity \
    --output results/sft_selection_experiment_3ep.json \
    --limit 30000

echo ""
echo "=== DSIR + embed_similarity re-scoring complete ==="
echo "Finished: $(date)"
