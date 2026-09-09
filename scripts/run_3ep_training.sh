#!/usr/bin/env bash
# Re-train all 6 selections at 3 epochs (vs the default 1 epoch).
# Uses a parallel runs directory so 1-epoch models are preserved.
#
#   bash scripts/run_3ep_training.sh 2>&1 | tee runs/sft-selection-experiment-3ep/training.log
set -euo pipefail
cd "$(dirname "$0")/.."

ORIG="runs/sft-selection-experiment"
DEST="runs/sft-selection-experiment-3ep"

echo "=== 3-epoch SFT training for all selections ==="
echo "Started: $(date)"

echo "Setting up selection symlinks..."
for scorer_dir in "$ORIG"/*/; do
    name=$(basename "$scorer_dir")
    sel="$scorer_dir/selection.json"
    if [ -f "$sel" ]; then
        mkdir -p "$DEST/$name"
        if [ ! -e "$DEST/$name/selection.json" ]; then
            ln -s "$(realpath "$sel")" "$DEST/$name/selection.json"
            echo "  linked $name/selection.json"
        else
            echo "  $name/selection.json already exists"
        fi
    fi
done

echo ""
echo "--- Training all selections at 3 epochs ---"
.venv/bin/python scripts/train_and_eval_selections.py \
    --runs-dir "$DEST" \
    --epochs 3 \
    --skip-base \
    --output results/sft_selection_experiment_3ep.json \
    --limit 30000

echo ""
echo "=== 3-epoch training complete ==="
echo "Finished: $(date)"
