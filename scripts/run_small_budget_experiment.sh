#!/usr/bin/env bash
# Small-budget experiment: 500-record selections (~25k tokens).
# Tests whether selection methods separate from random at lower budgets,
# confirming Jeong et al.'s finding that budget size drives convergence.
#
#   bash scripts/run_small_budget_experiment.sh 2>&1 | tee runs/sft-small-budget/experiment.log
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="runs/sft-small-budget"
BUDGET=500
LIMIT=30000
EPOCHS=3

echo "=== Small-budget experiment: ${BUDGET} records, ${EPOCHS} epochs ==="
echo "Started: $(date)"

echo ""
echo "--- Step 1: Create any missing selections ---"
.venv/bin/python scripts/create_selections_from_cache.py \
    --limit "$LIMIT" \
    --budget "$BUDGET" \
    --runs-dir "$RUNS"

echo ""
echo "--- Step 2: Train and evaluate (+ base eval) ---"
SCORERS=""
for d in "$RUNS"/*/; do
    [ -f "$d/selection.json" ] && SCORERS="$SCORERS $(basename "$d")"
done
SCORERS=$(echo $SCORERS | xargs)
echo "Scorers with selections: $SCORERS"

.venv/bin/python scripts/train_and_eval_selections.py \
    --runs-dir "$RUNS" \
    --epochs "$EPOCHS" \
    --scorers $SCORERS \
    --output results/sft_small_budget_experiment.json \
    --limit "$LIMIT"

echo ""
echo "=== Small-budget experiment complete ==="
echo "Finished: $(date)"
