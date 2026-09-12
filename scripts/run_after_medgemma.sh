#!/usr/bin/env bash
# Chains all remaining work after MedGemma scoring finishes:
#   1. 1-epoch fixup (perplexity + random)
#   2. LESS rescore with bugfix
#   3. Small-budget experiment (3ds, dsir, less, random at ~27k tokens)
#   4. Seed sweep (3ds, dsir, less, random x 3 seeds x 3 epochs)
#
# Run in a tmux session:
#   tmux send-keys -t less_run 'bash scripts/run_after_medgemma.sh 2>&1 | tee runs/after_medgemma.log' Enter
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Waiting for MedGemma to finish ==="
echo "Checking pipeline_watcher tmux session every 5 minutes..."
while tmux has-session -t pipeline_watcher 2>/dev/null; do
    echo "  $(date): pipeline_watcher still running..."
    sleep 300
done
echo "MedGemma done at $(date)"

echo ""
echo "========================================"
echo "=== Step 1: 1-epoch fixup ===="
echo "========================================"
.venv/bin/python scripts/train_and_eval_selections.py \
    --limit 30000 --scorers perplexity random --skip-base \
    --output results/sft_selection_experiment.json \
    2>&1 | tee runs/sft-selection-experiment/phase3_fixup.log
echo "1-epoch fixup done at $(date)"

echo ""
echo "========================================"
echo "=== Step 2: LESS rescore with bugfix ==="
echo "========================================"
bash scripts/rescore_less.sh
echo "LESS rescore done at $(date)"

echo ""
echo "========================================"
echo "=== Step 3: Small-budget experiment ===="
echo "========================================"
bash scripts/run_small_budget_experiment.sh 2>&1 | tee runs/sft-small-budget/experiment.log
echo "Small-budget done at $(date)"

echo ""
echo "========================================"
echo "=== Step 4: Seed sweep ================="
echo "========================================"
mkdir -p runs/sft-selection-experiment-sweep
bash scripts/run_seed_sweep.sh 2>&1 | tee runs/sft-selection-experiment-sweep/sweep.log
echo "Seed sweep done at $(date)"

echo ""
echo "=== ALL REMAINING WORK COMPLETE ==="
echo "Finished: $(date)"
