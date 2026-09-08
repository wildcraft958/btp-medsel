#!/usr/bin/env bash
# Phase 3 pipeline: create selections from caches, train all, evaluate all.
# Run on the remote after all scoring jobs finish.
#
#   bash scripts/run_phase3_pipeline.sh 2>&1 | tee runs/sft-selection-experiment/phase3.log
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 3: Create selections, train, evaluate ==="
echo "Started: $(date)"

echo ""
echo "--- Step 1: Create selection manifests from caches ---"
.venv/bin/python scripts/create_selections_from_cache.py --limit 30000

echo ""
echo "--- Step 2: Train and evaluate all selections ---"
.venv/bin/python scripts/train_and_eval_selections.py --limit 30000

echo ""
echo "--- Step 3: Run analysis ---"
.venv/bin/python scripts/analyze_sft_experiment.py --latex

echo ""
echo "=== Phase 3 complete ==="
echo "Finished: $(date)"
echo "Results: results/sft_selection_experiment.json"
