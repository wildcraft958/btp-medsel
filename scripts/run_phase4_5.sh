#!/usr/bin/env bash
# Phase 4 (judge ablation) and Phase 5 (analysis).
# Run after Phase 3 completes AND MedGemma cache is on the remote.
#
#   bash scripts/run_phase4_5.sh 2>&1 | tee runs/sft-selection-experiment/phase4_5.log
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 4: Judge Ablation ==="
echo "Started: $(date)"

MG_CACHE="runs/sft-selection-experiment/3ds_medgemma/cache/tds_cache.jsonl"
SELF_CACHE="runs/sft-selection-experiment/3ds/cache/tds_cache.jsonl"

if [ ! -f "$MG_CACHE" ]; then
    echo "ERROR: MedGemma cache not found at $MG_CACHE"
    echo "Transfer from Colab first."
    exit 1
fi

if [ ! -f "$SELF_CACHE" ]; then
    echo "ERROR: Self-judged 3DS cache not found at $SELF_CACHE"
    exit 1
fi

MG_COUNT=$(wc -l < "$MG_CACHE")
SELF_COUNT=$(wc -l < "$SELF_CACHE")
echo "  MedGemma cache: $MG_COUNT entries"
echo "  Self-judged cache: $SELF_COUNT entries"

echo ""
echo "--- Step 1: Run judge ablation (train + eval both judges) ---"
.venv/bin/python scripts/run_judge_ablation.py --limit 30000

echo ""
echo "--- Step 2: Full analysis with ablation ---"
.venv/bin/python scripts/analyze_sft_experiment.py \
    --ablation results/judge_ablation.json --latex

echo ""
echo "=== Phase 4+5 complete ==="
echo "Finished: $(date)"
echo "Main results: results/sft_selection_experiment.json"
echo "Ablation results: results/judge_ablation.json"
