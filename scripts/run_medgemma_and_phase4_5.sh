#!/usr/bin/env bash
# Run MedGemma 3DS scoring on the remote P5000 (4-bit NF4), then Phase 4+5.
# Chains after LESS + Phase 3 completes.
#
#   bash scripts/run_medgemma_and_phase4_5.sh 2>&1 | tee runs/sft-selection-experiment/medgemma_phase4_5.log
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== MedGemma 3DS scoring on remote ==="
echo "Started: $(date)"

# Score with MedGemma judge (4-bit quantization for P5000)
.venv/bin/python scripts/score_3ds_medgemma.py \
    --limit 30000 \
    --quantization 4bit

echo ""
echo "=== MedGemma scoring done, starting Phase 4+5 ==="
bash scripts/run_phase4_5.sh

echo ""
echo "=== Everything complete ==="
echo "Finished: $(date)"
