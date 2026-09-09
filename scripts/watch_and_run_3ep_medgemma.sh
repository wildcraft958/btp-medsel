#!/usr/bin/env bash
# Watch for 1-epoch Phase 3 to complete, then run:
#   1. 3-epoch SFT training for all selections
#   2. MedGemma 3DS scoring
#   3. Phase 4+5 analysis
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="runs/sft-selection-experiment/phase3.log"
MARKER="Phase 3 complete"

echo "Watching for 1-epoch Phase 3 completion..."
echo "  Looking for '$MARKER' in $LOG"
echo "  Checking every 2 minutes"

while true; do
    if [ -f "$LOG" ] && grep -q "$MARKER" "$LOG" 2>/dev/null; then
        echo ""
        echo "Phase 3 (1-epoch) complete detected at $(date)"
        echo ""
        echo "=== Step 1: 3-epoch training ==="
        bash scripts/run_3ep_training.sh
        echo ""
        echo "=== Step 2: MedGemma scoring + Phase 4+5 ==="
        bash scripts/run_medgemma_and_phase4_5.sh
        echo ""
        echo "=== All pipeline stages complete ==="
        echo "Finished: $(date)"
        exit 0
    fi
    sleep 120
done
