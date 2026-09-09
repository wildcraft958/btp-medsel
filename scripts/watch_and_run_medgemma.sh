#!/usr/bin/env bash
# Watch for LESS+Phase3 to complete, then run MedGemma scoring + Phase 4+5.
# Checks every 5 minutes for the phase3.log completion marker.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="runs/sft-selection-experiment/phase3.log"
MARKER="Phase 3 complete"

echo "Watching for Phase 3 completion..."
echo "  Looking for '$MARKER' in $LOG"
echo "  Checking every 5 minutes"

while true; do
    if [ -f "$LOG" ] && grep -q "$MARKER" "$LOG" 2>/dev/null; then
        echo ""
        echo "Phase 3 complete detected at $(date)"
        echo "Starting MedGemma scoring + Phase 4+5..."
        bash scripts/run_medgemma_and_phase4_5.sh
        exit 0
    fi
    sleep 300
done
