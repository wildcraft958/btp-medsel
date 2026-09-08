#!/usr/bin/env bash
# Transfer MedGemma (and optionally LESS) cache from local to remote.
# Run after downloading the tar.gz from Colab/Drive.
#
# Usage:
#   bash scripts/transfer_colab_results.sh /path/to/medgemma_cache.tar.gz
#   bash scripts/transfer_colab_results.sh /path/to/medgemma_cache.tar.gz /path/to/less_cache_partial.tar.gz
set -euo pipefail

REMOTE="medsel"
REMOTE_DIR="~/BTP/runs/sft-selection-experiment"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <medgemma_cache.tar.gz> [less_cache_partial.tar.gz]"
    exit 1
fi

MEDGEMMA_TAR="$1"
echo "=== Transferring MedGemma cache ==="
echo "  Source: $MEDGEMMA_TAR"
scp "$MEDGEMMA_TAR" "$REMOTE:/tmp/medgemma_cache.tar.gz"
ssh "$REMOTE" "cd $REMOTE_DIR && tar xzf /tmp/medgemma_cache.tar.gz && rm /tmp/medgemma_cache.tar.gz"

MG_COUNT=$(ssh "$REMOTE" "wc -l < $REMOTE_DIR/3ds_medgemma/cache/tds_cache.jsonl")
echo "  Transferred: $MG_COUNT entries"

if [ $# -ge 2 ]; then
    LESS_TAR="$2"
    echo ""
    echo "=== Transferring LESS partial cache ==="
    echo "  Source: $LESS_TAR"
    scp "$LESS_TAR" "$REMOTE:/tmp/less_cache.tar.gz"
    ssh "$REMOTE" "cd $REMOTE_DIR && tar xzf /tmp/less_cache.tar.gz && rm /tmp/less_cache.tar.gz"
    echo "  Done"
fi

echo ""
echo "=== Remote state ==="
ssh "$REMOTE" "ls -la $REMOTE_DIR/3ds_medgemma/cache/tds_cache.jsonl 2>/dev/null; ls -la $REMOTE_DIR/less/cache/ 2>/dev/null | head -5"
echo ""
echo "Ready for Phase 4+5:"
echo "  ssh $REMOTE 'cd ~/BTP && bash scripts/run_phase4_5.sh'"
