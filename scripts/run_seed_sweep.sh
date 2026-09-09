#!/usr/bin/env bash
# 3-seed sweep at 3 epochs for key methods (3ds, less, dsir, random).
# Each seed gets its own runs directory so models don't collide.
#
#   bash scripts/run_seed_sweep.sh 2>&1 | tee runs/sft-selection-experiment-sweep/sweep.log
set -euo pipefail
cd "$(dirname "$0")/.."

ORIG="runs/sft-selection-experiment"
SWEEP_DIR="runs/sft-selection-experiment-sweep"
SCORERS="3ds dsir less random"
SEEDS="42 43 44"

echo "=== Seed sweep: 3 seeds x 4 methods x 3 epochs ==="
echo "Started: $(date)"
echo "Methods: $SCORERS"
echo "Seeds: $SEEDS"

for seed in $SEEDS; do
    DEST="$SWEEP_DIR/seed_$seed"
    echo ""
    echo "--- Setting up seed $seed ---"

    for scorer in $SCORERS; do
        sel="$ORIG/$scorer/selection.json"
        if [ -f "$sel" ]; then
            mkdir -p "$DEST/$scorer"
            if [ ! -e "$DEST/$scorer/selection.json" ]; then
                ln -s "$(realpath "$sel")" "$DEST/$scorer/selection.json"
                echo "  linked $scorer/selection.json"
            fi
        else
            echo "  WARNING: $sel not found"
        fi
    done

    echo ""
    echo "=== Seed $seed: training 4 methods at 3 epochs ==="
    .venv/bin/python scripts/train_and_eval_selections.py \
        --runs-dir "$DEST" \
        --epochs 3 \
        --seed "$seed" \
        --skip-base \
        --scorers $SCORERS \
        --output "$SWEEP_DIR/results_seed_$seed.json" \
        --limit 30000

    echo "  Seed $seed done at $(date)"
done

echo ""
echo "=== Aggregating sweep results ==="
.venv/bin/python -c "
import json
from pathlib import Path

sweep_dir = Path('$SWEEP_DIR')
seeds = [42, 43, 44]
scorers = '$SCORERS'.split()
agg = {'seeds': seeds, 'epochs': 3, 'scorers': {}}

for scorer in scorers:
    accs = []
    for seed in seeds:
        result_file = sweep_dir / f'results_seed_{seed}.json'
        if result_file.exists():
            data = json.loads(result_file.read_text())
            run = data.get('runs', {}).get(scorer, {})
            acc = run.get('eval', {}).get('accuracy', {}).get('accuracy')
            if acc is not None:
                accs.append(acc)
    if accs:
        mean_acc = sum(accs) / len(accs)
        variance = sum((a - mean_acc) ** 2 for a in accs) / len(accs)
        std = variance ** 0.5
        agg['scorers'][scorer] = {
            'accuracies': accs,
            'mean': round(mean_acc, 4),
            'std': round(std, 4),
            'min': round(min(accs), 4),
            'max': round(max(accs), 4),
        }
        print(f'  {scorer}: {mean_acc:.4f} +/- {std:.4f}  ({accs})')

out = sweep_dir / 'sweep_aggregate.json'
out.write_text(json.dumps(agg, indent=2) + '\n')
print(f'\nSaved to {out}')
"

echo ""
echo "=== Seed sweep complete ==="
echo "Finished: $(date)"
