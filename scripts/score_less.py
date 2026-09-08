"""Run LESS scoring on MedMCQA train and cache per-example influence scores.

Crash-resumable: warmup checkpoints and projected gradients are cached to disk.
Restart and it picks up from the last cached gradient.

    .venv/bin/python scripts/score_less.py --limit 1000   # smoke test
    .venv/bin/python scripts/score_less.py --limit 30000  # experiment pool
"""

from __future__ import annotations

import argparse
import time

from medsel.registry import get_loader
from medsel.schema import QAExample
from medsel.selection.scorers.less import LESSScorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--cache-dir", default="runs/sft-selection-experiment/less/cache")
    parser.add_argument("--target", default="medmcqa")
    parser.add_argument("--target-limit", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    loader = get_loader(args.source)
    if loader.record_type is not QAExample:
        raise ValueError(f"LESS needs a QA source, got {args.source}")

    records = list(loader.load(args.split, limit=args.limit))
    print(f"pool: {len(records)} records from {args.source}:{args.split}")

    scorer = LESSScorer(
        judge=args.judge,
        target=args.target,
        target_limit=args.target_limit,
        cache_dir=args.cache_dir,
        lora_r=args.lora_r,
        warmup_epochs=args.warmup_epochs,
        proj_dim=args.proj_dim,
        max_length=args.max_length,
    )

    t0 = time.time()
    scores = scorer.score(records)
    elapsed = time.time() - t0

    finite = [s for s in scores if s != float("-inf")]
    print(f"scored {len(records)} in {elapsed:.0f}s ({elapsed / max(len(records), 1):.2f}s/ex)")
    print(f"  finite scores: {len(finite)} / {len(records)}")
    if finite:
        print(f"  score range: [{min(finite):.6f}, {max(finite):.6f}]")

    scorer.release_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
