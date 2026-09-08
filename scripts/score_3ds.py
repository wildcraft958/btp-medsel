"""Run 3DS scoring on MedMCQA train and cache the per-example metrics.

The experiment harness reads the same cache, so scoring can run independently
of SFT training. Crash-resumable: restart and it picks up from the last
cached example.

    .venv/bin/python scripts/score_3ds.py --limit 1000   # smoke test
    .venv/bin/python scripts/score_3ds.py                 # full 182k
"""

from __future__ import annotations

import argparse
import time

from medsel.registry import get_loader
from medsel.schema import QAExample
from medsel.selection.scorers.tds import ThreeDSScorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--cache-dir", default="runs/sft-selection-experiment/3ds/cache")
    parser.add_argument("--atten-method", default="mean", choices=["mean", "max"])
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    loader = get_loader(args.source)
    if loader.record_type is not QAExample:
        raise ValueError(f"3DS needs a QA source, got {args.source}")

    records = list(loader.load(args.split, limit=args.limit))
    print(f"pool: {len(records)} records from {args.source}:{args.split}")

    scorer = ThreeDSScorer(
        judge=args.judge,
        cache_dir=args.cache_dir,
        atten_method=args.atten_method,
        max_length=args.max_length,
    )

    cached = scorer._load_cache()
    print(f"cache: {len(cached)} entries already in {args.cache_dir}")

    t0 = time.time()
    scores = scorer.score(records)
    elapsed = time.time() - t0

    finite = [s for s in scores if s != float("-inf")]
    print(f"scored {len(records)} in {elapsed:.0f}s ({elapsed / max(len(records), 1):.2f}s/ex)")
    print(f"  finite (in Goldilocks zone): {len(finite)} / {len(records)}")
    if finite:
        print(f"  score range: [{min(finite):.2f}, {max(finite):.2f}]")

    updated_cache = scorer._load_cache()
    print(f"  cache now: {len(updated_cache)} entries")

    scorer.release_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
