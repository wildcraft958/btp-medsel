"""Score MedMCQA with 3DS using MedGemma as the judge model.

Uses 4-bit quantization (bitsandbytes NF4) to fit MedGemma-1.5-4b-it on a 16 GB GPU.
Produces the same cache format as score_3ds.py so results feed into the experiment pipeline.

Usage:
    python scripts/score_3ds_medgemma.py --limit 30000
"""

from __future__ import annotations

import argparse
import time


def main() -> int:
    from medsel.registry import get_loader
    from medsel.schema import QAExample
    from medsel.selection.scorers.tds import ThreeDSScorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", default="google/medgemma-1.5-4b-it")
    parser.add_argument(
        "--cache-dir",
        default="runs/sft-selection-experiment/3ds_medgemma/cache",
    )
    parser.add_argument("--atten-method", default="mean", choices=["mean", "max"])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--quantization", default="4bit", choices=["4bit", "8bit", None],
        help="bitsandbytes quantization (default 4bit for T4)",
    )
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
        quantization=args.quantization,
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
    print(f"\nCache directory: {args.cache_dir}")
    print("Download this directory and place it under runs/sft-selection-experiment/3ds_medgemma/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
