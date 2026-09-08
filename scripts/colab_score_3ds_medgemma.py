"""Score MedMCQA with 3DS using MedGemma as the judge model, on Google Colab.

Designed to run on a free Colab T4 (15 GB VRAM) with 4-bit quantization.
Produces the same cache format as the local score_3ds.py so results can be
downloaded and consumed by the local experiment pipeline.

Usage via Colab MCP or as a notebook:
    1. Clone the private repo and install medsel.
    2. Run this script. It caches per-example metrics to disk.
    3. Download the cache directory to merge with local results.

    python scripts/colab_score_3ds_medgemma.py --limit 30000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def ensure_environment() -> None:
    """Install medsel and dependencies if running on Colab."""
    if "COLAB_GPU" not in os.environ and "COLAB_RELEASE_TAG" not in os.environ:
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set. MedGemma is gated and requires acceptance at")
        print("  https://huggingface.co/google/medgemma-4b-pt")
        print("  Set it via Colab secrets or os.environ['HF_TOKEN'] = '...'")

    if not os.path.exists("/content/btp-medsel"):
        subprocess.check_call([
            "git", "clone",
            "https://github.com/wildcraft958/btp-medsel.git",
            "/content/btp-medsel",
        ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "-e", "/content/btp-medsel[train]",
        "bitsandbytes",
    ])

    if "/content/btp-medsel" not in sys.path:
        sys.path.insert(0, "/content/btp-medsel")

    os.chdir("/content/btp-medsel")


def main() -> int:
    ensure_environment()

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
