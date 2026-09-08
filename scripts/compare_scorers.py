"""Run every registered scorer over one pool and compare what each keeps.

This is the cheap experiment that should happen before an expensive one. It answers two questions
that decide whether a gradient method is worth days of compute:

1. Do the scorers actually disagree? If they select largely the same documents, the choice of
   method cannot explain a downstream difference and something else is going on.
2. Does the comparison hold its budget fixed in the units that matter? Ranking is by record, but
   training consumes tokens, and a scorer that prefers long documents gets more tokens for the same
   record count. That disparity is measured here rather than assumed away.

It trains nothing. Selection overlap is not model quality, and no conclusion about which method is
better belongs here.

    uv run python scripts/compare_scorers.py --limit 5000 --budget 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medsel.registry import get_loader
from medsel.selection.base import available_scorers, get_scorer
from medsel.selection.costs import load_tokenizer, token_costs
from medsel.selection.selector import select_top_k

# Constructor arguments per scorer. The two targeted scorers point at MedMCQA training data, which
# TargetedScorer checks is disjoint from the split MedMCQA is scored on.
SCORER_ARGS: dict[str, dict[str, Any]] = {
    "dsir": {"target": "medmcqa", "target_limit": 500},
    "embed_similarity": {"target": "medmcqa", "target_limit": 500},
    "perplexity": {"mode": "mid"},
}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="pubmed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--budget", type=int, default=500, help="records each scorer keeps")
    parser.add_argument("--loader", nargs="*", default=["num_shards=1"], metavar="KEY=VALUE")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="results/scorer_comparison.json")
    args = parser.parse_args()

    loader_kwargs: dict[str, Any] = {}
    for pair in args.loader:
        key, _, value = pair.partition("=")
        loader_kwargs[key] = int(value) if value.isdigit() else value

    loader = get_loader(args.source, **loader_kwargs)
    records = list(loader.load(args.split, limit=args.limit))
    print(f"pool: {len(records)} records from {args.source}:{args.split}")

    tokens = token_costs(records, load_tokenizer(args.tokenizer))
    total_tokens = sum(tokens)
    print(f"pool tokens: {total_tokens:,} ({total_tokens / len(records):.1f} per record)")

    # Every scorer gets the token budget that a random selection of `budget` records would consume,
    # so the token-matched comparison is anchored to the neutral method rather than to any scorer's
    # own preference.
    random_indices = select_top_k(get_scorer("random")(records), args.budget)
    token_budget = sum(tokens[i] for i in random_indices)
    print(f"matched token budget: {token_budget:,} (from a random {args.budget} record draw)\n")

    report: dict[str, Any] = {
        "source": args.source,
        "split": args.split,
        "n_candidates": len(records),
        "record_budget": args.budget,
        "token_budget": token_budget,
        "pool_tokens": total_tokens,
        "scorers": {},
    }

    for name in available_scorers():
        scorer = get_scorer(name, **SCORER_ARGS.get(name, {}))
        scores = scorer(records)

        by_record = select_top_k(scores, args.budget)
        by_token = select_top_k(scores, token_budget, costs=tokens)

        report["scorers"][name] = {
            "record_budget": {
                "n_selected": len(by_record),
                "tokens": sum(tokens[i] for i in by_record),
                "mean_tokens": sum(tokens[i] for i in by_record) / max(1, len(by_record)),
                "uids": [records[i].uid for i in by_record],
            },
            "token_budget": {
                "n_selected": len(by_token),
                "tokens": sum(tokens[i] for i in by_token),
                "uids": [records[i].uid for i in by_token],
            },
            "score_min": min(scores),
            "score_max": max(scores),
            "selected_mean": sum(scores[i] for i in by_record) / max(1, len(by_record)),
            "pool_mean": sum(scores) / len(scores),
        }
        print(f"  {name} done")

    names = list(report["scorers"])
    report["overlap_at_record_budget"] = {
        f"{a}|{b}": round(
            jaccard(
                set(report["scorers"][a]["record_budget"]["uids"]),
                set(report["scorers"][b]["record_budget"]["uids"]),
            ),
            4,
        )
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The uid lists are the bulky part and are only needed to recompute overlaps, so they stay out
    # of the committed report.
    trimmed = json.loads(json.dumps(report))
    for entry in trimmed["scorers"].values():
        entry["record_budget"].pop("uids")
        entry["token_budget"].pop("uids")
    path.write_text(json.dumps(trimmed, indent=2) + "\n")

    print(f"\n{'scorer':<18}{'records':>9}{'tokens':>12}{'tok/doc':>10}{'vs random':>11}")
    baseline = report["scorers"]["random"]["record_budget"]["tokens"]
    for name in names:
        entry = report["scorers"][name]["record_budget"]
        print(
            f"{name:<18}{entry['n_selected']:>9}{entry['tokens']:>12,}"
            f"{entry['mean_tokens']:>10.1f}{entry['tokens'] / baseline:>10.2f}x"
        )

    print(f"\nat a matched token budget of {token_budget:,}:")
    print(f"{'scorer':<18}{'records':>9}{'tokens':>12}")
    for name in names:
        entry = report["scorers"][name]["token_budget"]
        print(f"{name:<18}{entry['n_selected']:>9}{entry['tokens']:>12,}")

    print("\npairwise overlap at the record budget (Jaccard):")
    for pair, value in sorted(report["overlap_at_record_budget"].items(), key=lambda kv: -kv[1]):
        print(f"  {pair:<40}{value:.4f}")

    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
