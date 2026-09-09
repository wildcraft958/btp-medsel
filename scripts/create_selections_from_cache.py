"""Create token-matched selection manifests from pre-scored caches.

After score_3ds.py and score_less.py populate their caches, this script reads
those caches, computes the final scores, creates token-budget-matched selections,
and writes selection.json manifests that train_and_eval_selections.py consumes.

Also creates selections for scorers that run inline (embed_similarity, perplexity,
dsir, random) if their manifests are missing.

    .venv/bin/python scripts/create_selections_from_cache.py --limit 30000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medsel.prompts.templates import render_prompt
from medsel.registry import get_loader
from medsel.schema import QAExample
from medsel.selection.base import available_scorers, get_scorer
from medsel.selection.costs import load_tokenizer, token_costs
from medsel.selection.selector import select_top_k
from medsel.stages.sft import render_completion

SCORER_ARGS: dict[str, dict[str, Any]] = {
    "3ds": {"cache_dir": "runs/sft-selection-experiment/3ds/cache"},
    "dsir": {"target": "medmcqa", "target_limit": 1000},
    "embed_similarity": {"target": "medmcqa", "target_limit": 1000},
    "less": {"cache_dir": "runs/sft-selection-experiment/less/cache"},
    "perplexity": {"mode": "mid"},
}

SKIP_SCORERS = {"length"}


def sft_render(example: Any) -> str:
    return render_prompt(example) + render_completion(example)


def write_selection(
    path: Path, records: list[Any], chosen: list[int],
    scorer: str, source: str, split: str, tokens: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spent = sum(tokens[i] for i in chosen)
    path.write_text(
        json.dumps(
            {
                "source": source,
                "split": split,
                "scorer": scorer,
                "n_candidates": len(records),
                "n_selected": len(chosen),
                "tokens": spent,
                "selected_uids": [records[i].uid for i in chosen],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--runs-dir", default="runs/sft-selection-experiment")
    parser.add_argument(
        "--scorers", nargs="*", default=None,
        help="which scorers to create selections for (default: all with caches)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing manifests")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    loader = get_loader(args.source)

    if loader.record_type is not QAExample:
        raise ValueError(f"needs a QA source, got {args.source}")

    records = list(loader.load(args.split, limit=args.limit))
    tokenizer = load_tokenizer(args.model)
    tokens = token_costs(records, tokenizer, render=sft_render)
    print(f"pool: {len(records)} records, {sum(tokens):,} rendered tokens")

    anchor = select_top_k(get_scorer("random")(records), args.budget)
    token_budget = sum(tokens[i] for i in anchor)
    print(f"token budget: {token_budget:,} (from random {args.budget}-record anchor)")

    scorer_names = args.scorers or [
        n for n in available_scorers() if n not in SKIP_SCORERS
    ]

    for name in scorer_names:
        sel_path = runs_dir / name / "selection.json"
        if sel_path.exists() and not args.force:
            manifest = json.loads(sel_path.read_text())
            print(f"  {name}: already exists ({manifest['n_selected']} records), skipping")
            continue

        print(f"  {name}: scoring...")
        try:
            scorer = get_scorer(name, **SCORER_ARGS.get(name, {}))
            scores = scorer(records)
            chosen = select_top_k(scores, token_budget, costs=tokens)
            write_selection(sel_path, records, chosen, name, args.source, args.split, tokens)
            spent = sum(tokens[i] for i in chosen)
            print(f"    selected {len(chosen)} records, {spent:,} tokens -> {sel_path}")
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    print("\nSelection manifests:")
    for d in sorted(runs_dir.iterdir()):
        sel = d / "selection.json"
        if sel.exists():
            m = json.loads(sel.read_text())
            tok = m.get('tokens', '?')
            tok_str = f"{tok:,}" if isinstance(tok, int) else str(tok)
            print(f"  {d.name}: {m['n_selected']} records, {tok_str} tokens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
