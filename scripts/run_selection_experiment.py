"""Train one model per scorer at a matched token budget, then evaluate all of them.

This is the experiment the whole selection layer exists to run. Every scorer picks a subset of the
same pool under the same token budget, each subset gets an identical CPT run, and each result is
scored against the same benchmark with a confidence interval attached.

What makes it a fair comparison, and what to check if a result looks surprising:

- **The budget is in tokens, not records.** ``length`` spends 2.06 times the tokens of ``random``
  for the same record count, so a record-matched run would hand it twice the training data.
- **The budget is anchored to random.** The token budget is whatever a random draw of
  ``--budget`` records costs, so it is set by the neutral method rather than by any scorer's own
  preference.
- **Everything except the data is held fixed.** Same base model, same hyperparameters, same seed,
  same evaluation. The subset is the only thing that varies.
- **The base model is evaluated too.** A CPT result is a delta, and without the untrained control
  in the same table the deltas have nothing to be measured against.

A caveat that belongs in any write-up of the output: identical hyperparameters across recipes is
itself a source of unreliability, because the optimal configuration is data dependent, so a
ranking taken from a single learning rate may not survive a sweep. See section 5 of
docs/research_update_2026-08.md. This script fixes the learning rate to isolate the data effect;
that is the right first experiment and the wrong last one.

    uv run python scripts/run_selection_experiment.py --limit 20000 --budget 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from medsel.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from medsel.registry import get_loader
from medsel.selection.base import available_scorers, get_scorer, record_text
from medsel.selection.costs import load_tokenizer, token_costs
from medsel.selection.selector import select_top_k
from medsel.stages.cpt import CPTStage

SCORER_ARGS: dict[str, dict[str, Any]] = {
    "dsir": {"target": "medmcqa", "target_limit": 1000},
    "embed_similarity": {"target": "medmcqa", "target_limit": 1000},
    "perplexity": {"mode": "mid"},
}


def write_selection(
    path: Path, records: list[Any], chosen: list[int], scorer: str, source: str, split: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": source,
                "split": split,
                "scorer": scorer,
                "n_candidates": len(records),
                "n_selected": len(chosen),
                "selected_uids": [records[i].uid for i in chosen],
            },
            indent=2,
        )
        + "\n"
    )


def train_one(
    scorer: str, selection: Path, args: argparse.Namespace, out_dir: Path
) -> dict[str, Any]:
    config = ExperimentConfig(
        name=f"select-{scorer}",
        stage="cpt",
        data=DataConfig(
            source=args.source,
            split=args.split,
            limit=args.limit,
            loader={"num_shards": args.num_shards},
            selection=str(selection),
        ),
        model=ModelConfig(
            name_or_path=args.model,
            dtype="auto",
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            gradient_checkpointing=True,
        ),
        train=TrainConfig(
            output_dir=str(out_dir),
            block_size=args.block_size,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=args.learning_rate,
            num_train_epochs=1.0,
            logging_steps=20,
            save_steps=10**9,  # above any step count, so only the final save fires
            seed=args.seed,
        ),
        notes=f"Selection experiment, scorer={scorer}, matched token budget.",
    )
    started = time.time()
    result = CPTStage(config).run()
    return {
        "train_runtime_s": round(time.time() - started, 1),
        "n_blocks": result.metrics.get("n_blocks"),
        "tokens_seen": result.metrics.get("tokens_seen"),
        "train_loss": result.metrics.get("train_loss"),
    }


def evaluate_one(
    model_path: str,
    args: argparse.Namespace,
    held_out: list[str],
    target: list[str],
) -> dict[str, Any]:
    """Accuracy plus the two perplexities, from one model load.

    Perplexity is here because accuracy alone cannot answer this experiment's question at this
    compute budget: a 95% interval on MedMCQA is 2.9 points wide and a small LoRA run does not
    move accuracy that far. Perplexity is continuous, uses every token, and responds to a small
    amount of training.
    """
    from medsel.eval.perplexity import evaluate_perplexity
    from medsel.eval.runner import evaluate_task, load_model

    model, tokenizer = load_model(model_path)
    accuracy = evaluate_task(
        model, tokenizer, args.task, model_path, limit=args.eval_limit, progress=True
    )

    payload: dict[str, Any] = {"accuracy": accuracy.to_dict()}
    for name, texts in (("held_out_corpus", held_out), ("target", target)):
        report = evaluate_perplexity(
            model, tokenizer, texts, name, max_length=args.block_size, progress=True
        )
        if report is not None:
            payload[name] = report.to_dict()

    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="pubmed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20_000, help="pool size to select from")
    parser.add_argument(
        "--budget", type=int, default=2000, help="records in the anchoring random draw"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--task", default="medmcqa", help="benchmark to score every run on")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="runs/selection-experiment")
    parser.add_argument(
        "--holdout",
        type=int,
        default=500,
        help="documents for held-out perplexity, taken from what no scorer selected",
    )
    parser.add_argument("--scorers", nargs="*", default=None, help="defaults to all registered")
    parser.add_argument("--output", default="results/selection_experiment.json")
    parser.add_argument(
        "--skip-base", action="store_true", help="do not evaluate the untrained model"
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    loader = get_loader(args.source, num_shards=args.num_shards)
    records = list(loader.load(args.split, limit=args.limit))
    tokens = token_costs(records, load_tokenizer(args.model))
    print(f"pool: {len(records)} records, {sum(tokens):,} tokens")

    anchor = select_top_k(get_scorer("random")(records), args.budget)
    token_budget = sum(tokens[i] for i in anchor)
    print(f"matched token budget: {token_budget:,} from a random {args.budget} record draw")

    scorer_names = list(args.scorers or available_scorers())

    # Score every scorer up front, so the held-out set can exclude everything any of them would
    # train on. A perplexity measured on text one method selected and another did not would
    # reward that method for having seen the evaluation data.
    print("scoring the pool with every scorer")
    selections: dict[str, list[int]] = {}
    for name in scorer_names:
        scorer = get_scorer(name, **SCORER_ARGS.get(name, {}))
        selections[name] = select_top_k(scorer(records), token_budget, costs=tokens)
        spent = sum(tokens[i] for i in selections[name])
        print(f"  {name}: {len(selections[name])} records, {spent:,} tokens")

    ever_selected = {index for chosen in selections.values() for index in chosen}
    held_out_pool = [i for i in range(len(records)) if i not in ever_selected]
    held_out = [record_text(records[i]) for i in held_out_pool[: args.holdout]]
    print(
        f"held out {len(held_out)} of {len(held_out_pool)} documents no scorer selected "
        f"({len(ever_selected)} were selected by at least one)"
    )

    target_texts = get_scorer("dsir", **SCORER_ARGS["dsir"]).target_texts()[: args.holdout]
    print(f"target perplexity set: {len(target_texts)} MedMCQA prompts\n")

    report: dict[str, Any] = {
        "pool": {
            "source": args.source,
            "split": args.split,
            "num_shards": args.num_shards,
            "n_candidates": len(records),
            "pool_tokens": sum(tokens),
        },
        "setup": {
            "model": args.model,
            "task": args.task,
            "token_budget": token_budget,
            "anchor_records": args.budget,
            "block_size": args.block_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "lora": True,
        },
        "holdout": {"n_held_out": len(held_out), "n_target": len(target_texts)},
        "runs": {},
    }

    if not args.skip_base:
        print("evaluating the untrained base model")
        report["runs"]["__base__"] = {
            "selection": None,
            "eval": evaluate_one(args.model, args, held_out, target_texts),
        }

    for name in scorer_names:
        print(f"\n=== {name} ===")
        chosen = selections[name]
        spent = sum(tokens[i] for i in chosen)

        selection = runs_dir / name / "selection.json"
        write_selection(selection, records, chosen, name, args.source, args.split)

        training = train_one(name, selection, args, runs_dir / name / "model")
        print(f"{name}: trained in {training['train_runtime_s']}s")
        evaluation = evaluate_one(str(runs_dir / name / "model"), args, held_out, target_texts)

        report["runs"][name] = {
            "selection": {
                "n_selected": len(chosen),
                "tokens": spent,
                "scorer_args": SCORER_ARGS.get(name, {}),
            },
            "training": training,
            "eval": evaluation,
        }

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")

    summarise(report, args)
    print(f"\nwritten to {args.output}")
    return 0


def summarise(report: dict[str, Any], args: argparse.Namespace) -> None:
    runs = report["runs"]
    header = (
        f"{'run':<18}{'records':>9}{'tokens':>11}{'accuracy':>10}"
        f"{'95% interval':>20}{'held-out ppl':>15}{'target ppl':>13}"
    )
    print(f"\n{header}")
    for name, entry in runs.items():
        accuracy = entry["eval"]["accuracy"]
        low, high = accuracy.get("accuracy_ci") or (float("nan"), float("nan"))
        selection = entry.get("selection") or {}
        held = entry["eval"].get("held_out_corpus", {}).get("perplexity", float("nan"))
        target = entry["eval"].get("target", {}).get("perplexity", float("nan"))
        print(
            f"{name:<18}{selection.get('n_selected', 0):>9}{selection.get('tokens', 0):>11,}"
            f"{accuracy['accuracy']:>10.4f}  [{low:.4f}, {high:.4f}]"
            f"{held:>15.4f}{target:>13.4f}"
        )

    base = runs.get("__base__")
    if not base:
        return

    accuracy = base["eval"]["accuracy"]
    interval = accuracy.get("accuracy_ci") or (0.0, 0.0)
    width = interval[1] - interval[0]
    print(f"\nagainst the untrained base on {args.task} (n={accuracy['n_scored']}):")
    print(
        f"  the base interval is {width * 100:.2f} points wide. An accuracy delta smaller than "
        f"that is not evidence of anything."
    )
    for name, entry in runs.items():
        if name == "__base__":
            continue
        delta = entry["eval"]["accuracy"]["accuracy"] - accuracy["accuracy"]
        verdict = "inside noise" if abs(delta) < width else "clears the interval"
        held = entry["eval"].get("held_out_corpus", {}).get("perplexity")
        base_held = base["eval"].get("held_out_corpus", {}).get("perplexity")
        target = entry["eval"].get("target", {}).get("perplexity")
        base_target = base["eval"].get("target", {}).get("perplexity")
        parts = [f"acc {delta:+.4f} ({verdict})"]
        if held and base_held:
            parts.append(f"held-out ppl {held / base_held:.4f}x")
        if target and base_target:
            parts.append(f"target ppl {target / base_target:.4f}x")
        print(f"  {name:<18}{'  '.join(parts)}")


if __name__ == "__main__":
    raise SystemExit(main())
