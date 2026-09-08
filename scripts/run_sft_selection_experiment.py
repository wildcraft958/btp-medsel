"""Train one SFT model per scorer at a matched token budget, then evaluate all.

The MedMCQA counterpart of run_selection_experiment.py. Every scorer picks a subset of
MedMCQA train under the same token budget, each subset gets an identical SFT run, and
each result is scored on MedMCQA validation with a confidence interval.

Budget is in tokens of rendered prompt + completion pairs, not raw question text, because
SFT trains on the full rendered example and that is what the budget must reflect.

    uv run python scripts/run_sft_selection_experiment.py --limit 20000 --budget 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from medsel.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from medsel.prompts.templates import render_prompt
from medsel.registry import get_loader
from medsel.schema import QAExample
from medsel.selection.base import available_scorers, get_scorer
from medsel.selection.costs import load_tokenizer, token_costs
from medsel.selection.selector import select_top_k
from medsel.stages.sft import SFTStage, render_completion

SCORER_ARGS: dict[str, dict[str, Any]] = {
    "3ds": {"cache_dir": "runs/sft-selection-experiment/3ds/cache"},
    "dsir": {
        "target": "medmcqa",
        "target_limit": 1000,
        "dsir_note": "per-token normalized, not summed (see dsir.py docstring)",
    },
    "embed_similarity": {"target": "medmcqa", "target_limit": 1000},
    "less": {"cache_dir": "runs/sft-selection-experiment/less/cache"},
    "perplexity": {"mode": "mid"},
}


def sft_render(example: Any) -> str:
    """The text whose token count is the SFT training cost of one example."""
    return render_prompt(example) + render_completion(example)


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
        name=f"sft-select-{scorer}",
        stage="sft",
        data=DataConfig(
            source=args.source,
            split=args.split,
            limit=args.limit,
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
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            logging_steps=20,
            save_steps=10**9,
            seed=args.seed,
            max_length=args.max_length,
        ),
        notes=f"SFT selection experiment, scorer={scorer}, matched token budget.",
    )
    started = time.time()
    result = SFTStage(config).run()
    return {
        "train_runtime_s": round(time.time() - started, 1),
        "n_examples": result.metrics.get("n_examples"),
        "train_loss": result.metrics.get("train_loss"),
    }


def evaluate_one(model_path: str, args: argparse.Namespace) -> dict[str, Any]:
    from medsel.eval.runner import evaluate_task, load_model

    model, tokenizer = load_model(model_path)
    accuracy = evaluate_task(
        model, tokenizer, args.task, model_path, limit=args.eval_limit, progress=True
    )
    payload: dict[str, Any] = {"accuracy": accuracy.to_dict()}

    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="pool size to select from")
    parser.add_argument(
        "--budget", type=int, default=2000, help="records in the anchoring random draw"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--task", default="medmcqa", help="benchmark to evaluate on")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="runs/sft-selection-experiment")
    parser.add_argument("--scorers", nargs="*", default=None, help="defaults to all registered")
    parser.add_argument("--output", default="results/sft_selection_experiment.json")
    parser.add_argument(
        "--skip-base", action="store_true", help="do not evaluate the untrained model"
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="score and write selection manifests, then stop (no training or eval)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    loader = get_loader(args.source)

    if loader.record_type is not QAExample:
        raise ValueError(f"SFT selection needs a QA source, got {args.source}")

    records = list(loader.load(args.split, limit=args.limit))
    tokenizer = load_tokenizer(args.model)
    tokens = token_costs(records, tokenizer, render=sft_render)
    print(f"pool: {len(records)} records, {sum(tokens):,} rendered tokens")

    anchor = select_top_k(get_scorer("random")(records), args.budget)
    token_budget = sum(tokens[i] for i in anchor)
    print(f"matched token budget: {token_budget:,} from a random {args.budget} record draw")

    scorer_names = list(args.scorers or [
        n for n in available_scorers() if n not in ("length",)
    ])

    print("scoring the pool with every scorer")
    selections: dict[str, list[int]] = {}
    for name in scorer_names:
        scorer = get_scorer(name, **SCORER_ARGS.get(name, {}))
        selections[name] = select_top_k(scorer(records), token_budget, costs=tokens)
        spent = sum(tokens[i] for i in selections[name])
        print(f"  {name}: {len(selections[name])} records, {spent:,} tokens")

    if args.score_only:
        for name in scorer_names:
            sel_path = runs_dir / name / "selection.json"
            write_selection(sel_path, records, selections[name], name, args.source, args.split)
            print(f"  wrote {sel_path}")
        print("score-only: selection manifests written, stopping before training")
        return 0

    report: dict[str, Any] = {
        "pool": {
            "source": args.source,
            "split": args.split,
            "n_candidates": len(records),
            "pool_tokens": sum(tokens),
        },
        "setup": {
            "model": args.model,
            "task": args.task,
            "token_budget": token_budget,
            "anchor_records": args.budget,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "seed": args.seed,
            "lora": True,
        },
        "runs": {},
    }

    if not args.skip_base:
        print("evaluating the untrained base model")
        report["runs"]["__base__"] = {
            "selection": None,
            "eval": evaluate_one(args.model, args),
        }

    for name in scorer_names:
        print(f"\n=== {name} ===")
        chosen = selections[name]
        spent = sum(tokens[i] for i in chosen)

        selection = runs_dir / name / "selection.json"
        write_selection(selection, records, chosen, name, args.source, args.split)

        training = train_one(name, selection, args, runs_dir / name / "model")
        print(f"{name}: trained in {training['train_runtime_s']}s")
        evaluation = evaluate_one(str(runs_dir / name / "model"), args)

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
    header = f"{'run':<18}{'records':>9}{'tokens':>11}{'accuracy':>10}{'95% interval':>20}"
    print(f"\n{header}")
    for name, entry in runs.items():
        accuracy = entry["eval"]["accuracy"]
        low, high = accuracy.get("accuracy_ci") or (float("nan"), float("nan"))
        selection = entry.get("selection") or {}
        print(
            f"{name:<18}{selection.get('n_selected', 0):>9}{selection.get('tokens', 0):>11,}"
            f"{accuracy['accuracy']:>10.4f}  [{low:.4f}, {high:.4f}]"
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
        print(f"  {name:<18}acc {delta:+.4f} ({verdict})")


if __name__ == "__main__":
    raise SystemExit(main())
