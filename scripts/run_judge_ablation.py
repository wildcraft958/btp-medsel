"""Phase 4: Judge ablation -- compare 3DS self-judged vs MedGemma-judged.

Reads the 3DS cache (self-judged, Qwen) and the 3ds_medgemma cache (MedGemma-judged),
creates selection manifests from each at the same token budget, trains one SFT model
per judge, evaluates both, and writes the ablation results.

    .venv/bin/python scripts/run_judge_ablation.py --limit 30000
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
from medsel.selection.base import get_scorer
from medsel.selection.costs import load_tokenizer, token_costs
from medsel.selection.selector import select_top_k
from medsel.stages.sft import SFTStage, render_completion


def sft_render(example: Any) -> str:
    return render_prompt(example) + render_completion(example)


def evaluate_one(model_path: str, task: str, eval_limit: int | None) -> dict[str, Any]:
    from medsel.eval.runner import evaluate_task, load_model

    model, tokenizer = load_model(model_path)
    accuracy = evaluate_task(model, tokenizer, task, model_path, limit=eval_limit, progress=True)
    payload: dict[str, Any] = {"accuracy": accuracy.to_dict()}
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return payload


def train_one(
    name: str, selection_path: Path, args: argparse.Namespace, out_dir: Path,
) -> dict[str, Any]:
    config = ExperimentConfig(
        name=f"sft-ablation-{name}",
        stage="sft",
        data=DataConfig(
            source=args.source, split=args.split, limit=args.limit,
            selection=str(selection_path),
        ),
        model=ModelConfig(
            name_or_path=args.model, dtype="auto",
            use_lora=True, lora_r=16, lora_alpha=32,
            gradient_checkpointing=True,
        ),
        train=TrainConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            logging_steps=20, save_steps=10**9,
            seed=args.seed, max_length=args.max_length,
        ),
        notes=f"Judge ablation, judge={name}.",
    )
    started = time.time()
    result = SFTStage(config).run()
    return {
        "train_runtime_s": round(time.time() - started, 1),
        "n_examples": result.metrics.get("n_examples"),
        "train_loss": result.metrics.get("train_loss"),
    }


def write_selection(
    path: Path, records: list[Any], chosen: list[int],
    scorer: str, source: str, split: str, tokens: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spent = sum(tokens[i] for i in chosen)
    path.write_text(
        json.dumps({
            "source": source, "split": split, "scorer": scorer,
            "n_candidates": len(records), "n_selected": len(chosen),
            "tokens": spent,
            "selected_uids": [records[i].uid for i in chosen],
        }, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--task", default="medmcqa")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="runs/sft-selection-experiment")
    parser.add_argument("--output", default="results/judge_ablation.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    self_cache = runs_dir / "3ds" / "cache"
    medgemma_cache = runs_dir / "3ds_medgemma" / "cache"

    for name, cache_dir in [("3ds (self)", self_cache), ("3ds_medgemma", medgemma_cache)]:
        cache_file = cache_dir / "tds_cache.jsonl"
        if not cache_file.exists():
            print(f"ERROR: {name} cache not found at {cache_file}")
            return 1
        n = sum(1 for _ in cache_file.open())
        print(f"  {name}: {n} cached entries")

    loader = get_loader(args.source)
    records = list(loader.load(args.split, limit=args.limit))
    tokenizer = load_tokenizer(args.model)
    tokens = token_costs(records, tokenizer, render=sft_render)

    anchor = select_top_k(get_scorer("random")(records), args.budget)
    token_budget = sum(tokens[i] for i in anchor)
    print(f"pool: {len(records)} records, token budget: {token_budget:,}")

    report: dict[str, Any] = {
        "setup": {
            "model": args.model, "task": args.task,
            "token_budget": token_budget,
            "self_judge": "Qwen/Qwen3-1.7B-Base",
            "medgemma_judge": "google/medgemma-1.5-4b-it (4-bit NF4)",
        },
        "runs": {},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judges = [
        ("3ds", self_cache),
        ("3ds_medgemma", medgemma_cache),
    ]

    for judge_name, cache_dir in judges:
        print(f"\n=== {judge_name} ===")

        from medsel.selection.scorers.tds import ThreeDSScorer
        scorer = ThreeDSScorer(cache_dir=str(cache_dir))
        scores = scorer.score(records)
        chosen = select_top_k(scores, token_budget, costs=tokens)
        spent = sum(tokens[i] for i in chosen)
        print(f"  selected {len(chosen)} records, {spent:,} tokens")

        sel_path = runs_dir / judge_name / "selection.json"
        write_selection(sel_path, records, chosen, judge_name, args.source, args.split, tokens)

        model_dir = runs_dir / judge_name / "model"
        if (model_dir / "adapter_model.safetensors").exists():
            print(f"  model exists at {model_dir}, skipping training")
            training = {"train_runtime_s": 0}
        else:
            training = train_one(judge_name, sel_path, args, model_dir)
            print(f"  trained in {training['train_runtime_s']}s")

        evaluation = evaluate_one(str(model_dir), args.task, args.eval_limit)
        acc = evaluation["accuracy"]["accuracy"]
        ci = evaluation["accuracy"].get("accuracy_ci", [0, 0])
        print(f"  accuracy: {acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

        report["runs"][judge_name] = {
            "selection": {"n_selected": len(chosen), "tokens": spent},
            "training": training,
            "eval": evaluation,
        }

        out_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== Judge Ablation Summary ===")
    for name, entry in report["runs"].items():
        acc = entry["eval"]["accuracy"]
        ci = acc.get("accuracy_ci", [0, 0])
        print(f"  {name}: {acc['accuracy']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    r = report["runs"]
    if "3ds" in r and "3ds_medgemma" in r:
        delta = r["3ds_medgemma"]["eval"]["accuracy"]["accuracy"] - r["3ds"]["eval"]["accuracy"]["accuracy"]
        print(f"  delta (MedGemma - self): {delta:+.4f}")

    print(f"\nResults: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
