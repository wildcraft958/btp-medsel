"""Train and evaluate pre-scored selection manifests.

Reads selection manifests from runs/sft-selection-experiment/<scorer>/selection.json,
trains one LoRA SFT model per scorer, and evaluates each on MedMCQA validation.

This is the Phase 3 script: scoring is done, selections are written, now train and eval.

    .venv/bin/python scripts/train_and_eval_selections.py
    .venv/bin/python scripts/train_and_eval_selections.py --scorers 3ds random less
    .venv/bin/python scripts/train_and_eval_selections.py --skip-base --scorers 3ds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from medsel.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from medsel.stages.sft import SFTStage


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
    scorer: str,
    selection_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    config = ExperimentConfig(
        name=f"sft-select-{scorer}",
        stage="sft",
        data=DataConfig(
            source=args.source,
            split=args.split,
            limit=args.limit,
            selection=str(selection_path),
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
        notes=f"SFT from pre-scored selection, scorer={scorer}.",
    )
    started = time.time()
    result = SFTStage(config).run()
    return {
        "train_runtime_s": round(time.time() - started, 1),
        "n_examples": result.metrics.get("n_examples"),
        "train_loss": result.metrics.get("train_loss"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="medmcqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=30000)
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
    parser.add_argument("--scorers", nargs="*", default=None)
    parser.add_argument("--output", default="results/sft_selection_experiment.json")
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    if args.scorers:
        scorer_names = args.scorers
    else:
        scorer_names = sorted(
            d.name for d in runs_dir.iterdir()
            if d.is_dir() and (d / "selection.json").exists()
        )

    print(f"Found selections for: {scorer_names}")
    for name in scorer_names:
        sel_path = runs_dir / name / "selection.json"
        if not sel_path.exists():
            print(f"  WARNING: {sel_path} not found, skipping {name}")
            scorer_names = [n for n in scorer_names if n != name]
            continue
        manifest = json.loads(sel_path.read_text())
        print(f"  {name}: {manifest['n_selected']} records from {manifest['n_candidates']}")

    if not scorer_names:
        print("No selections found. Run scoring first.")
        return 1

    report: dict[str, Any] = {
        "setup": {
            "model": args.model,
            "task": args.task,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "seed": args.seed,
            "lora": True,
        },
        "runs": {},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_base:
        print("\n=== base (no SFT) ===")
        report["runs"]["__base__"] = {
            "selection": None,
            "eval": evaluate_one(args.model, args.task, args.eval_limit),
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        base_acc = report["runs"]["__base__"]["eval"]["accuracy"]["accuracy"]
        print(f"  base accuracy: {base_acc:.4f}")

    for name in scorer_names:
        print(f"\n=== {name} ===")
        sel_path = runs_dir / name / "selection.json"
        manifest = json.loads(sel_path.read_text())
        model_dir = runs_dir / name / "model"

        if (model_dir / "adapter_model.safetensors").exists():
            print(f"  model already trained at {model_dir}, skipping training")
            training = {"train_runtime_s": 0, "n_examples": manifest["n_selected"]}
        else:
            training = train_one(name, sel_path, args, model_dir)
            print(f"  trained in {training['train_runtime_s']}s")

        evaluation = evaluate_one(str(model_dir), args.task, args.eval_limit)
        acc = evaluation["accuracy"]["accuracy"]
        print(f"  accuracy: {acc:.4f}")

        report["runs"][name] = {
            "selection": {
                "n_selected": manifest["n_selected"],
                "tokens": manifest.get("tokens"),
                "scorer": name,
            },
            "training": training,
            "eval": evaluation,
        }

        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  saved to {out_path}")

    print(f"\nAll done. Results in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
