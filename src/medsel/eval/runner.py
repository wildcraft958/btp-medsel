"""Wire a checkpoint to the loaders and produce evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medsel.eval.mcq import EvalReport, evaluate
from medsel.registry import get_loader
from medsel.utils.device import pick_device, resolve_dtype
from medsel.utils.seed import set_seed

__all__ = ["load_model", "evaluate_task", "run_evaluation"]

# Splits that actually carry gold answers. MedMCQA's published `test` does not.
DEFAULT_EVAL_SPLITS = {
    "medqa": "test",
    "medmcqa": "validation",
    "pubmedqa": "train",
}


def load_model(
    name_or_path: str,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    quantization: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer onto the best available device.

    ``quantization`` accepts ``"4bit"`` or ``"8bit"`` and uses bitsandbytes NF4/INT8.
    When quantized, the model is loaded directly onto the accelerator (no ``.to()``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = pick_device()
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_dtype(dtype, device),
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = resolve_dtype(dtype, device)

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    if quantization is None:
        model = model.to(device)
    return model, tokenizer


def evaluate_task(
    model: Any,
    tokenizer: Any,
    task: str,
    model_name: str,
    split: str | None = None,
    limit: int | None = None,
    mode: str = "letter",
    loader_kwargs: dict[str, Any] | None = None,
    progress: bool = True,
) -> EvalReport:
    """Evaluate one dataset, defaulting to the split that has labels."""
    loader = get_loader(task, **(loader_kwargs or {}))
    resolved: str = (
        split or str(getattr(loader, "eval_split", "")) or DEFAULT_EVAL_SPLITS.get(task, "test")
    )

    if not loader.has_labels(resolved.split("[", 1)[0]):
        raise ValueError(
            f"{task}:{resolved} has no gold labels. "
            f"Use {DEFAULT_EVAL_SPLITS.get(task, 'a labelled split')} instead."
        )

    examples = loader.load(resolved, limit=limit, progress=False)
    return evaluate(
        model,
        tokenizer,
        examples,
        source=task,
        split=resolved,
        model_name=model_name,
        mode=mode,
        progress=progress,
        total=limit,
    )


def run_evaluation(
    model_name_or_path: str,
    tasks: list[str],
    split: str | None = None,
    limit: int | None = None,
    mode: str = "letter",
    dtype: str = "auto",
    seed: int = 42,
    output: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate a checkpoint across several datasets and optionally write the report to disk."""
    set_seed(seed)
    model, tokenizer = load_model(model_name_or_path, dtype=dtype)

    reports = {}
    for task in tasks:
        report = evaluate_task(
            model,
            tokenizer,
            task,
            model_name=model_name_or_path,
            split=split,
            limit=limit,
            mode=mode,
            progress=progress,
        )
        reports[task] = report.to_dict()

    payload = {
        "model": model_name_or_path,
        "mode": mode,
        "device": pick_device(),
        "results": reports,
    }

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        payload["written_to"] = str(path)

    return payload
