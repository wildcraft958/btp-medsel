"""Command line entry point.

``argparse`` rather than a CLI framework: the surface is small, and a shared research repo is
better off with zero avoidable dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medsel import __version__

__all__ = ["main", "build_parser", "build_manifest", "score_summary"]


def _coerce(value: str) -> Any:
    """Turn a ``key=value`` string into the obvious Python type."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _parse_kwargs(pairs: list[str] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        kwargs[key.strip()] = _coerce(value.strip())
    return kwargs


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_data_list(args: argparse.Namespace) -> int:
    from medsel.registry import available_loaders, get_loader_class

    rows = []
    for name in available_loaders():
        cls = get_loader_class(name)
        rows.append(
            {
                "name": name,
                "hf_id": cls.hf_id,
                "record_type": cls.record_type.__name__,
                "splits": list(cls.splits),
                "unlabeled_splits": sorted(cls.unlabeled_splits),
            }
        )
    _emit(rows)
    return 0


def cmd_data_stats(args: argparse.Namespace) -> int:
    from medsel.registry import get_loader

    loader = get_loader(args.source, **_parse_kwargs(args.loader))
    split = args.split or getattr(loader, "eval_split", None) or "train"

    stats = loader.stats(split, limit=args.limit)
    if not loader.has_labels(split):
        stats["warning"] = (
            f"{args.source}:{split} ships no gold labels; use "
            f"{getattr(loader, 'eval_split', 'a labelled split')} for evaluation"
        )
    _emit(stats)
    return 0


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from medsel.data.cache import cache_is_valid, write_cache
    from medsel.registry import get_loader

    loader = get_loader(args.source, **_parse_kwargs(args.loader))
    split = args.split or "train"

    if not args.force and cache_is_valid(args.source, split, loader.cache_key, loader.record_type):
        _emit({"status": "up-to-date", "source": args.source, "split": split})
        return 0

    path = write_cache(
        loader.load(split, limit=args.limit),
        source=args.source,
        split=split,
        cache_key=loader.cache_key,
        record_type=loader.record_type,
        extra={"hf_id": loader.hf_id, "config": loader.config},
    )
    _emit({"status": "written", "path": str(path)})
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from medsel.config import load_experiment
    from medsel.stages import get_stage

    config = load_experiment(args.config)
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.limit is not None:
        config.data.limit = args.limit
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps

    stage = get_stage(config.stage)(config)
    result = stage.run()
    _emit(
        {
            "stage": result.stage,
            "output_dir": str(result.output_dir),
            "manifest": str(result.manifest_path),
            "metrics": result.metrics,
        }
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from medsel.eval.runner import run_evaluation

    payload = run_evaluation(
        args.model,
        tasks=args.tasks,
        split=args.split,
        limit=args.limit,
        mode=args.mode,
        dtype=args.dtype,
        seed=args.seed,
        output=args.output,
    )
    _emit(payload)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from medsel.registry import available_loaders
    from medsel.stages import available_stages
    from medsel.utils.device import describe_device

    _emit(
        {
            "medsel": __version__,
            "python": sys.version.split()[0],
            "hardware": describe_device(),
            "loaders": available_loaders(),
            "stages": available_stages(),
        }
    )
    return 0


def _budget(value: str) -> int | float:
    """A whole number is a count of records; a decimal is a fraction of the pool.

    The distinction carries the meaning, so ``--budget 50`` keeps fifty records and
    ``--budget 0.5`` keeps half. Parsing everything as a float would make the first of those an
    out-of-range fraction.
    """
    text = value.strip()
    return int(text) if text.lstrip("+-").isdigit() else float(text)


def score_summary(scores: list[float], chosen: list[int]) -> dict[str, float]:
    """Where the pool sat and where the kept records sat, on the same scale."""
    if not scores:
        return {}
    summary = {"min": min(scores), "max": max(scores), "mean": sum(scores) / len(scores)}
    if chosen:
        picked = [scores[i] for i in chosen]
        summary["selected_min"] = min(picked)
        summary["selected_max"] = max(picked)
        summary["selected_mean"] = sum(picked) / len(picked)
    return summary


def build_manifest(
    *,
    source: str,
    split: str,
    scorer: str,
    scorer_args: dict[str, Any],
    budget: int | float,
    strategy: str,
    n_candidates: int,
    selected_uids: list[str],
    summary: dict[str, float],
) -> dict[str, Any]:
    """Describe a selection completely enough to reproduce and to audit it.

    The uids rather than the records: a manifest is meant to be committed and read, and a training
    run resolves it back through the loader. Score summaries are included because the first
    question asked of any selection is whether the scorer actually discriminated, and a flat
    distribution says the answer is no.
    """
    return {
        "source": source,
        "split": split,
        "scorer": scorer,
        "scorer_args": scorer_args,
        "budget": budget,
        "strategy": strategy,
        "n_candidates": n_candidates,
        "n_selected": len(selected_uids),
        "score_summary": summary,
        "selected_uids": selected_uids,
    }


def cmd_select(args: argparse.Namespace) -> int:
    from medsel.registry import get_loader
    from medsel.selection.base import get_scorer
    from medsel.selection.selector import select_stratified, select_top_k

    loader = get_loader(args.source, **_parse_kwargs(args.loader))
    split = args.split or getattr(loader, "eval_split", None) or "train"
    scorer_args = _parse_kwargs(args.scorer_arg)
    scorer = get_scorer(args.scorer, **scorer_args)
    strategy = f"stratified:{args.stratify_by}" if args.stratify_by else "top_k"

    if args.stream:
        from medsel.selection.stream import select_stream

        result = select_stream(
            loader.load(split, limit=args.limit, progress=False),
            scorer,
            args.budget,
            fit_size=args.fit_size,
            chunk_size=args.chunk_size,
            stratify_by=args.stratify_by,
            min_per_group=args.min_per_group,
        )
        n_candidates, selected_uids, summary = (
            result.n_candidates,
            result.uids,
            result.score_summary,
        )
    else:
        records = list(loader.load(split, limit=args.limit))
        scores = scorer(records)
        if args.stratify_by:
            groups = [
                str(getattr(r, "labels", {}).get(args.stratify_by, "unknown")) for r in records
            ]
            chosen = select_stratified(scores, groups, args.budget, args.min_per_group)
        else:
            chosen = select_top_k(scores, args.budget)
        n_candidates = len(records)
        selected_uids = [records[i].uid for i in chosen]
        summary = score_summary(scores, chosen)

    manifest = build_manifest(
        source=args.source,
        split=split,
        scorer=args.scorer,
        scorer_args=scorer_args,
        budget=args.budget,
        strategy=strategy,
        n_candidates=n_candidates,
        selected_uids=selected_uids,
        summary=summary,
    )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest = {**manifest, "written_to": str(path)}
        manifest.pop("selected_uids", None)

    _emit(manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medsel", description=__doc__)
    parser.add_argument("--version", action="version", version=f"medsel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show environment, loaders and stages")
    info.set_defaults(func=cmd_info)

    data = sub.add_parser("data", help="dataset inspection and caching")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    listing = data_sub.add_parser("list", help="list registered loaders")
    listing.set_defaults(func=cmd_data_list)

    for name, handler, help_text in (
        ("stats", cmd_data_stats, "summarise a split"),
        ("prepare", cmd_data_prepare, "normalise a split into the parquet cache"),
    ):
        cmd = data_sub.add_parser(name, help=help_text)
        cmd.add_argument("--source", required=True, help="loader name, e.g. medmcqa")
        cmd.add_argument("--split", help="defaults to the loader's evaluation split")
        cmd.add_argument("--limit", type=int, help="stop after N records")
        cmd.add_argument(
            "--loader",
            nargs="*",
            metavar="KEY=VALUE",
            help="loader kwargs, e.g. num_shards=2 min_chars=300",
        )
        if name == "prepare":
            cmd.add_argument("--force", action="store_true", help="rebuild even if cache is valid")
        cmd.set_defaults(func=handler)

    train = sub.add_parser("train", help="run a training stage from an experiment config")
    train.add_argument("--config", required=True, help="path to configs/experiment/*.yaml")
    train.add_argument("--output-dir", help="override train.output_dir")
    train.add_argument("--limit", type=int, help="override data.limit")
    train.add_argument("--max-steps", type=int, help="override train.max_steps")
    train.set_defaults(func=cmd_train)

    evaluation = sub.add_parser("eval", help="score a checkpoint on the QA benchmarks")
    evaluation.add_argument("--model", required=True, help="HF id or local checkpoint path")
    evaluation.add_argument(
        "--tasks",
        nargs="+",
        default=["medmcqa"],
        metavar="TASK",
        help="loader names to evaluate, e.g. medqa medmcqa pubmedqa",
    )
    evaluation.add_argument("--split", help="defaults to each task's labelled split")
    evaluation.add_argument("--limit", type=int, help="score only the first N examples")
    evaluation.add_argument(
        "--mode",
        default="letter",
        choices=["letter", "text"],
        help=(
            "score the option letter (published convention) or its wording (kinder to base models)"
        ),
    )
    evaluation.add_argument("--dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--output", help="write the report JSON here")
    evaluation.set_defaults(func=cmd_eval)

    selection = sub.add_parser("select", help="score a pool and choose a subset under a budget")
    selection.add_argument("--source", required=True, help="loader name, e.g. pubmed")
    selection.add_argument("--scorer", default="random", help="scorer name, e.g. dsir perplexity")
    selection.add_argument(
        "--budget",
        type=_budget,
        default=0.1,
        help="fraction of the pool to keep (0.1), or a whole number of records (5000)",
    )
    selection.add_argument("--split", help="defaults to the loader's evaluation split")
    selection.add_argument("--limit", type=int, help="score only the first N records")
    selection.add_argument(
        "--loader", nargs="*", metavar="KEY=VALUE", help="loader kwargs, e.g. num_shards=1"
    )
    selection.add_argument(
        "--scorer-arg",
        nargs="*",
        metavar="KEY=VALUE",
        help="scorer kwargs, e.g. target=medmcqa mode=mid",
    )
    selection.add_argument(
        "--stratify-by",
        help="label key to allocate the budget across, e.g. subject_name",
    )
    selection.add_argument(
        "--min-per-group",
        type=int,
        default=0,
        help="floor of records per group, the capability-retention lever",
    )
    selection.add_argument(
        "--stream",
        action="store_true",
        help="score in chunks and keep a bounded heap, for pools too large to hold in memory",
    )
    selection.add_argument(
        "--fit-size",
        type=int,
        default=10_000,
        help="records used to fit pool-relative scorers when streaming",
    )
    selection.add_argument(
        "--chunk-size", type=int, default=1_000, help="records scored at once when streaming"
    )
    selection.add_argument("--output", help="write the selection manifest JSON here")
    selection.set_defaults(func=cmd_select)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
