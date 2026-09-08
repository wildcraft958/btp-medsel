"""Analyze the SFT selection experiment results.

Reads the results JSON from run_sft_selection_experiment.py and produces:
  1. A comparison table (3DS vs LESS vs Random and all other scorers)
  2. Judge ablation summary (self-judged vs MedGemma-judged 3DS)
  3. Statistical significance checks against the base model
  4. A LaTeX-ready table for the report

    uv run python scripts/analyze_sft_experiment.py
    uv run python scripts/analyze_sft_experiment.py --ablation results/judge_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"Results file not found: {path}")
        sys.exit(1)
    return json.loads(p.read_text())


def ci_overlaps(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def print_comparison_table(report: dict) -> None:
    runs = report["runs"]
    setup = report["setup"]

    print("=" * 80)
    print(f"SFT Selection Experiment: {setup['model']}")
    print(f"Task: {setup['task']}  |  Token budget: {setup['token_budget']:,}")
    print(f"LoRA: r={16}, epochs={setup['epochs']}, lr={setup['learning_rate']}")
    print("=" * 80)

    header = (
        f"{'Scorer':<20}{'Records':>8}{'Tokens':>10}"
        f"{'Accuracy':>10}{'95% CI':>20}{'vs Base':>10}"
    )
    print(f"\n{header}")
    print("-" * 78)

    base_acc = None
    base_ci = None
    if "__base__" in runs:
        b = runs["__base__"]["eval"]["accuracy"]
        base_acc = b["accuracy"]
        base_ci = tuple(b.get("accuracy_ci", (0, 0)))
        print(
            f"{'base (no SFT)':<20}{'--':>8}{'--':>10}"
            f"{base_acc:>10.4f}  [{base_ci[0]:.4f}, {base_ci[1]:.4f}]{'--':>10}"
        )

    for name, entry in runs.items():
        if name == "__base__":
            continue
        sel = entry.get("selection", {})
        acc = entry["eval"]["accuracy"]
        ci = tuple(acc.get("accuracy_ci", (0, 0)))
        delta = ""
        if base_acc is not None:
            d = acc["accuracy"] - base_acc
            delta = f"{d:+.4f}"
        print(
            f"{name:<20}{sel.get('n_selected', 0):>8}{sel.get('tokens', 0):>10,}"
            f"{acc['accuracy']:>10.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]{delta:>10}"
        )

    if base_acc is None or base_ci is None:
        return

    print(f"\nBase CI width: {(base_ci[1] - base_ci[0]) * 100:.2f} percentage points")
    print("\nStatistical significance (non-overlapping CIs):")
    for name, entry in runs.items():
        if name == "__base__":
            continue
        acc = entry["eval"]["accuracy"]
        ci = tuple(acc.get("accuracy_ci", (0, 0)))
        overlaps = ci_overlaps(base_ci, ci)
        delta = acc["accuracy"] - base_acc
        status = "NOT significant" if overlaps else "SIGNIFICANT"
        direction = "improvement" if delta > 0 else "degradation"
        print(f"  {name:<18} {delta:+.4f} ({direction}) -- {status}")


def print_head_to_head(report: dict) -> None:
    runs = report["runs"]
    scorers = [n for n in runs if n != "__base__"]

    if len(scorers) < 2:
        return

    print("\n" + "=" * 80)
    print("Head-to-head (non-overlapping CIs)")
    print("=" * 80)

    for i, a in enumerate(scorers):
        ci_a = tuple(runs[a]["eval"]["accuracy"].get("accuracy_ci", (0, 0)))
        for b in scorers[i + 1 :]:
            ci_b = tuple(runs[b]["eval"]["accuracy"].get("accuracy_ci", (0, 0)))
            overlaps = ci_overlaps(ci_a, ci_b)
            acc_a = runs[a]["eval"]["accuracy"]["accuracy"]
            acc_b = runs[b]["eval"]["accuracy"]["accuracy"]
            diff = acc_a - acc_b
            winner = a if diff > 0 else b
            if overlaps:
                print(f"  {a} vs {b}: {diff:+.4f} -- NOT distinguishable")
            else:
                print(f"  {a} vs {b}: {diff:+.4f} -- {winner} wins")


def print_judge_ablation(main: dict, ablation: dict) -> None:
    print("\n" + "=" * 80)
    print("Judge Ablation: self-judged vs MedGemma-judged 3DS")
    print("=" * 80)

    main_3ds = main["runs"].get("3ds") or ablation["runs"].get("3ds")
    ablation_3ds = ablation["runs"].get("3ds_medgemma")

    if not main_3ds or not ablation_3ds:
        print("  Missing 3DS results in one or both files.")
        return

    main_acc = main_3ds["eval"]["accuracy"]
    abl_acc = ablation_3ds["eval"]["accuracy"]
    main_ci = tuple(main_acc.get("accuracy_ci", (0, 0)))
    abl_ci = tuple(abl_acc.get("accuracy_ci", (0, 0)))

    print(f"  {'Judge':<25}{'Accuracy':>10}{'95% CI':>22}")
    print(f"  {'Self (Qwen3-1.7B)':<25}{main_acc['accuracy']:>10.4f}"
          f"  [{main_ci[0]:.4f}, {main_ci[1]:.4f}]")
    print(f"  {'MedGemma':<25}{abl_acc['accuracy']:>10.4f}"
          f"  [{abl_ci[0]:.4f}, {abl_ci[1]:.4f}]")

    diff = abl_acc["accuracy"] - main_acc["accuracy"]
    overlaps = ci_overlaps(main_ci, abl_ci)
    print(f"\n  Delta: {diff:+.4f} ({'NOT significant' if overlaps else 'SIGNIFICANT'})")

    main_sel = main_3ds.get("selection", {})
    abl_sel = ablation_3ds.get("selection", {})
    print(f"  Self-judged selected: {main_sel.get('n_selected', '?')} records")
    print(f"  MedGemma-judged selected: {abl_sel.get('n_selected', '?')} records")


def print_latex_table(report: dict) -> None:
    runs = report["runs"]

    print("\n" + "=" * 80)
    print("LaTeX table (copy-paste ready)")
    print("=" * 80)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{SFT selection experiment on MedMCQA}")
    print(r"\begin{tabular}{lrrcc}")
    print(r"\toprule")
    print(r"Scorer & Records & Tokens & Accuracy & 95\% CI \\")
    print(r"\midrule")

    for name, entry in runs.items():
        label = "Base (no SFT)" if name == "__base__" else name.upper()
        sel = entry.get("selection") or {}
        acc = entry["eval"]["accuracy"]
        ci = acc.get("accuracy_ci", (0, 0))
        recs = sel.get("n_selected", "--")
        toks = f"{sel.get('tokens', 0):,}" if sel.get("tokens") else "--"
        print(
            f"{label} & {recs} & {toks} & "
            f"{acc['accuracy']:.4f} & [{ci[0]:.4f}, {ci[1]:.4f}] \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/sft_selection_experiment.json",
        help="main experiment results file",
    )
    parser.add_argument(
        "--ablation",
        default=None,
        help="judge ablation results file (optional)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="also print a LaTeX table",
    )
    args = parser.parse_args()

    report = load_results(args.results)
    print_comparison_table(report)
    print_head_to_head(report)

    if args.ablation:
        ablation = load_results(args.ablation)
        print_judge_ablation(report, ablation)

    if args.latex:
        print_latex_table(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
