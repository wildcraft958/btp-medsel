"""Evaluation."""

from medsel.eval.mcq import EvalReport, OptionScores, evaluate, score_example
from medsel.eval.runner import evaluate_task, load_model, run_evaluation

__all__ = [
    "EvalReport",
    "OptionScores",
    "evaluate",
    "score_example",
    "evaluate_task",
    "load_model",
    "run_evaluation",
]
