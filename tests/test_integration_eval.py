"""Integration test: evaluation pipeline with a real model (SmolLM2-135M)."""

import pytest

MODEL = "HuggingFaceTB/SmolLM2-135M"


@pytest.mark.slow
def test_evaluate_medmcqa_sample():
    from medsel.eval.runner import evaluate_task, load_model

    model, tokenizer = load_model(MODEL)
    report = evaluate_task(
        model, tokenizer, "medmcqa", MODEL, limit=10, progress=False
    )
    assert 0.0 <= report.accuracy <= 1.0
    assert report.n_scored == 10
    assert isinstance(report.by_subject, dict)
    assert len(report.by_subject) > 0

    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
