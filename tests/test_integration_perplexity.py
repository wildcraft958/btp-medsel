"""Integration test: PerplexityScorer with a real model (SmolLM2-135M)."""

import numpy as np
import pytest

from medsel.schema import CorpusDoc

MODEL = "HuggingFaceTB/SmolLM2-135M"

DOCS = [
    CorpusDoc(uid="d0", source="test", text="Aspirin inhibits cyclooxygenase enzymes."),
    CorpusDoc(uid="d1", source="test", text="The mitochondria is the powerhouse of the cell."),
    CorpusDoc(uid="d2", source="test", text="Hypertension is a major risk factor for stroke."),
    CorpusDoc(uid="d3", source="test", text="Penicillin was discovered by Alexander Fleming."),
    CorpusDoc(uid="d4", source="test", text="The liver is the primary site of drug metabolism."),
]


@pytest.mark.slow
def test_perplexity_scores_finite():
    from medsel.selection.scorers.perplexity import PerplexityScorer

    scorer = PerplexityScorer(model=MODEL, mode="mid", progress=False)
    scores = scorer.score(DOCS)
    assert len(scores) == len(DOCS)
    for s in scores:
        assert np.isfinite(s), f"NLL score should be finite, got {s}"
