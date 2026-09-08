"""Integration test: EmbedSimilarityScorer with a real model (SmolLM2-135M)."""

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

TARGET = [
    "What drug inhibits COX?",
    "Name a risk factor for cardiovascular disease.",
    "Which organ metabolizes most drugs?",
]


@pytest.mark.slow
def test_embed_similarity_scores_in_range():
    from medsel.selection.scorers.embed_similarity import EmbedSimilarityScorer

    scorer = EmbedSimilarityScorer(
        model=MODEL, target=TARGET, progress=False,
    )
    scores = scorer.score(DOCS)
    assert len(scores) == len(DOCS)
    for s in scores:
        assert -1.0 <= s <= 1.0, f"cosine similarity should be in [-1, 1], got {s}"
