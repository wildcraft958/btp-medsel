"""Integration test: 3DS scorer with a real model (SmolLM2-135M)."""

import numpy as np
import pytest

from medsel.schema import QAExample

MODEL = "HuggingFaceTB/SmolLM2-135M"


def _example(uid: str, question: str, answer_key: str = "A") -> QAExample:
    return QAExample(
        uid=uid,
        source="test",
        split="train",
        question=question,
        options={
            "A": "Inhibition of cyclooxygenase",
            "B": "Activation of prostaglandins",
            "C": "Inhibition of lipoxygenase",
            "D": "Activation of thromboxane",
        },
        answer_key=answer_key,
    )


POOL = [
    _example("t/0", "What is the mechanism of action of aspirin?"),
    _example("t/1", "Which enzyme does ibuprofen inhibit?"),
    _example("t/2", "What is the primary site of drug metabolism?", "B"),
    _example("t/3", "Which receptor does atropine block?", "C"),
    _example("t/4", "What neurotransmitter is deficient in Parkinson disease?", "D"),
]


@pytest.mark.slow
def test_3ds_scores_are_finite(tmp_path):
    from medsel.selection.scorers.tds import ThreeDSScorer

    scorer = ThreeDSScorer(
        judge=MODEL,
        cache_dir=str(tmp_path / "cache"),
        low_th=0.0,
        up_th=100.0,
        progress=False,
    )
    scores = scorer.score(POOL)
    assert len(scores) == len(POOL)
    finite = [s for s in scores if s != float("-inf")]
    assert len(finite) > 0, "at least some examples should survive quality + Goldilocks"
    for s in finite:
        assert np.isfinite(s), f"score should be finite, got {s}"
    scorer.release_model()


@pytest.mark.slow
def test_3ds_cache_persists_embeddings(tmp_path):
    from medsel.selection.scorers.tds import ThreeDSScorer

    cache = tmp_path / "cache"
    scorer = ThreeDSScorer(
        judge=MODEL,
        cache_dir=str(cache),
        low_th=0.0,
        up_th=100.0,
        progress=False,
    )
    scorer.score(POOL[:2])
    scorer.release_model()

    emb_dir = cache / "embeddings"
    assert emb_dir.exists(), "embeddings directory should be created"
    npy_files = list(emb_dir.glob("*.npy"))
    assert len(npy_files) > 0, "should have at least one .npy embedding"
    emb = np.load(npy_files[0])
    assert emb.ndim == 1
    assert emb.shape[0] > 0
