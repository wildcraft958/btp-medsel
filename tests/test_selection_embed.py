"""Embedding similarity to a target centroid, and the guard it shares with dsir."""

import numpy as np
import pytest

from medsel.selection.base import available_scorers, get_scorer
from medsel.selection.scorers.embed_similarity import unit_centroid

TEXTS = ["one", "two", "three"]


class TestUnitCentroid:
    def test_averages_then_normalises(self):
        result = unit_centroid(np.array([[3.0, 0.0], [0.0, 3.0]]))
        assert result == pytest.approx([2**-0.5, 2**-0.5])

    def test_is_unit_length(self):
        result = unit_centroid(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert float(np.linalg.norm(result)) == pytest.approx(1.0)

    def test_single_vector_normalises_to_itself(self):
        assert unit_centroid(np.array([[0.0, 5.0]])) == pytest.approx([0.0, 1.0])

    def test_rejects_an_empty_target(self):
        with pytest.raises(ValueError, match="no embeddings"):
            unit_centroid(np.zeros((0, 4)))

    def test_rejects_a_degenerate_centroid(self):
        # Opposed vectors average to the origin, which has no direction to compare against.
        with pytest.raises(ValueError, match="cancel"):
            unit_centroid(np.array([[1.0, 0.0], [-1.0, 0.0]]))


class TestEmbedSimilarityScorer:
    def test_registered_under_its_name(self):
        assert "embed_similarity" in available_scorers()

    def test_rejects_a_nonpositive_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            get_scorer("embed_similarity", batch_size=0)

    def test_empty_pool_needs_no_model(self):
        assert get_scorer("embed_similarity", model="does-not-exist/nope")([]) == []

    def test_accepts_explicit_target_texts(self):
        assert get_scorer("embed_similarity", target=TEXTS).target == TEXTS


class TestSharedTargetGuard:
    """Both targeted scorers must refuse a target that overlaps the evaluation split."""

    @pytest.mark.parametrize("scorer", ["dsir", "embed_similarity"])
    @pytest.mark.parametrize(
        ("task", "expected"),
        [("medqa", "train"), ("medmcqa", "train"), ("pubmedqa", "train[500:]")],
    )
    def test_defaults_to_a_training_split(self, scorer, task, expected):
        assert get_scorer(scorer, target=task).resolve_target_split() == expected

    @pytest.mark.parametrize("scorer", ["dsir", "embed_similarity"])
    @pytest.mark.parametrize(
        ("task", "split"),
        [("medmcqa", "validation"), ("medqa", "test"), ("pubmedqa", "train[:500]")],
    )
    def test_rejects_the_evaluation_split(self, scorer, task, split):
        with pytest.raises(ValueError, match="scored on"):
            get_scorer(scorer, target=task, target_split=split).resolve_target_split()

    @pytest.mark.parametrize("scorer", ["dsir", "embed_similarity"])
    def test_explicit_texts_need_no_split(self, scorer):
        assert get_scorer(scorer, target=TEXTS).resolve_target_split() is None
