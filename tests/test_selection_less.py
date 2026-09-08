"""Tests for the LESS (Selecting Influential Data) scorer."""

import pytest
import torch

from medsel.schema import QAExample
from medsel.selection.base import get_scorer


def _example(
    uid: str = "test/train/000001",
    question: str = "What is the mechanism of action of aspirin?",
    answer_key: str = "A",
) -> QAExample:
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


class TestRademacherProject:
    def test_output_dimension(self):
        from medsel.selection.scorers.less import _rademacher_project

        grad = torch.randn(1000)
        proj = _rademacher_project(grad, proj_dim=128, seed=42)
        assert proj.shape == (128,)

    def test_deterministic_with_same_seed(self):
        from medsel.selection.scorers.less import _rademacher_project

        grad = torch.randn(500)
        a = _rademacher_project(grad, proj_dim=64, seed=7)
        b = _rademacher_project(grad, proj_dim=64, seed=7)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        from medsel.selection.scorers.less import _rademacher_project

        grad = torch.randn(500)
        a = _rademacher_project(grad, proj_dim=64, seed=1)
        b = _rademacher_project(grad, proj_dim=64, seed=2)
        assert not torch.allclose(a, b)

    def test_preserves_dot_product_approximately(self):
        """Johnson-Lindenstrauss: projected dot product approximates original."""
        from medsel.selection.scorers.less import _rademacher_project

        torch.manual_seed(0)
        a = torch.randn(10000)
        b = torch.randn(10000)
        true_dot = (a @ b).item()

        proj_a = _rademacher_project(a, proj_dim=4096, seed=42)
        proj_b = _rademacher_project(b, proj_dim=4096, seed=42)
        approx_dot = (proj_a @ proj_b).item()

        assert abs(approx_dot - true_dot) / (abs(true_dot) + 1e-8) < 0.7

    def test_zero_grad_gives_zero(self):
        from medsel.selection.scorers.less import _rademacher_project

        grad = torch.zeros(1000)
        proj = _rademacher_project(grad, proj_dim=128, seed=42)
        assert torch.allclose(proj, torch.zeros(128))


class TestAdamCorrect:
    def test_basic_correction(self):
        from medsel.selection.scorers.less import _adam_correct

        grad = torch.tensor([1.0, -2.0, 3.0])
        m = torch.zeros(3)
        v = torch.zeros(3)
        corrected = _adam_correct(grad, m, v)
        assert corrected.shape == (3,)
        assert torch.all(torch.isfinite(corrected))

    def test_zero_grad_gives_zero(self):
        from medsel.selection.scorers.less import _adam_correct

        grad = torch.zeros(5)
        m = torch.zeros(5)
        v = torch.zeros(5)
        corrected = _adam_correct(grad, m, v)
        assert torch.allclose(corrected, torch.zeros(5))

    def test_sign_preserved(self):
        from medsel.selection.scorers.less import _adam_correct

        grad = torch.tensor([1.0, -1.0])
        m = torch.zeros(2)
        v = torch.zeros(2)
        corrected = _adam_correct(grad, m, v)
        assert corrected[0] > 0
        assert corrected[1] < 0


class TestLESSScorerInterface:
    def test_registered_under_less(self):
        scorer = get_scorer("less", judge="ignored")
        assert scorer.name == "less"

    def test_empty_pool_returns_empty(self):
        from medsel.selection.scorers.less import LESSScorer

        scorer = LESSScorer(judge="ignored")
        assert scorer.score([]) == []

    def test_rejects_invalid_proj_dim(self):
        from medsel.selection.scorers.less import LESSScorer

        with pytest.raises(ValueError, match="proj_dim"):
            LESSScorer(judge="x", proj_dim=0)

    def test_repr(self):
        from medsel.selection.scorers.less import LESSScorer

        s = LESSScorer(judge="model/name", proj_dim=4096)
        r = repr(s)
        assert "model/name" in r
        assert "4096" in r
