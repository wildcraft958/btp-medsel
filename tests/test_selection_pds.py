"""Tests for the PDS (PMP-based Data Selection) scorer."""

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


class TestPMPConfig:
    def test_default_checkpoint_interval(self):
        from medsel.selection.scorers.pds_solver import PMPConfig

        config = PMPConfig(pmp_steps=100)
        assert config.checkpoint_interval == 10

    def test_custom_checkpoint_interval(self):
        from medsel.selection.scorers.pds_solver import PMPConfig

        config = PMPConfig(pmp_steps=100, checkpoint_interval=5)
        assert config.checkpoint_interval == 5

    def test_sqrt_for_nonstandard_steps(self):
        from medsel.selection.scorers.pds_solver import PMPConfig

        config = PMPConfig(pmp_steps=64)
        assert config.checkpoint_interval == 8


class TestCheckpointManager:
    def test_save_and_load_roundtrip(self, tmp_path):
        from medsel.selection.scorers.pds_solver import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpts")
        params = {"w": torch.randn(3, 4)}
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        mgr.save_checkpoint(0, params, batch)
        loaded_params, loaded_batch = mgr.load_checkpoint(0)
        assert torch.allclose(params["w"], loaded_params["w"])
        assert torch.equal(batch["input_ids"], loaded_batch["input_ids"])

    def test_checkpoint_steps_sorted(self, tmp_path):
        from medsel.selection.scorers.pds_solver import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpts")
        for step in [20, 0, 10]:
            mgr.save_checkpoint(step, {"w": torch.zeros(1)}, {"x": torch.zeros(1)})
        assert mgr.checkpoint_steps() == [0, 10, 20]

    def test_is_complete(self, tmp_path):
        from medsel.selection.scorers.pds_solver import CheckpointManager

        mgr = CheckpointManager(tmp_path / "ckpts")
        mgr.save_checkpoint(0, {"w": torch.zeros(1)}, {"x": torch.zeros(1)})
        mgr.save_checkpoint(5, {"w": torch.zeros(1)}, {"x": torch.zeros(1)})
        assert not mgr.is_complete(total_steps=10, interval=5)
        mgr.save_checkpoint(10, {"w": torch.zeros(1)}, {"x": torch.zeros(1)})
        assert mgr.is_complete(total_steps=10, interval=5)


class TestPDSScorerInterface:
    def test_registered_under_pds(self):
        scorer = get_scorer(
            "pds", proxy="ignored", cache_dir="/tmp/pds_test_cache"
        )
        assert scorer.name == "pds"

    def test_empty_pool_returns_empty(self):
        from medsel.selection.scorers.pds import PDSScorer

        scorer = PDSScorer(proxy="ignored", cache_dir="/tmp/pds_test_cache")
        assert scorer.score([]) == []

    def test_rejects_invalid_pmp_steps(self):
        from medsel.selection.scorers.pds import PDSScorer

        with pytest.raises(ValueError, match="pmp_steps"):
            PDSScorer(proxy="x", pmp_steps=0, cache_dir="/tmp/x")

    def test_rejects_negative_noise_ratio(self):
        from medsel.selection.scorers.pds import PDSScorer

        with pytest.raises(ValueError, match="noise_ratio"):
            PDSScorer(proxy="x", noise_ratio=-1.0, cache_dir="/tmp/x")

    def test_requires_cache_dir(self):
        from medsel.selection.scorers.pds import PDSScorer

        scorer = PDSScorer(proxy="x")
        with pytest.raises(ValueError, match="cache_dir"):
            scorer.score([_example()])

    def test_resolve_target_split_uses_validation(self):
        from medsel.selection.scorers.pds import PDSScorer

        scorer = PDSScorer(proxy="x", target="medmcqa", cache_dir="/tmp/x")
        assert scorer.resolve_target_split() == "validation"

    def test_repr_includes_key_params(self):
        from medsel.selection.scorers.pds import PDSScorer

        s = PDSScorer(proxy="model/name", pmp_steps=50, cache_dir="/tmp/x")
        r = repr(s)
        assert "model/name" in r
        assert "50" in r

    def test_default_checkpoint_interval_from_pmp_steps(self):
        from medsel.selection.scorers.pds import PDSScorer

        s = PDSScorer(proxy="x", pmp_steps=100, cache_dir="/tmp/x")
        assert s.checkpoint_interval == 10

    def test_custom_checkpoint_interval(self):
        from medsel.selection.scorers.pds import PDSScorer

        s = PDSScorer(proxy="x", pmp_steps=100, checkpoint_interval=5, cache_dir="/tmp/x")
        assert s.checkpoint_interval == 5
