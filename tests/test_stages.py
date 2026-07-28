"""Stage wiring. No model is downloaded: every case fails before a tokenizer is needed."""

import pytest

from medsel.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from medsel.stages import STAGES, AlignStage, CPTStage, SFTStage, available_stages, get_stage
from medsel.stages.base import Stage


def config(source="pubmed", **train_kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        name="unit",
        stage="cpt",
        data=DataConfig(source=source, limit=4),
        model=ModelConfig(name_or_path="tiny/model"),
        train=TrainConfig(**train_kwargs),
    )


class TestStageRegistry:
    def test_pipeline_has_the_three_proposal_stages(self):
        assert available_stages() == ["cpt", "sft", "align"]

    @pytest.mark.parametrize(
        ("name", "cls"), [("cpt", CPTStage), ("sft", SFTStage), ("align", AlignStage)]
    )
    def test_lookup_returns_the_right_class(self, name, cls):
        assert get_stage(name) is cls

    def test_unknown_stage_lists_alternatives(self):
        with pytest.raises(KeyError, match="cpt, sft, align"):
            get_stage("rlhf")

    def test_every_stage_implements_the_interface(self):
        assert all(issubclass(cls, Stage) for cls in STAGES.values())


class TestUnimplementedStages:
    @pytest.mark.parametrize("cls", [SFTStage, AlignStage])
    def test_prepare_raises_with_guidance(self, cls):
        with pytest.raises(NotImplementedError, match="src/medsel/stages"):
            cls(config()).prepare()

    @pytest.mark.parametrize("cls", [SFTStage, AlignStage])
    def test_run_raises_with_guidance(self, cls):
        with pytest.raises(NotImplementedError):
            cls(config()).run()

    def test_sft_message_names_the_intended_trainer(self):
        with pytest.raises(NotImplementedError, match="SFTTrainer"):
            SFTStage(config()).prepare()

    def test_align_message_names_the_real_blocker(self):
        # The blocker is dataset choice, not training code; the message should say so.
        with pytest.raises(NotImplementedError, match="preference dataset"):
            AlignStage(config()).prepare()


class TestCPTSourceValidation:
    @pytest.mark.parametrize("source", ["medqa", "medmcqa", "pubmedqa"])
    def test_qa_sources_are_rejected_before_any_download(self, source):
        with pytest.raises(ValueError, match="CorpusDoc"):
            CPTStage(config(source=source)).prepare()

    def test_rejection_points_at_the_right_source(self):
        with pytest.raises(ValueError, match="'pubmed' here"):
            CPTStage(config(source="medmcqa")).prepare()

    def test_unknown_source_is_rejected(self):
        with pytest.raises(KeyError, match="unknown loader"):
            CPTStage(config(source="not_a_dataset")).prepare()


class TestStageBasics:
    def test_output_dir_comes_from_train_config(self, tmp_path):
        stage = CPTStage(config(output_dir=str(tmp_path / "out")))
        assert stage.output_dir == tmp_path / "out"

    def test_repr_names_the_experiment(self):
        assert "unit" in repr(CPTStage(config()))

    def test_manifest_records_config_and_hardware(self, tmp_path):
        stage = CPTStage(config(output_dir=str(tmp_path / "out")))
        path = stage.write_manifest({"train_loss": 1.23})

        import json

        manifest = json.loads(path.read_text())
        assert manifest["stage"] == "cpt"
        assert manifest["metrics"]["train_loss"] == 1.23
        assert manifest["config"]["data"]["source"] == "pubmed"
        assert "hardware" in manifest and "finished_at" in manifest
