"""Config loading. The load-bearing behaviour is that typos fail loudly rather than silently."""

from pathlib import Path

import pytest
import yaml

from medsel.config import ExperimentConfig, TrainConfig, load_experiment, to_dict

REPO_CONFIGS = Path(__file__).parent.parent / "configs" / "experiment"

MINIMAL = {
    "name": "unit",
    "stage": "cpt",
    "data": {"source": "pubmed", "limit": 10},
    "model": {"name_or_path": "tiny/model"},
}


def write(tmp_path: Path, payload: dict, name: str = "exp.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


class TestLoading:
    def test_builds_nested_dataclasses(self, tmp_path):
        config = load_experiment(write(tmp_path, MINIMAL))
        assert isinstance(config, ExperimentConfig)
        assert config.data.source == "pubmed"
        assert config.model.name_or_path == "tiny/model"

    def test_train_section_defaults_when_absent(self, tmp_path):
        config = load_experiment(write(tmp_path, MINIMAL))
        assert config.train == TrainConfig()

    def test_values_override_defaults(self, tmp_path):
        payload = {**MINIMAL, "train": {"block_size": 2048, "learning_rate": 1e-4}}
        config = load_experiment(write(tmp_path, payload))
        assert config.train.block_size == 2048
        assert config.train.learning_rate == pytest.approx(1e-4)

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no config at"):
            load_experiment(tmp_path / "absent.yaml")


class TestStrictKeys:
    def test_typo_in_train_section_is_rejected(self, tmp_path):
        # 'learning_rare' would otherwise be ignored and the run would train at the default rate.
        payload = {**MINIMAL, "train": {"learning_rare": 1e-4}}
        with pytest.raises(ValueError, match="learning_rare"):
            load_experiment(write(tmp_path, payload))

    def test_error_lists_valid_keys(self, tmp_path):
        payload = {**MINIMAL, "model": {"name_or_path": "x", "dtpye": "bf16"}}
        with pytest.raises(ValueError, match="use_lora"):
            load_experiment(write(tmp_path, payload))

    def test_typo_at_top_level_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="stagge"):
            load_experiment(write(tmp_path, {**MINIMAL, "stagge": "cpt"}))

    def test_non_mapping_section_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be a mapping"):
            load_experiment(write(tmp_path, {**MINIMAL, "train": [1, 2, 3]}))

    def test_non_mapping_document_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(TypeError, match="mapping at the top level"):
            load_experiment(path)


class TestSectionReferences:
    def test_section_can_point_at_a_shared_file(self, tmp_path):
        (tmp_path / "model.yaml").write_text(yaml.safe_dump({"name_or_path": "shared/model"}))
        payload = {**MINIMAL, "model": "model.yaml"}
        assert load_experiment(write(tmp_path, payload)).model.name_or_path == "shared/model"

    def test_reference_resolves_relative_to_the_config(self, tmp_path):
        (tmp_path / "shared").mkdir()
        (tmp_path / "shared" / "m.yaml").write_text(yaml.safe_dump({"name_or_path": "nested/m"}))
        (tmp_path / "exp").mkdir()
        payload = {**MINIMAL, "model": "../shared/m.yaml"}
        path = write(tmp_path / "exp", payload)
        assert load_experiment(path).model.name_or_path == "nested/m"

    def test_dangling_reference_names_the_missing_path(self, tmp_path):
        payload = {**MINIMAL, "model": "nope.yaml"}
        with pytest.raises(FileNotFoundError, match="nope.yaml"):
            load_experiment(write(tmp_path, payload))


class TestShippedConfigs:
    """The configs in the repo must actually load - they are the documented entry points."""

    @pytest.mark.parametrize("path", sorted(REPO_CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
    def test_repo_config_loads(self, path):
        config = load_experiment(path)
        assert config.name and config.stage
        assert config.model.name_or_path

    def test_smoke_config_keeps_the_packing_remainder(self, tmp_path):
        # A tiny corpus can pack to zero full blocks; the smoke test must still have data.
        config = load_experiment(REPO_CONFIGS / "smoke_cpu.yaml")
        assert config.train.drop_remainder is False
        assert config.data.limit is not None

    def test_smoke_config_uses_an_ungated_model(self):
        config = load_experiment(REPO_CONFIGS / "smoke_cpu.yaml")
        assert "gemma" not in config.model.name_or_path.lower()


class TestToDict:
    def test_round_trips_to_plain_types(self, tmp_path):
        payload = to_dict(load_experiment(write(tmp_path, MINIMAL)))
        assert payload["data"]["source"] == "pubmed"
        assert isinstance(payload["train"], dict)
