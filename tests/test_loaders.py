"""Loader tests. Offline: every case runs against real Hub rows captured in tests/fixtures."""

import json
from pathlib import Path

import pytest

from medsel.data.medmcqa import MedMCQALoader
from medsel.data.medqa import MedQALoader
from medsel.data.pubmedqa import DECISION_OPTIONS, PubMedQALoader
from medsel.registry import available_loaders, get_loader
from medsel.schema import QAExample

FIXTURES = Path(__file__).parent / "fixtures"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / f"{name}.jsonl").open()]


def normalized(loader, fixture: str, split: str) -> list[QAExample]:
    return [loader.normalize(row, i, split) for i, row in enumerate(rows(fixture))]


class TestMedQA:
    def test_unpacks_the_nested_data_object(self):
        [first, *_] = normalized(MedQALoader(), "medqa_train", "train")
        assert first.question.startswith("A 23-year-old pregnant woman")
        assert set(first.options) == {"A", "B", "C", "D"}
        assert first.answer_key == "D"
        assert first.answer_text == "Nitrofurantoin"

    def test_answer_key_indexes_into_options(self):
        for example in normalized(MedQALoader(), "medqa_train", "train"):
            assert example.options[example.answer_key] == example.answer_text

    def test_empty_subject_name_is_dropped(self):
        # MedQA ships subject_name as "" for every row; a blank label is worse than no label.
        for example in normalized(MedQALoader(), "medqa_train", "train"):
            assert "subject_name" not in example.labels

    def test_source_id_is_preserved(self):
        [first, *_] = normalized(MedQALoader(), "medqa_train", "train")
        assert first.labels["source_id"] == rows("medqa_train")[0]["id"]

    def test_uses_dev_not_validation(self):
        loader = MedQALoader()
        loader.check_split("dev")
        with pytest.raises(ValueError, match="train, dev, test"):
            loader.check_split("validation")


class TestMedMCQA:
    def test_maps_cop_index_to_letter(self):
        examples = normalized(MedMCQALoader(), "medmcqa_validation", "validation")
        expected = [row["cop"] for row in rows("medmcqa_validation")]
        assert [e.answer_key for e in examples] == ["ABCD"[c] for c in expected]

    def test_flat_option_columns_become_a_mapping(self):
        [first, *_] = normalized(MedMCQALoader(), "medmcqa_validation", "validation")
        raw = rows("medmcqa_validation")[0]
        assert first.options == {"A": raw["opa"], "B": raw["opb"], "C": raw["opc"], "D": raw["opd"]}

    def test_withheld_test_labels_become_none(self):
        # The published test split ships cop = -1 for every row.
        examples = normalized(MedMCQALoader(), "medmcqa_test", "test")
        assert all(e.answer_key is None for e in examples)
        assert not any(e.is_labeled for e in examples)

    def test_test_split_is_declared_unlabeled(self):
        loader = MedMCQALoader()
        assert not loader.has_labels("test")
        assert loader.has_labels("validation")
        assert loader.eval_split == "validation"

    def test_subject_and_topic_reach_labels(self):
        [first, *_] = normalized(MedMCQALoader(), "medmcqa_validation", "validation")
        raw = rows("medmcqa_validation")[0]
        assert first.labels["subject_name"] == raw["subject_name"]
        assert first.labels["choice_type"] == raw["choice_type"]

    def test_explanation_becomes_rationale(self):
        examples = normalized(MedMCQALoader(), "medmcqa_validation", "validation")
        for example, raw in zip(examples, rows("medmcqa_validation"), strict=True):
            assert example.rationale == ((raw["exp"] or "").strip() or None)

    def test_null_explanation_is_none(self):
        # MedMCQA leaves exp null on a large fraction of rows, so rationale must be optional
        # rather than assumed present by anything downstream.
        raws = rows("medmcqa_validation")
        assert any(raw["exp"] is None for raw in raws)
        null_index = next(i for i, raw in enumerate(raws) if raw["exp"] is None)
        assert (
            normalized(MedMCQALoader(), "medmcqa_validation", "validation")[null_index].rationale
            is None
        )

    def test_blank_explanation_is_none_not_empty_string(self):
        row = dict(rows("medmcqa_validation")[0], exp="   ")
        assert MedMCQALoader().normalize(row, 0, "validation").rationale is None

    def test_missing_option_becomes_empty_string_not_the_word_none(self):
        row = dict(rows("medmcqa_validation")[0], opd=None)
        assert MedMCQALoader().normalize(row, 0, "validation").options["D"] == ""


class TestPubMedQA:
    def test_decision_maps_to_fixed_option_letters(self):
        examples = normalized(PubMedQALoader(), "pubmedqa_labeled", "train")
        decisions = [row["final_decision"] for row in rows("pubmedqa_labeled")]
        assert [e.answer_text for e in examples] == decisions
        assert all(e.options == DECISION_OPTIONS for e in examples)

    def test_abstract_becomes_contexts(self):
        [first, *_] = normalized(PubMedQALoader(), "pubmedqa_labeled", "train")
        assert first.contexts == rows("pubmedqa_labeled")[0]["context"]["contexts"]

    def test_mesh_terms_are_kept_for_coverage_analysis(self):
        [first, *_] = normalized(PubMedQALoader(), "pubmedqa_labeled", "train")
        meshes = rows("pubmedqa_labeled")[0]["context"]["meshes"]
        assert first.labels["meshes"].split("; ") == meshes

    def test_long_answer_becomes_rationale(self):
        [first, *_] = normalized(PubMedQALoader(), "pubmedqa_labeled", "train")
        assert first.rationale == rows("pubmedqa_labeled")[0]["long_answer"].strip()

    def test_unlabeled_config_has_no_decision_column_at_all(self):
        # pqa_unlabeled omits final_decision entirely rather than nulling it.
        assert "final_decision" not in rows("pubmedqa_unlabeled")[0]
        loader = PubMedQALoader(config="pqa_unlabeled")
        examples = normalized(loader, "pubmedqa_unlabeled", "train")
        assert all(e.answer_key is None for e in examples)

    def test_labelling_keys_on_config_not_split(self):
        assert PubMedQALoader(config="pqa_labeled").has_labels("train")
        assert not PubMedQALoader(config="pqa_unlabeled").has_labels("train")

    def test_config_is_recorded_in_labels(self):
        [first, *_] = normalized(
            PubMedQALoader(config="pqa_artificial"), "pubmedqa_labeled", "train"
        )
        assert first.labels["config"] == "pqa_artificial"

    def test_default_config_is_the_expert_labelled_set(self):
        assert PubMedQALoader().config == "pqa_labeled"

    def test_eval_split_slice_passes_validation(self):
        PubMedQALoader().check_split("train[:500]")


class TestRegistryWiring:
    def test_all_three_loaders_are_discoverable(self):
        assert {"medqa", "medmcqa", "pubmedqa"} <= set(available_loaders())

    @pytest.mark.parametrize("name", ["medqa", "medmcqa", "pubmedqa"])
    def test_get_loader_returns_a_configured_instance(self, name):
        loader = get_loader(name)
        assert loader.name == name and loader.hf_id

    def test_cache_keys_are_distinct_across_loaders(self):
        keys = {get_loader(n).cache_key for n in ("medqa", "medmcqa", "pubmedqa")}
        assert len(keys) == 3


class TestUidsAreStable:
    @pytest.mark.parametrize(
        ("loader", "fixture", "split", "prefix"),
        [
            (MedQALoader(), "medqa_train", "train", "medqa/train/"),
            (MedMCQALoader(), "medmcqa_validation", "validation", "medmcqa/validation/"),
            (PubMedQALoader(), "pubmedqa_labeled", "train", "pubmedqa/train/"),
        ],
    )
    def test_uid_encodes_source_split_and_index(self, loader, fixture, split, prefix):
        examples = normalized(loader, fixture, split)
        assert [e.uid for e in examples] == [f"{prefix}{i:06d}" for i in range(len(examples))]
        assert len({e.uid for e in examples}) == len(examples)
