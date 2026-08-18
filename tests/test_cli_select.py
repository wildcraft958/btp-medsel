"""The select subcommand: argument handling and manifest assembly."""

import pytest

from medsel.cli import build_manifest, build_parser
from medsel.schema import CorpusDoc


def docs(n):
    return [CorpusDoc(uid=f"d{i}", source="pubmed", text=f"document {i}") for i in range(n)]


class TestSelectParser:
    def test_requires_a_source(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["select", "--scorer", "random"])

    def test_defaults_the_scorer_to_random(self):
        args = build_parser().parse_args(["select", "--source", "pubmed"])
        assert args.scorer == "random"

    def test_budget_is_a_float_fraction_by_default(self):
        args = build_parser().parse_args(["select", "--source", "pubmed"])
        assert isinstance(args.budget, float)

    def test_accepts_scorer_arguments(self):
        args = build_parser().parse_args(
            ["select", "--source", "pubmed", "--scorer", "dsir", "--scorer-arg", "target=medmcqa"]
        )
        assert args.scorer_arg == ["target=medmcqa"]

    def test_accepts_stratification(self):
        args = build_parser().parse_args(
            [
                "select",
                "--source",
                "medmcqa",
                "--stratify-by",
                "subject_name",
                "--min-per-group",
                "2",
            ]
        )
        assert args.stratify_by == "subject_name" and args.min_per_group == 2


class TestBuildManifest:
    def make(self, chosen=(0, 2), n=4):
        return build_manifest(
            records=docs(n),
            scores=[0.9, 0.1, 0.8, 0.2],
            chosen=list(chosen),
            source="pubmed",
            split="train",
            scorer="random",
            scorer_args={"seed": 1},
            budget=0.5,
            strategy="top_k",
        )

    def test_records_the_selected_uids_in_choice_order(self):
        assert self.make(chosen=[2, 0])["selected_uids"] == ["d2", "d0"]

    def test_counts_candidates_and_selection(self):
        manifest = self.make()
        assert manifest["n_candidates"] == 4 and manifest["n_selected"] == 2

    def test_carries_the_provenance_needed_to_reproduce_it(self):
        manifest = self.make()
        assert manifest["source"] == "pubmed"
        assert manifest["scorer"] == "random"
        assert manifest["scorer_args"] == {"seed": 1}
        assert manifest["strategy"] == "top_k"

    def test_summarises_the_score_distribution(self):
        stats = self.make()["score_summary"]
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.9)
        assert stats["mean"] == pytest.approx(0.5)

    def test_reports_the_selected_score_range_separately(self):
        stats = self.make(chosen=[0, 2])["score_summary"]
        assert stats["selected_min"] == pytest.approx(0.8)

    def test_empty_selection_is_representable(self):
        manifest = self.make(chosen=[])
        assert manifest["n_selected"] == 0 and manifest["selected_uids"] == []

    def test_empty_pool_is_representable(self):
        manifest = build_manifest(
            records=[],
            scores=[],
            chosen=[],
            source="pubmed",
            split="train",
            scorer="random",
            scorer_args={},
            budget=0.5,
            strategy="top_k",
        )
        assert manifest["n_candidates"] == 0 and manifest["score_summary"] == {}
