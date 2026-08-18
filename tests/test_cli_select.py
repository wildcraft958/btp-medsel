"""The select subcommand: argument handling and manifest assembly."""

import pytest

from medsel.cli import _budget, build_manifest, build_parser, score_summary


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

    def test_streaming_is_off_by_default(self):
        assert build_parser().parse_args(["select", "--source", "pubmed"]).stream is False

    def test_accepts_streaming(self):
        args = build_parser().parse_args(
            ["select", "--source", "pubmed", "--stream", "--budget", "500", "--chunk-size", "64"]
        )
        assert args.stream and args.budget == 500 and args.chunk_size == 64

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


SCORES = [0.9, 0.1, 0.8, 0.2]


class TestBudgetType:
    def test_whole_number_is_a_count(self):
        assert _budget("50") == 50 and isinstance(_budget("50"), int)

    def test_decimal_is_a_fraction(self):
        assert _budget("0.1") == pytest.approx(0.1) and isinstance(_budget("0.1"), float)

    def test_one_point_zero_stays_a_fraction(self):
        assert isinstance(_budget("1.0"), float)


class TestScoreSummary:
    def test_summarises_the_pool(self):
        stats = score_summary(SCORES, [0, 2])
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.9)
        assert stats["mean"] == pytest.approx(0.5)

    def test_reports_the_selected_range_separately(self):
        assert score_summary(SCORES, [0, 2])["selected_min"] == pytest.approx(0.8)

    def test_no_selection_means_no_selected_keys(self):
        assert "selected_min" not in score_summary(SCORES, [])

    def test_empty_pool(self):
        assert score_summary([], []) == {}


class TestBuildManifest:
    def make(self, uids=("d0", "d2"), n=4):
        return build_manifest(
            source="pubmed",
            split="train",
            scorer="random",
            scorer_args={"seed": 1},
            budget=0.5,
            strategy="top_k",
            n_candidates=n,
            selected_uids=list(uids),
            summary=score_summary(SCORES, [0, 2]),
        )

    def test_records_the_selected_uids_in_choice_order(self):
        assert self.make(uids=["d2", "d0"])["selected_uids"] == ["d2", "d0"]

    def test_counts_candidates_and_selection(self):
        manifest = self.make()
        assert manifest["n_candidates"] == 4 and manifest["n_selected"] == 2

    def test_carries_the_provenance_needed_to_reproduce_it(self):
        manifest = self.make()
        assert manifest["source"] == "pubmed"
        assert manifest["scorer"] == "random"
        assert manifest["scorer_args"] == {"seed": 1}
        assert manifest["strategy"] == "top_k"

    def test_empty_selection_is_representable(self):
        manifest = self.make(uids=[])
        assert manifest["n_selected"] == 0 and manifest["selected_uids"] == []
