"""Uncertainty and spread on an evaluation result."""

import pytest

from medsel.eval.stats import accuracy_ci, subject_spread


class TestAccuracyCI:
    def test_brackets_the_point_estimate(self):
        low, high = accuracy_ci(500, 1000)
        assert low < 0.5 < high

    def test_narrows_as_the_sample_grows(self):
        small = accuracy_ci(50, 100)
        large = accuracy_ci(5000, 10_000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_is_reproducible_for_a_fixed_seed(self):
        assert accuracy_ci(400, 1273, seed=7) == accuracy_ci(400, 1273, seed=7)

    def test_medqa_sized_interval_is_wider_than_a_point_and_a_half(self):
        """The reason this exists: MedQA has 1,273 items, so small deltas are inside noise."""
        low, high = accuracy_ci(484, 1273)
        assert (high - low) > 0.015

    def test_a_wider_confidence_gives_a_wider_interval(self):
        narrow = accuracy_ci(500, 1000, confidence=0.80)
        wide = accuracy_ci(500, 1000, confidence=0.99)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_stays_inside_zero_and_one(self):
        for correct in (0, 1, 999, 1000):
            low, high = accuracy_ci(correct, 1000)
            assert 0.0 <= low <= high <= 1.0

    def test_zero_examples_gives_no_interval(self):
        assert accuracy_ci(0, 0) is None

    def test_rejects_more_correct_than_scored(self):
        with pytest.raises(ValueError, match="correct"):
            accuracy_ci(11, 10)

    def test_rejects_an_out_of_range_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            accuracy_ci(5, 10, confidence=1.5)


class TestSubjectSpread:
    BY_SUBJECT = {
        "Anatomy": {"accuracy": 0.40, "n": 100},
        "Physiology": {"accuracy": 0.60, "n": 100},
        "Surgery": {"accuracy": 0.50, "n": 100},
    }

    def test_reports_the_variance_across_subjects(self):
        assert subject_spread(self.BY_SUBJECT)["variance"] == pytest.approx(0.0066667, rel=1e-3)

    def test_reports_the_spread_between_best_and_worst(self):
        spread = subject_spread(self.BY_SUBJECT)
        assert spread["min"] == pytest.approx(0.40)
        assert spread["max"] == pytest.approx(0.60)
        assert spread["range"] == pytest.approx(0.20)

    def test_names_the_weakest_subject(self):
        assert subject_spread(self.BY_SUBJECT)["worst"] == "Anatomy"

    def test_counts_the_subjects(self):
        assert subject_spread(self.BY_SUBJECT)["n_subjects"] == 3

    def test_a_single_subject_has_zero_variance(self):
        assert subject_spread({"Anatomy": {"accuracy": 0.4, "n": 10}})["variance"] == 0.0

    def test_no_subjects_gives_nothing(self):
        assert subject_spread({}) == {}

    def test_ignores_subjects_below_the_minimum_sample(self):
        """A subject with three items has an accuracy that is mostly noise."""
        noisy = {**self.BY_SUBJECT, "Rare": {"accuracy": 1.0, "n": 2}}
        assert subject_spread(noisy, min_n=10)["n_subjects"] == 3


class TestReportCarriesStats:
    """The report itself has to carry the interval, or nobody will compute it."""

    def make(self):
        from medsel.eval.mcq import EvalReport

        return EvalReport(
            source="medmcqa",
            split="validation",
            model="m",
            mode="letter",
            n_scored=1000,
            n_skipped_unlabeled=0,
            accuracy=0.5,
            accuracy_norm=0.5,
            accuracy_ci=(0.47, 0.53),
        )

    def test_interval_survives_serialisation(self):
        assert self.make().to_dict()["accuracy_ci"] == (0.47, 0.53)

    def test_defaults_leave_the_fields_absent_rather_than_wrong(self):
        from medsel.eval.mcq import EvalReport

        report = EvalReport(
            source="s",
            split="p",
            model="m",
            mode="letter",
            n_scored=1,
            n_skipped_unlabeled=0,
            accuracy=1.0,
            accuracy_norm=1.0,
        )
        assert report.accuracy_ci is None and report.subject_spread == {}
