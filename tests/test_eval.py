"""Evaluator tests against a deterministic stand-in model - no weights are downloaded."""

from types import SimpleNamespace

import pytest
import torch

from medsel.eval.mcq import EvalReport, OptionScores, evaluate, score_example
from medsel.eval.runner import DEFAULT_EVAL_SPLITS, evaluate_task
from medsel.schema import QAExample

VOCAB = 256


class FakeTokenizer:
    """Character-level tokenizer, so token ids are readable in assertions."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % VOCAB for c in text]


class FavouringModel(torch.nn.Module):
    """Assigns overwhelming probability to one token id, whatever the context."""

    def __init__(self, favoured_char: str):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.favoured = ord(favoured_char) % VOCAB

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, VOCAB)
        logits[:, :, self.favoured] = 20.0
        return SimpleNamespace(logits=logits)


def example(idx=0, answer_key="B", subject=None, **overrides) -> QAExample:
    base = dict(
        uid=f"medmcqa/validation/{idx:06d}",
        source="medmcqa",
        split="validation",
        question="Which nerve innervates the diaphragm?",
        options={"A": "Vagus", "B": "Phrenic", "C": "Ulnar", "D": "Radial"},
        answer_key=answer_key,
        labels={"subject_name": subject} if subject else {},
    )
    base.update(overrides)
    return QAExample(**base)


class TestOptionScores:
    def test_prediction_is_the_highest_total(self):
        scores = OptionScores("u", "A", {"A": -1.0, "B": -5.0}, {"A": -0.5, "B": -2.5})
        assert scores.predicted_key == "A" and scores.correct

    def test_length_normalisation_can_change_the_winner(self):
        # The long option wins on total but loses per token - exactly what acc_norm corrects for.
        scores = OptionScores("u", "B", {"A": -3.0, "B": -4.0}, {"A": -3.0, "B": -1.0})
        assert scores.predicted_key == "A"
        assert scores.predicted_key_norm == "B"
        assert not scores.correct and scores.correct_norm

    def test_unlabeled_example_is_never_correct(self):
        scores = OptionScores("u", None, {"A": -1.0, "B": -5.0}, {"A": -1.0, "B": -5.0})
        assert not scores.correct and not scores.correct_norm


class TestScoreExample:
    def test_favoured_letter_wins_in_letter_mode(self):
        scores = score_example(FavouringModel("C"), FakeTokenizer(), example(), mode="letter")
        assert scores.predicted_key == "C"

    def test_every_option_receives_a_score(self):
        scores = score_example(FavouringModel("B"), FakeTokenizer(), example(), mode="letter")
        assert set(scores.total) == {"A", "B", "C", "D"}
        assert set(scores.normalized) == {"A", "B", "C", "D"}

    def test_scores_are_negative_log_probabilities(self):
        scores = score_example(FavouringModel("B"), FakeTokenizer(), example(), mode="letter")
        assert all(value <= 0 for value in scores.total.values())

    def test_gold_key_is_carried_through(self):
        scores = score_example(FavouringModel("B"), FakeTokenizer(), example(), mode="letter")
        assert scores.gold_key == "B" and scores.correct

    def test_text_mode_scores_option_wording(self):
        # 'Vagus' is the only option containing 'g', so favouring 'g' must select A.
        scores = score_example(FavouringModel("g"), FakeTokenizer(), example(), mode="text")
        assert scores.predicted_key == "A"

    def test_empty_option_does_not_crash(self):
        blank = example(options={"A": "Vagus", "B": ""}, answer_key="A")
        scores = score_example(FavouringModel("V"), FakeTokenizer(), blank, mode="text")
        assert set(scores.total) == {"A", "B"}

    def test_example_without_options_is_rejected(self):
        naked = QAExample(uid="u/s/0", source="u", split="s", question="q?")
        with pytest.raises(ValueError, match="no options"):
            score_example(FavouringModel("A"), FakeTokenizer(), naked)


class TestEvaluate:
    def run(self, examples, favoured="B", mode="letter"):
        return evaluate(
            FavouringModel(favoured),
            FakeTokenizer(),
            examples,
            source="medmcqa",
            split="validation",
            model_name="fake",
            mode=mode,
            progress=False,
        )

    def test_perfect_score_when_the_model_favours_the_gold_letter(self):
        report = self.run([example(i, answer_key="B") for i in range(5)])
        assert report.accuracy == 1.0 and report.n_scored == 5

    def test_zero_score_when_it_favours_a_distractor(self):
        report = self.run([example(i, answer_key="A") for i in range(4)])
        assert report.accuracy == 0.0

    def test_mixed_gold_gives_a_partial_score(self):
        examples = [example(0, "B"), example(1, "B"), example(2, "A"), example(3, "C")]
        assert self.run(examples).accuracy == 0.5

    def test_unlabeled_examples_are_skipped_not_scored(self):
        examples = [example(0, "B"), example(1, answer_key=None), example(2, "B")]
        report = self.run(examples)
        assert report.n_scored == 2 and report.n_skipped_unlabeled == 1
        assert report.accuracy == 1.0

    def test_all_unlabeled_raises_and_names_the_fix(self):
        examples = [example(i, answer_key=None) for i in range(3)]
        with pytest.raises(ValueError, match="validation"):
            self.run(examples)

    def test_per_subject_breakdown_is_reported(self):
        examples = [
            example(0, "B", subject="Anatomy"),
            example(1, "A", subject="Anatomy"),
            example(2, "B", subject="Physiology"),
        ]
        report = self.run(examples)
        assert report.by_subject["Anatomy"] == {"accuracy": 0.5, "n": 2}
        assert report.by_subject["Physiology"] == {"accuracy": 1.0, "n": 1}

    def test_examples_without_a_subject_are_absent_from_the_breakdown(self):
        assert self.run([example(0, "B")]).by_subject == {}

    def test_report_records_provenance(self):
        report = self.run([example(0, "B")])
        assert report.model == "fake" and report.mode == "letter"
        assert report.template_version >= 1

    def test_report_serialises(self):
        payload = self.run([example(0, "B")]).to_dict()
        assert payload["source"] == "medmcqa"
        assert "accuracy_norm" in payload and "template_version" in payload


class TestRunnerSplitDefaults:
    def test_defaults_avoid_the_unlabeled_medmcqa_split(self):
        assert DEFAULT_EVAL_SPLITS["medmcqa"] == "validation"
        assert DEFAULT_EVAL_SPLITS["medmcqa"] != "test"

    def test_explicitly_asking_for_medmcqa_test_is_refused(self):
        # Refused before a model is touched, hence model=None.
        with pytest.raises(ValueError, match="no gold labels"):
            evaluate_task(None, None, "medmcqa", model_name="fake", split="test")

    def test_error_points_at_the_usable_split(self):
        with pytest.raises(ValueError, match="validation"):
            evaluate_task(None, None, "medmcqa", model_name="fake", split="test")


def test_eval_report_defaults_are_sane():
    report = EvalReport(
        source="medqa",
        split="test",
        model="m",
        mode="letter",
        n_scored=10,
        n_skipped_unlabeled=0,
        accuracy=0.5,
        accuracy_norm=0.5,
    )
    assert report.by_subject == {}
