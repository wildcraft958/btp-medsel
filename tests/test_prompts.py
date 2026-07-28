import pytest

from medsel.prompts.templates import TEMPLATE_VERSION, render_continuation, render_prompt
from medsel.schema import QAExample


def example(**overrides) -> QAExample:
    base = dict(
        uid="medmcqa/validation/000000",
        source="medmcqa",
        split="validation",
        question="Which nerve innervates the diaphragm?",
        options={"A": "Vagus", "B": "Phrenic", "C": "Ulnar", "D": "Radial"},
        answer_key="B",
    )
    base.update(overrides)
    return QAExample(**base)


class TestRenderPrompt:
    def test_lists_options_in_letter_order(self):
        prompt = render_prompt(example())
        assert "A. Vagus" in prompt and "D. Radial" in prompt
        assert prompt.index("A. Vagus") < prompt.index("B. Phrenic")

    def test_ends_at_answer_without_trailing_space(self):
        # The space belongs to the continuation; tokenizers attach it to the following word.
        prompt = render_prompt(example())
        assert prompt.endswith("Answer:")
        assert not prompt.endswith(" ")

    def test_includes_the_question(self):
        assert "Which nerve innervates the diaphragm?" in render_prompt(example())

    def test_contexts_are_included_when_present(self):
        prompt = render_prompt(example(contexts=["Background on phrenic anatomy."]))
        assert "Context: Background on phrenic anatomy." in prompt

    def test_contexts_can_be_suppressed(self):
        prompt = render_prompt(example(contexts=["hidden"]), include_contexts=False)
        assert "hidden" not in prompt

    def test_no_context_line_when_there_are_none(self):
        assert "Context:" not in render_prompt(example())

    def test_long_context_is_truncated_from_the_head(self):
        # PubMedQA puts conclusions last, so the tail is the part worth keeping.
        tail = "CONCLUSION: yes."
        prompt = render_prompt(example(contexts=["x" * 5000 + tail]), max_context_chars=100)
        assert tail in prompt
        assert len(prompt) < 1000

    def test_example_without_options_is_rejected(self):
        naked = QAExample(uid="u/s/0", source="u", split="s", question="q?")
        with pytest.raises(ValueError, match="no options"):
            render_prompt(naked)


class TestRenderContinuation:
    def test_letter_mode_scores_the_key(self):
        assert render_continuation(example(), "B", mode="letter") == " B"

    def test_text_mode_scores_the_wording(self):
        assert render_continuation(example(), "B", mode="text") == " Phrenic"

    def test_unknown_option_is_rejected(self):
        with pytest.raises(ValueError, match="no option"):
            render_continuation(example(), "Z")

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown scoring mode"):
            render_continuation(example(), "A", mode="embedding")


def test_template_version_is_recorded():
    assert isinstance(TEMPLATE_VERSION, int) and TEMPLATE_VERSION >= 1
