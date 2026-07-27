from dataclasses import FrozenInstanceError

import pytest

from medsel.schema import CorpusDoc, QAExample, make_uid


def make_qa(**overrides):
    base = dict(
        uid="medmcqa/train/000001",
        source="medmcqa",
        split="train",
        question="Which nerve innervates the diaphragm?",
        options={"A": "Vagus", "B": "Phrenic", "C": "Ulnar", "D": "Radial"},
        answer_key="B",
        answer_text="Phrenic",
    )
    base.update(overrides)
    return QAExample(**base)


class TestQAExample:
    def test_roundtrip_through_dict(self):
        ex = make_qa(contexts=["ctx one"], labels={"subject_name": "Anatomy"})
        assert QAExample.from_dict(ex.to_dict()) == ex

    def test_is_labeled_tracks_answer_key(self):
        assert make_qa().is_labeled
        assert not make_qa(answer_key=None, answer_text=None).is_labeled

    def test_n_options(self):
        assert make_qa().n_options == 4
        assert make_qa(options=None, answer_key=None, answer_text=None).n_options == 0

    def test_answer_text_backfilled_from_options(self):
        assert make_qa(answer_text=None).answer_text == "Phrenic"

    def test_rejects_answer_key_absent_from_options(self):
        with pytest.raises(ValueError, match="answer_key"):
            make_qa(answer_key="Z", answer_text=None)

    def test_rejects_blank_question(self):
        with pytest.raises(ValueError, match="question"):
            make_qa(question="   ")

    def test_rejects_blank_uid(self):
        with pytest.raises(ValueError, match="uid"):
            make_qa(uid="")

    def test_is_immutable(self):
        with pytest.raises(FrozenInstanceError):
            make_qa().question = "changed"

    def test_contexts_coerced_to_list(self):
        assert make_qa(contexts=("a", "b")).contexts == ["a", "b"]

    def test_labels_values_coerced_to_str(self):
        assert make_qa(labels={"cop": 1}).labels == {"cop": "1"}

    def test_unlabeled_example_needs_no_answer(self):
        ex = make_qa(answer_key=None, answer_text=None)
        assert ex.answer_key is None and ex.options is not None


class TestCorpusDoc:
    def test_full_text_joins_title_and_body(self):
        doc = CorpusDoc(uid="pubmed/1", source="pubmed", title="A title", text="Body text.")
        assert doc.full_text == "A title\n\nBody text."

    def test_full_text_without_title_is_body(self):
        assert CorpusDoc(uid="pubmed/1", source="pubmed", text="Body.").full_text == "Body."

    def test_rejects_blank_text(self):
        with pytest.raises(ValueError, match="text"):
            CorpusDoc(uid="pubmed/1", source="pubmed", text="  ")

    def test_roundtrip_through_dict(self):
        doc = CorpusDoc(uid="pubmed/1", source="pubmed", title="T", text="B", meta={"PMID": 57})
        assert CorpusDoc.from_dict(doc.to_dict()) == doc


def test_make_uid_is_zero_padded_and_stable():
    assert make_uid("medqa", "test", 7) == "medqa/test/000007"
    assert make_uid("medqa", "test", 1234567) == "medqa/test/1234567"
