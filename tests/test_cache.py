import json

import pytest

from medsel.data.cache import (
    cache_is_valid,
    cache_paths,
    clear_cache,
    read_cache,
    read_manifest,
    write_cache,
)
from medsel.schema import CorpusDoc, QAExample

KEY = "medmcqa-None-v1"


def qa(idx: int) -> QAExample:
    return QAExample(
        uid=f"medmcqa/train/{idx:06d}",
        source="medmcqa",
        split="train",
        question=f"Question {idx}?",
        options={"A": "alpha", "B": "beta"},
        answer_key="A",
        contexts=[f"context {idx}"],
        rationale="because",
        labels={"subject_name": "Anatomy", "topic_name": "Thorax"},
    )


def doc(idx: int) -> CorpusDoc:
    return CorpusDoc(
        uid=f"pubmed/{idx}", source="pubmed", title=f"Title {idx}", text="Body.", meta={"PMID": idx}
    )


class TestRoundTrip:
    def test_qa_examples_survive_roundtrip(self, tmp_path):
        originals = [qa(i) for i in range(5)]
        write_cache(originals, "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert list(read_cache("medmcqa", "train", tmp_path, progress=False)) == originals

    def test_corpus_docs_survive_roundtrip(self, tmp_path):
        originals = [doc(i) for i in range(3)]
        write_cache(originals, "pubmed", "train", "pubmed-v1", CorpusDoc, tmp_path, progress=False)
        assert list(read_cache("pubmed", "train", tmp_path, progress=False)) == originals

    def test_unlabeled_example_survives_roundtrip(self, tmp_path):
        original = QAExample(
            uid="medmcqa/test/000000",
            source="medmcqa",
            split="test",
            question="Unlabeled?",
            options={"A": "a", "B": "b"},
            answer_key=None,
        )
        write_cache([original], "medmcqa", "test", KEY, QAExample, tmp_path, progress=False)
        [restored] = read_cache("medmcqa", "test", tmp_path, progress=False)
        assert restored == original and not restored.is_labeled

    def test_batching_handles_more_rows_than_one_batch(self, tmp_path):
        originals = [qa(i) for i in range(2500)]  # exceeds the 1000-row flush size
        write_cache(originals, "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert list(read_cache("medmcqa", "train", tmp_path, progress=False)) == originals

    def test_limit_truncates_read(self, tmp_path):
        write_cache(
            [qa(i) for i in range(10)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False
        )
        assert len(list(read_cache("medmcqa", "train", tmp_path, limit=4, progress=False))) == 4


class TestManifest:
    def test_records_row_count_and_key(self, tmp_path):
        write_cache(
            [qa(i) for i in range(7)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False
        )
        manifest = read_manifest("medmcqa", "train", tmp_path)
        assert manifest["n_rows"] == 7
        assert manifest["cache_key"] == KEY
        assert manifest["record_type"] == "QAExample"

    def test_extra_fields_are_persisted(self, tmp_path):
        write_cache(
            [qa(0)],
            "medmcqa",
            "train",
            KEY,
            QAExample,
            tmp_path,
            extra={"hf_id": "openlifescienceai/medmcqa"},
            progress=False,
        )
        assert read_manifest("medmcqa", "train", tmp_path)["hf_id"] == "openlifescienceai/medmcqa"

    def test_missing_manifest_reads_as_none(self, tmp_path):
        assert read_manifest("nope", "train", tmp_path) is None


class TestValidity:
    def test_valid_immediately_after_write(self, tmp_path):
        write_cache([qa(0)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert cache_is_valid("medmcqa", "train", KEY, QAExample, tmp_path)

    def test_invalid_when_loader_version_changed(self, tmp_path):
        write_cache([qa(0)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert not cache_is_valid("medmcqa", "train", "medmcqa-None-v2", QAExample, tmp_path)

    def test_invalid_when_record_type_differs(self, tmp_path):
        write_cache([qa(0)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert not cache_is_valid("medmcqa", "train", KEY, CorpusDoc, tmp_path)

    def test_invalid_when_schema_hash_drifts(self, tmp_path):
        write_cache([qa(0)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        _, manifest_path = cache_paths("medmcqa", "train", tmp_path)
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_hash"] = "deadbeefdeadbeef"
        manifest_path.write_text(json.dumps(manifest))
        assert not cache_is_valid("medmcqa", "train", KEY, QAExample, tmp_path)

    def test_invalid_when_nothing_cached(self, tmp_path):
        assert not cache_is_valid("medmcqa", "train", KEY, QAExample, tmp_path)


class TestFailureHandling:
    def test_partial_write_leaves_no_parquet(self, tmp_path):
        def exploding():
            yield qa(0)
            raise RuntimeError("upstream died")

        with pytest.raises(RuntimeError, match="upstream died"):
            write_cache(exploding(), "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)

        parquet_path, _ = cache_paths("medmcqa", "train", tmp_path)
        assert not parquet_path.exists()

    def test_reading_absent_cache_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="medsel data prepare"):
            list(read_cache("medmcqa", "train", tmp_path, progress=False))

    def test_empty_input_still_produces_valid_cache(self, tmp_path):
        write_cache([], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert cache_is_valid("medmcqa", "train", KEY, QAExample, tmp_path)
        assert read_manifest("medmcqa", "train", tmp_path)["n_rows"] == 0


class TestClearCache:
    def test_removes_files_and_counts_them(self, tmp_path):
        write_cache([qa(0)], "medmcqa", "train", KEY, QAExample, tmp_path, progress=False)
        assert clear_cache("medmcqa", tmp_path) == 2  # parquet + manifest
        assert not cache_is_valid("medmcqa", "train", KEY, QAExample, tmp_path)

    def test_clearing_absent_source_is_a_no_op(self, tmp_path):
        assert clear_cache("nope", tmp_path) == 0
