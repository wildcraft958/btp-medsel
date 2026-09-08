"""Training on a selected subset rather than on the whole pool."""

import json

import pytest

from medsel.schema import CorpusDoc
from medsel.stages.base import apply_selection, read_selection


def docs(n, offset=0):
    return [
        CorpusDoc(uid=f"pubmed/train/{i + offset:06d}", source="pubmed", text=f"doc {i + offset}")
        for i in range(n)
    ]


def manifest(tmp_path, uids, *, source="pubmed", split="train", n_candidates=10):
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "source": source,
                "split": split,
                "scorer": "random",
                "n_candidates": n_candidates,
                "n_selected": len(uids),
                "selected_uids": list(uids),
            }
        )
    )
    return path


class TestApplySelection:
    def test_keeps_only_the_selected_records(self, tmp_path):
        pool = docs(10)
        chosen = [pool[2].uid, pool[5].uid]
        kept = list(apply_selection(pool, manifest(tmp_path, chosen), "pubmed", "train"))
        assert [r.uid for r in kept] == chosen

    def test_preserves_pool_order_not_manifest_order(self, tmp_path):
        pool = docs(10)
        chosen = [pool[7].uid, pool[1].uid]
        kept = list(apply_selection(pool, manifest(tmp_path, chosen), "pubmed", "train"))
        assert [r.uid for r in kept] == [pool[1].uid, pool[7].uid]

    def test_raises_when_a_selected_record_is_missing(self, tmp_path):
        """A stale manifest against a changed pool would silently train on the wrong subset."""
        pool = docs(10)
        chosen = [pool[2].uid, "pubmed/train/999999"]
        with pytest.raises(ValueError, match="1 of 2"):
            list(apply_selection(pool, manifest(tmp_path, chosen), "pubmed", "train"))

    def test_raises_on_a_source_mismatch(self, tmp_path):
        path = manifest(tmp_path, [], source="medmcqa")
        with pytest.raises(ValueError, match="medmcqa"):
            list(apply_selection(docs(2), path, "pubmed", "train"))

    def test_raises_on_a_split_mismatch(self, tmp_path):
        path = manifest(tmp_path, [], split="validation")
        with pytest.raises(ValueError, match="validation"):
            list(apply_selection(docs(2), path, "pubmed", "train"))

    def test_raises_on_an_empty_selection(self, tmp_path):
        with pytest.raises(ValueError, match="no records"):
            list(apply_selection(docs(2), manifest(tmp_path, []), "pubmed", "train"))

    def test_missing_manifest_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.json"):
            list(apply_selection(docs(2), tmp_path / "nope.json", "pubmed", "train"))


class TestReadSelectionFile:
    def test_reads_the_pool_size(self, tmp_path):
        path = manifest(tmp_path, ["a", "b"], n_candidates=42)
        assert read_selection(path)["n_candidates"] == 42

    def test_rejects_a_manifest_without_uids(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"source": "pubmed", "split": "train"}))
        with pytest.raises(ValueError, match="selected_uids"):
            read_selection(path)
