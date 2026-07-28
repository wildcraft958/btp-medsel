"""PubMed CPT loader tests. Offline: shard reads are redirected at a fixture file."""

import json
from pathlib import Path

import pytest

from medsel.data.pubmed import N_SHARDS, PubMedCorpusLoader, shard_name
from medsel.registry import available_loaders, get_loader
from medsel.schema import CorpusDoc

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_sample.jsonl"


def raw_rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.open()]


@pytest.fixture
def local_loader(tmp_path, monkeypatch):
    """A loader whose shard download is replaced by the fixture file."""

    def build(**kwargs):
        loader = PubMedCorpusLoader(**kwargs)
        monkeypatch.setattr(loader, "download_shards", lambda progress=True: [FIXTURE])
        return loader

    return build


class TestShardNaming:
    def test_matches_upstream_one_based_numbering(self):
        assert shard_name(1) == "chunk/pubmed23n0001.jsonl"
        assert shard_name(1166) == "chunk/pubmed23n1166.jsonl"

    @pytest.mark.parametrize("index", [0, -1, N_SHARDS + 1])
    def test_rejects_out_of_range(self, index):
        with pytest.raises(ValueError, match="out of range"):
            shard_name(index)


class TestShardBudget:
    def test_default_budget_is_small(self):
        # The full corpus is ~70 GB; defaulting to everything would be hostile.
        assert PubMedCorpusLoader().num_shards < N_SHARDS

    def test_offset_lets_collaborators_split_the_corpus(self):
        first = PubMedCorpusLoader(num_shards=4, shard_offset=0).shard_names()
        second = PubMedCorpusLoader(num_shards=4, shard_offset=4).shard_names()
        assert not set(first) & set(second)
        assert first[0] == "chunk/pubmed23n0001.jsonl"
        assert second[0] == "chunk/pubmed23n0005.jsonl"

    def test_budget_is_clipped_at_the_end_of_the_corpus(self):
        loader = PubMedCorpusLoader(num_shards=10, shard_offset=N_SHARDS - 3)
        assert len(loader.shard_names()) == 3

    @pytest.mark.parametrize("kwargs", [{"num_shards": 0}, {"num_shards": N_SHARDS + 1}])
    def test_rejects_impossible_budget(self, kwargs):
        with pytest.raises(ValueError, match="num_shards"):
            PubMedCorpusLoader(**kwargs)

    def test_rejects_out_of_range_offset(self):
        with pytest.raises(ValueError, match="shard_offset"):
            PubMedCorpusLoader(shard_offset=N_SHARDS)

    def test_cache_key_distinguishes_different_budgets(self):
        keys = {
            PubMedCorpusLoader(num_shards=2).cache_key,
            PubMedCorpusLoader(num_shards=4).cache_key,
            PubMedCorpusLoader(num_shards=2, shard_offset=2).cache_key,
            PubMedCorpusLoader(num_shards=2, min_chars=500).cache_key,
        }
        assert len(keys) == 4


class TestNormalization:
    def test_yields_corpus_docs_not_qa_examples(self, local_loader):
        docs = list(local_loader().load(progress=False))
        assert docs and all(isinstance(d, CorpusDoc) for d in docs)
        assert PubMedCorpusLoader.record_type is CorpusDoc

    def test_keeps_title_and_abstract_separately(self, local_loader):
        [first, *_] = local_loader().load(progress=False)
        raw = raw_rows()[0]
        assert first.title == raw["title"].strip()
        assert first.text == raw["content"].strip()

    def test_redundant_contents_field_is_not_stored(self, local_loader):
        # Upstream 'contents' is title + " " + content; storing it too would double the footprint.
        raw = raw_rows()[0]
        assert raw["contents"] == f"{raw['title']} {raw['content']}"
        [first, *_] = local_loader().load(progress=False)
        assert raw["contents"] not in (first.text, first.title)
        assert first.meta.keys() == {"PMID"}

    def test_full_text_recombines_title_and_abstract(self, local_loader):
        [first, *_] = local_loader().load(progress=False)
        assert first.full_text == f"{first.title}\n\n{first.text}"

    def test_uid_is_unique_and_source_prefixed(self, local_loader):
        docs = list(local_loader().load(progress=False))
        assert len({d.uid for d in docs}) == len(docs)
        assert all(d.uid.startswith("pubmed/") for d in docs)

    def test_pmid_is_preserved(self, local_loader):
        docs = list(local_loader().load(progress=False))
        assert [d.meta["PMID"] for d in docs] == [r["PMID"] for r in raw_rows()]


class TestFiltering:
    def test_min_chars_drops_short_abstracts(self, local_loader):
        lengths = sorted(len(r["content"].strip()) for r in raw_rows())
        threshold = lengths[-1]  # keeps only the single longest document
        assert len(list(local_loader(min_chars=threshold).load(progress=False))) == 1

    def test_min_chars_zero_keeps_everything(self, local_loader):
        assert len(list(local_loader(min_chars=0).load(progress=False))) == len(raw_rows())

    def test_limit_caps_output(self, local_loader):
        assert len(list(local_loader(min_chars=0).load(limit=2, progress=False))) == 2

    def test_dedup_removes_repeat_pmids(self, tmp_path, monkeypatch):
        rows = raw_rows()
        doubled = tmp_path / "doubled.jsonl"
        with doubled.open("w") as handle:
            for row in rows + rows:
                handle.write(json.dumps(row) + "\n")

        loader = PubMedCorpusLoader(min_chars=0)
        monkeypatch.setattr(loader, "download_shards", lambda progress=True: [doubled])
        assert len(list(loader.load(progress=False))) == len(rows)

    def test_dedup_off_keeps_repeats(self, tmp_path, monkeypatch):
        rows = raw_rows()
        doubled = tmp_path / "doubled.jsonl"
        with doubled.open("w") as handle:
            for row in rows + rows:
                handle.write(json.dumps(row) + "\n")

        loader = PubMedCorpusLoader(min_chars=0, dedup=False)
        monkeypatch.setattr(loader, "download_shards", lambda progress=True: [doubled])
        assert len(list(loader.load(progress=False))) == 2 * len(rows)

    def test_blank_lines_in_a_shard_are_skipped(self, tmp_path, monkeypatch):
        messy = tmp_path / "messy.jsonl"
        messy.write_text("\n".join(["", json.dumps(raw_rows()[0]), "  ", ""]))
        loader = PubMedCorpusLoader(min_chars=0)
        monkeypatch.setattr(loader, "download_shards", lambda progress=True: [messy])
        assert len(list(loader.load(progress=False))) == 1


class TestLocalShardDir:
    def test_missing_local_shard_names_the_file(self, tmp_path):
        loader = PubMedCorpusLoader(num_shards=1, shard_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="pubmed23n0001.jsonl"):
            loader.download_shards()

    def test_present_local_shards_are_used_without_download(self, tmp_path):
        (tmp_path / "pubmed23n0001.jsonl").write_text(json.dumps(raw_rows()[0]) + "\n")
        loader = PubMedCorpusLoader(num_shards=1, shard_dir=tmp_path, min_chars=0)
        assert len(list(loader.load(progress=False))) == 1


class TestStats:
    def test_reports_document_count_and_shard_range(self, local_loader):
        stats = local_loader(min_chars=0).stats(progress=False)
        assert stats["n_documents"] == len(raw_rows())
        assert stats["shards"] == "1..8"
        assert stats["mean_chars"] > 0


class TestRegistryWiring:
    def test_pubmed_is_discoverable(self):
        assert "pubmed" in available_loaders()

    def test_get_loader_accepts_shard_budget(self):
        loader = get_loader("pubmed", num_shards=3, shard_offset=10)
        assert loader.num_shards == 3 and loader.shard_offset == 10
