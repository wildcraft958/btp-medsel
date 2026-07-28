"""Packing tests. A stand-in tokenizer keeps these free of any transformers dependency."""

import pytest

from medsel.data.packing import pack_documents
from medsel.schema import CorpusDoc


class FakeTokenizer:
    """One token per character, so block boundaries are trivially checkable."""

    eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class NoEosTokenizer(FakeTokenizer):
    eos_token_id = None


def docs(*texts: str) -> list[CorpusDoc]:
    return [
        CorpusDoc(uid=f"pubmed/{i}", source="pubmed", text=t, title="") for i, t in enumerate(texts)
    ]


def pack(documents, **kwargs):
    kwargs.setdefault("progress", False)
    return list(pack_documents(documents, FakeTokenizer(), **kwargs))


class TestBlockShape:
    def test_every_block_is_exactly_block_size(self):
        blocks = pack(docs("a" * 100), block_size=8)
        assert blocks and all(len(b) == 8 for b in blocks)

    def test_remainder_is_dropped_by_default(self):
        # 10 chars + 1 EOS = 11 tokens -> one block of 8, 3 discarded.
        assert len(pack(docs("a" * 10), block_size=8)) == 1

    def test_remainder_kept_when_requested(self):
        blocks = pack(docs("a" * 10), block_size=8, drop_remainder=False)
        assert [len(b) for b in blocks] == [8, 3]

    def test_corpus_shorter_than_one_block_yields_nothing_by_default(self):
        assert pack(docs("abc"), block_size=64) == []

    def test_documents_are_concatenated_across_the_boundary(self):
        blocks = pack(docs("abcd", "efgh"), block_size=5, add_eos=False)
        assert blocks == [[ord(c) for c in "abcde"]]


class TestEos:
    def test_eos_separates_documents(self):
        [block] = pack(docs("ab", "cd"), block_size=6)
        assert block == [ord("a"), ord("b"), 999, ord("c"), ord("d"), 999]

    def test_eos_can_be_disabled(self):
        [block] = pack(docs("ab", "cd"), block_size=4, add_eos=False)
        assert 999 not in block

    def test_tokenizer_without_eos_is_handled(self):
        blocks = list(
            pack_documents(docs("a" * 20), NoEosTokenizer(), block_size=8, progress=False)
        )
        assert all(999 not in b for b in blocks)


class TestTitleHandling:
    def test_title_is_packed_along_with_the_body(self):
        titled = [CorpusDoc(uid="pubmed/1", source="pubmed", title="T", text="body")]
        [block] = list(
            pack_documents(titled, FakeTokenizer(), block_size=7, add_eos=False, progress=False)
        )
        assert block == [ord(c) for c in "T\n\nbody"]


class TestValidation:
    @pytest.mark.parametrize("block_size", [0, -1])
    def test_rejects_non_positive_block_size(self, block_size):
        with pytest.raises(ValueError, match="block_size must be positive"):
            pack(docs("abc"), block_size=block_size)

    def test_empty_corpus_yields_no_blocks(self):
        assert pack([], block_size=8) == []


class TestLaziness:
    def test_is_streaming_not_materialising(self):
        """Packing must not exhaust its input before yielding - corpora exceed memory."""
        consumed = []

        def generate():
            for i in range(1000):
                consumed.append(i)
                yield CorpusDoc(uid=f"pubmed/{i}", source="pubmed", text="a" * 16, title="")

        first = next(pack_documents(generate(), FakeTokenizer(), block_size=8, progress=False))
        assert len(first) == 8
        assert len(consumed) == 1  # one document read, not all thousand
