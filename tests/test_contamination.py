"""Contamination-control tests. Offline: nothing here touches the Hub."""

import json

import pytest

from medsel.data.contamination import (
    DEFAULT_CONFIGS,
    EVAL_CONFIGS,
    fingerprint,
    resolve_exclusions,
)


class TestFingerprint:
    def test_is_order_independent(self):
        # Set iteration order must not change a cache key.
        assert fingerprint([3, 1, 2]) == fingerprint([2, 3, 1])

    def test_ignores_duplicates(self):
        assert fingerprint([1, 1, 2]) == fingerprint([1, 2])

    def test_different_sets_differ(self):
        assert fingerprint([1, 2]) != fingerprint([1, 3])

    def test_empty_set_is_named_not_hashed(self):
        # A readable cache key matters more than a uniform one for the common no-filter case.
        assert fingerprint([]) == "none"

    def test_carries_the_count(self):
        assert fingerprint([5, 6, 7]).startswith("3@")


class TestResolveExclusions:
    def test_none_means_no_filtering(self):
        assert resolve_exclusions(None) == frozenset()

    def test_accepts_an_iterable_of_ints(self):
        assert resolve_exclusions([1, 2, 3]) == frozenset({1, 2, 3})

    def test_coerces_stringy_ints(self):
        assert resolve_exclusions(["7", 8]) == frozenset({7, 8})

    def test_reads_a_json_file(self, tmp_path):
        path = tmp_path / "pmids.json"
        path.write_text(json.dumps([11, 22, 33]))
        assert resolve_exclusions(path) == frozenset({11, 22, 33})

    def test_reads_a_json_path_given_as_a_string(self, tmp_path):
        path = tmp_path / "pmids.json"
        path.write_text(json.dumps([44]))
        assert resolve_exclusions(str(path)) == frozenset({44})

    def test_missing_file_names_the_path(self, tmp_path):
        missing = tmp_path / "absent.json"
        with pytest.raises(FileNotFoundError, match="absent.json"):
            resolve_exclusions(missing)

    def test_rejects_an_unknown_pubmedqa_config(self):
        # Caught before any network call, so a typo fails in milliseconds rather than after a
        # dataset download.
        with pytest.raises(ValueError, match="unknown PubMedQA config"):
            resolve_exclusions("pubmedqa:pqa_typo")

    def test_default_config_is_the_evaluation_set_only(self):
        assert DEFAULT_CONFIGS == ("pqa_labeled",)

    def test_known_configs_cover_all_three(self):
        assert set(EVAL_CONFIGS) == {"pqa_labeled", "pqa_artificial", "pqa_unlabeled"}
