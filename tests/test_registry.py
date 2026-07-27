import pytest

from medsel import registry
from medsel.data.base import BaseLoader, letter_key
from medsel.schema import QAExample, make_uid


@pytest.fixture
def clean_registry():
    """Isolate global registry state so tests cannot leak into each other."""
    saved = dict(registry._LOADERS)
    saved_discovered = registry._discovered
    yield
    registry._LOADERS.clear()
    registry._LOADERS.update(saved)
    registry._discovered = saved_discovered


def make_dummy(name="dummy"):
    @registry.register_loader(name)
    class DummyLoader(BaseLoader):
        hf_id = "fake/dataset"
        splits = ("train", "test")
        unlabeled_splits = frozenset({"test"})

        def normalize(self, row, idx, split):
            return QAExample(
                uid=make_uid(self.name, split, idx),
                source=self.name,
                split=split,
                question=row["q"],
                options={"A": "a", "B": "b"},
                answer_key=None if split in self.unlabeled_splits else "A",
            )

    return DummyLoader


class TestRegistry:
    def test_registration_sets_name_and_is_retrievable(self, clean_registry):
        cls = make_dummy()
        assert cls.name == "dummy"
        assert registry.get_loader_class("dummy") is cls
        assert isinstance(registry.get_loader("dummy"), cls)

    def test_duplicate_name_is_rejected(self, clean_registry):
        make_dummy("dupe")
        with pytest.raises(ValueError, match="already registered"):
            make_dummy("dupe")

    def test_unknown_name_lists_alternatives(self, clean_registry):
        make_dummy("known")
        with pytest.raises(KeyError, match="known"):
            registry.get_loader_class("nonexistent")

    def test_available_loaders_is_sorted(self, clean_registry):
        make_dummy("zeta")
        make_dummy("alpha")
        available = registry.available_loaders()
        assert available == sorted(available)
        assert {"alpha", "zeta"} <= set(available)

    def test_kwargs_reach_the_constructor(self, clean_registry):
        make_dummy("kw")
        loader = registry.get_loader("kw", hf_id="override/id", config="cfg")
        assert loader.hf_id == "override/id" and loader.config == "cfg"


class TestBaseLoader:
    def test_rejects_unknown_split_with_available_list(self, clean_registry):
        loader = make_dummy("splits")()
        with pytest.raises(ValueError, match="train, test"):
            loader.check_split("validation")

    def test_has_labels_reflects_unlabeled_splits(self, clean_registry):
        loader = make_dummy("labels")()
        assert loader.has_labels("train")
        assert not loader.has_labels("test")

    def test_cache_key_encodes_config_and_version(self, clean_registry):
        loader = make_dummy("ck")(config="pqa_labeled")
        assert loader.cache_key == "ck-pqa_labeled-v1"

    def test_repr_shows_hf_id(self, clean_registry):
        assert "fake/dataset" in repr(make_dummy("repr_test")())


class TestLetterKey:
    def test_maps_index_to_letter(self):
        assert [letter_key(i) for i in range(4)] == ["A", "B", "C", "D"]

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="exceeds supported width"):
            letter_key(99)
