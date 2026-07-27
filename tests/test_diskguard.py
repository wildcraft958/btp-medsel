import pytest

from medsel.utils.diskguard import (
    InsufficientDiskSpace,
    free_bytes,
    human_bytes,
    require_free,
)


class TestHumanBytes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0.0 B"), (512, "512.0 B"), (1024, "1.0 KB"), (1024**3, "1.0 GB")],
    )
    def test_formats_units(self, value, expected):
        assert human_bytes(value) == expected


class TestFreeBytes:
    def test_reports_positive_free_space(self, tmp_path):
        assert free_bytes(tmp_path) > 0

    def test_walks_up_to_existing_parent(self, tmp_path):
        missing = tmp_path / "does" / "not" / "exist"
        assert free_bytes(missing) == free_bytes(tmp_path)


class TestRequireFree:
    def test_allows_small_write(self, tmp_path):
        require_free(1024, tmp_path)

    def test_blocks_impossible_write(self, tmp_path):
        with pytest.raises(InsufficientDiskSpace, match="margin"):
            require_free(10**18, tmp_path)

    def test_error_names_the_path(self, tmp_path):
        with pytest.raises(InsufficientDiskSpace, match=str(tmp_path)):
            require_free(10**18, tmp_path)

    def test_margin_is_applied(self, tmp_path):
        available = free_bytes(tmp_path)
        # Exactly the free space fits without a margin but not with one.
        require_free(available, tmp_path, margin=0.0)
        with pytest.raises(InsufficientDiskSpace):
            require_free(available, tmp_path, margin=0.5)

    def test_rejects_negative(self, tmp_path):
        with pytest.raises(ValueError, match="non-negative"):
            require_free(-1, tmp_path)
