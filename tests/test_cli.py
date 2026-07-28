"""CLI argument handling and the read-only commands."""

import json

import pytest

from medsel.cli import _coerce, _parse_kwargs, build_parser, main


def run(capsys, argv) -> dict | list:
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


class TestCoercion:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("true", True),
            ("False", False),
            ("none", None),
            ("8", 8),
            ("0.05", 0.05),
            ("medmcqa", "medmcqa"),
            ("1e-5", 1e-5),
        ],
    )
    def test_infers_the_obvious_type(self, text, expected):
        assert _coerce(text) == expected

    def test_parses_loader_kwargs(self):
        assert _parse_kwargs(["num_shards=2", "min_chars=300", "dedup=false"]) == {
            "num_shards": 2,
            "min_chars": 300,
            "dedup": False,
        }

    def test_empty_kwargs_is_an_empty_dict(self):
        assert _parse_kwargs(None) == {}

    def test_missing_equals_sign_is_rejected(self):
        with pytest.raises(SystemExit, match="key=value"):
            _parse_kwargs(["num_shards"])


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_data_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["data"])

    def test_stats_requires_a_source(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["data", "stats"])

    def test_train_requires_a_config(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train"])

    def test_train_accepts_overrides(self):
        args = build_parser().parse_args(
            ["train", "--config", "c.yaml", "--max-steps", "5", "--limit", "10"]
        )
        assert args.max_steps == 5 and args.limit == 10


class TestInfo:
    def test_reports_loaders_stages_and_hardware(self, capsys):
        payload = run(capsys, ["info"])
        assert {"medqa", "medmcqa", "pubmedqa", "pubmed"} <= set(payload["loaders"])
        assert payload["stages"] == ["cpt", "sft", "align"]
        assert "device" in payload["hardware"]


class TestDataList:
    def test_lists_every_loader_with_its_record_type(self, capsys):
        rows = {row["name"]: row for row in run(capsys, ["data", "list"])}
        assert rows["pubmed"]["record_type"] == "CorpusDoc"
        assert rows["medmcqa"]["record_type"] == "QAExample"

    def test_surfaces_the_unlabeled_split_trap(self, capsys):
        rows = {row["name"]: row for row in run(capsys, ["data", "list"])}
        assert rows["medmcqa"]["unlabeled_splits"] == ["test"]

    def test_reports_hub_ids(self, capsys):
        rows = {row["name"]: row for row in run(capsys, ["data", "list"])}
        assert rows["pubmed"]["hf_id"] == "MedRAG/pubmed"
