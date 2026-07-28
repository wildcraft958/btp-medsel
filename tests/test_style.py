"""Repo-wide prose style checks.

A style rule stated only in a contributing guide gets forgotten. This one is enforced.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Built from codepoints so this file does not trip its own check.
DASHES = {chr(0x2013): "en dash", chr(0x2014): "em dash"}

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".bib", ".cfg", ".txt"}

SKIP_DIRS = {".git", ".venv", ".ruff_cache", "__pycache__", "data", "runs", "fixtures"}


def _text_files() -> list[Path]:
    found = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            found.append(path)
    return sorted(found)


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_typographic_dashes(path: Path) -> None:
    """Em and en dashes are banned in code, comments and docs alike.

    Use a comma, colon, full stop or parentheses instead; use a hyphen in number ranges.
    Test fixtures are exempt because they are verbatim upstream rows.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    offences = [
        f"{path.relative_to(REPO_ROOT)}:{n}: {name} in {line.strip()[:80]!r}"
        for n, line in enumerate(lines, start=1)
        for char, name in DASHES.items()
        if char in line
    ]
    assert not offences, "\n".join(offences)
