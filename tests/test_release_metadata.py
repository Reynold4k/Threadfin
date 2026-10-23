"""The version is written in four places; they must agree."""

import re
from pathlib import Path

import threadfin as tf

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere():
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    citation = re.search(r"^version: (\S+)", (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M).group(1)
    api = re.search(r"API reference \(threadfin (\S+)\)", (ROOT / "docs" / "API.md").read_text(encoding="utf-8")).group(1)
    assert tf.__version__ == pyproject == citation == api, (tf.__version__, pyproject, citation, api)
