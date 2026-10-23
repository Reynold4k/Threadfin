"""docs/API.md is generated from the docstrings; it must match the code it describes."""

import importlib.util
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_api_reference_is_current():
    spec = importlib.util.spec_from_file_location("make_api_reference", DOCS / "make_api_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (DOCS / "API.md").read_text(encoding="utf-8") == mod.render(), "run: python docs/make_api_reference.py"
