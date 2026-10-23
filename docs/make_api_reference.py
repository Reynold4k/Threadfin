#!/usr/bin/env python
"""Write docs/API.md, the reference of every public function, from the docstrings in the package.

The page is generated, so it cannot drift from the code: tests/test_api_reference.py fails when it is out of date.

Usage: python docs/make_api_reference.py          (rewrite docs/API.md)
       python docs/make_api_reference.py --check  (exit 1 if docs/API.md is out of date)
"""
from __future__ import annotations

import inspect
import re
import sys
import textwrap
from pathlib import Path

import threadfin as tf

OUT = Path(__file__).resolve().parent / "API.md"

SECTIONS = [
    ("One call", "The complete analysis and its result.", [
        ("tf.run", tf.run), ("tf.ThreadfinResult", tf.ThreadfinResult),
        ("ThreadfinResult.summary", tf.ThreadfinResult.summary), ("ThreadfinResult.methods", tf.ThreadfinResult.methods),
        ("ThreadfinResult.plot", tf.ThreadfinResult.plot)]),
    ("Reading data and calling clones", "Receptor tables in, one clone id per cell out.", [
        (f"tf.{n}", getattr(tf, n)) for n in ("read_bcr", "read_10x_vdj", "read_airr", "attach_bcr", "read_contigs",
                                               "define_clones_dandelion", "define_clones")]),
    ("Preprocessing: tf.pp", "The cell embedding from which clone profiles are built.",
     [(f"tf.pp.{n}", getattr(tf.pp, n)) for n in tf.pp.__all__]),
    ("Analysis steps: tf.tl", "The steps that `tf.run` calls, and further analyses, one by one.",
     [(f"tf.tl.{n}", getattr(tf.tl, n)) for n in sorted(tf.tl.__all__)]),
    ("Figures: tf.pl", "Every figure takes the AnnData that the analysis wrote to.",
     [(f"tf.pl.{n}", getattr(tf.pl, n)) for n in tf.pl.__all__]),
    ("Simulated data: tf.sim", "Repertoires with a known answer, for testing and teaching.",
     [(f"tf.sim.{n}", getattr(tf.sim, n)) for n in tf.sim.__all__]),
]
LEGACY = ["annotate_specificity", "bcr_reclustering", "bcr_similarity_graph", "clonal_pseudotime", "clone_centroids",
          "clone_distribution", "clonotype_recluster", "community_markers", "community_score", "expansion_index",
          "integration_diagnostics", "joint_embedding", "migration_index", "transition_index", "specificity_enrichment"]


def anchor(name: str) -> str:
    """GitHub's heading anchor: lower case, punctuation other than - and _ removed, spaces as -."""
    return re.sub(r"[^\w\- ]", "", name.lower()).replace(" ", "-")


def signature(name, obj) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return name
    short = name.split(".")[-1] if name.startswith("ThreadfinResult") else name
    return textwrap.fill(f"{short}{sig}", 100, subsequent_indent="    ", break_long_words=False, break_on_hyphens=False)


def first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n")[0].replace("\n", " ").strip()


def render() -> str:
    lines = [f"# API reference (threadfin {tf.__version__})", "",
             "Generated from the docstrings by `python docs/make_api_reference.py`; do not edit by hand. Start with "
             "[`tf.run`](#tfrun): it runs every step below with the recommended settings. The tutorials in "
             "[`tutorials/`](tutorials/) show each function on real data.", ""]
    for title, intro, items in SECTIONS:
        lines += [f"- [{title}](#{anchor(title)})"]
    lines += ["- [Earlier (v3) functions](#earlier-v3-functions)", ""]
    for title, intro, items in SECTIONS:
        lines += [f"## {title}", "", intro, "", "| function | what it does |", "|---|---|"]
        lines += [f"| [`{n}`](#{anchor(n)}) | {first_line(o)} |" for n, o in items]
        lines.append("")
        for n, o in items:
            lines += [f"### {n}", "", "```python", signature(n, o), "```", ""]
            doc = inspect.getdoc(o) or "(no description)"
            lines += ["```text", doc, "```", ""]
    lines += ["## Earlier (v3) functions", "",
              "Kept importable for analyses made with Threadfin 3; new analyses should use the functions above.", "",
              "| function | what it does |", "|---|---|"]
    lines += [f"| `tf.{n}` | {first_line(getattr(tf, n))} |" for n in LEGACY if hasattr(tf, n)]
    return "\n".join(lines) + "\n"


def main(check: bool) -> int:
    text = render()
    if check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("docs/API.md is out of date: run python docs/make_api_reference.py")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
