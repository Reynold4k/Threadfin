#!/usr/bin/env python
"""Execute every ```python block of Tutorials 1, 3 and 4 in order, exactly as written, as a user copying them would.

Each tutorial runs in its own namespace. All three share one working directory, <out_dir>, which should be empty
at the start; files a tutorial writes there stay for the next one. The full printed output of each block is saved
next to the output shown in the tutorial (the ``` block without a language tag that follows it), so that stale
numbers can be spotted; ```text blocks (diagrams) are not outputs. Writes <out>/tutorial_blocks.json; exits 1 if any
block raises.

Usage: python run_tutorial_blocks.py <docs/tutorials> <out_dir>
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)


def blocks(md: str):
    """(code, shown_output) for each python block; the output is the next fence if it has no language."""
    found = [(m.group(1), m.group(2), m.start(), m.end()) for m in FENCE.finditer(md)]
    out = []
    for i, (lang, body, _, end) in enumerate(found):
        if lang != "python":
            continue
        shown = found[i + 1][1] if i + 1 < len(found) and found[i + 1][0] == "" else ""
        out.append((body, shown))
    return out


def main(tutorials: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    report, failed = [], 0
    for name in ("01_getting_started.md", "03_clone_map.md", "04_step_by_step.md"):
        ns: dict = {"__name__": "__main__"}
        for k, (code, shown) in enumerate(blocks((tutorials / name).read_text(encoding="utf-8")), 1):
            buf = io.StringIO()
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buf):
                warnings.simplefilter("always")
                try:
                    exec(compile(code, f"{name}:block{k}", "exec"), ns)
                    error = None
                except Exception:                                  # noqa: BLE001
                    error = traceback.format_exc(limit=4)
            printed = buf.getvalue()
            if error:
                failed += 1
            report.append({"tutorial": name, "block": k, "code": code, "error": error, "printed": printed,
                           "shown_in_tutorial": shown,
                           "warnings": sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught})})
            print(f"{name} block {k}: {'ERROR' if error else 'ok'}", file=sys.stderr)
    (out / "tutorial_blocks.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(report)} blocks, {failed} failed", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
