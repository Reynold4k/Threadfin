#!/usr/bin/env python
"""Selection on the antibody sequence, measured independently of gene expression.

Somatic hypermutation scatters mutations across the V region. Mutations that
change an amino acid (replacement) are visible to selection; silent ones are
not. A clone whose antibody has been selected for therefore carries more
replacement mutations than mutation alone would produce. The expected share is
not a constant: it depends on that clone's own germline sequence, because the
genetic code makes some positions more likely to change an amino acid than
others. The expectation is therefore computed per clone by mutating its own
germline at random.

This gives a per-clone selection score that uses **only the receptor
sequences**. Comparing it with Threadfin's expression-based description of the
same clones is an orthogonal check: the two have no measurement in common.

Usage:
    python sequence_selection.py <dataset>      # writes results/<dataset>/sequence_selection.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
BASES = ("A", "C", "G", "T")


def _codon_table():
    from threadfin.sequence import _CODON_TABLE

    return _CODON_TABLE


def replacement_fraction(seq: str, germline: str, frame: int) -> tuple[int, int]:
    """Replacement and silent mutations between a V region and its germline."""
    table = _codon_table()
    r = s = 0
    for start in range(frame, min(len(seq), len(germline)) - 2, 3):
        q, g = seq[start:start + 3].upper(), germline[start:start + 3].upper()
        if len(q) < 3 or set(q) - set(BASES) or set(g) - set(BASES) or q == g:
            continue
        aa_q, aa_g = table.get(q, "X"), table.get(g, "X")
        if "X" in (aa_q, aa_g):
            continue
        for a, b in zip(q, g):
            if a != b:
                r += 1 if aa_q != aa_g else 0
                s += 0 if aa_q != aa_g else 1
    return r, s


def expected_replacement_fraction(germline: str, frame: int, n_mut: int, n_draw: int = 200, seed: int = 0) -> float:
    """Share of replacement mutations expected if ``n_mut`` mutations fell at random on this germline."""
    table = _codon_table()
    rng = np.random.default_rng(seed)
    g = germline.upper()
    positions = [i for i in range(frame, len(g) - 2) if g[i] in BASES]
    if not positions or n_mut == 0:
        return np.nan
    hits = []
    for _ in range(n_draw):
        r = s = 0
        for i in rng.choice(positions, size=min(n_mut, len(positions)), replace=False):
            codon_start = frame + 3 * ((i - frame) // 3)
            codon = g[codon_start:codon_start + 3]
            if len(codon) < 3 or set(codon) - set(BASES):
                continue
            new = rng.choice([b for b in BASES if b != g[i]])
            mutated = codon[: i - codon_start] + new + codon[i - codon_start + 1:]
            aa_g, aa_m = table.get(codon, "X"), table.get(mutated, "X")
            if "X" in (aa_g, aa_m):
                continue
            r, s = (r + 1, s) if aa_g != aa_m else (r, s + 1)
        if r + s:
            hits.append(r / (r + s))
    return float(np.mean(hits)) if hits else np.nan


def clone_selection(cells: pd.DataFrame, min_mutations: int = 3) -> pd.DataFrame:
    """Per clone: observed and expected share of replacement mutations, and their difference.

    ``cells`` needs ``clone``, ``seq``, ``germline`` and ``start`` (the germline
    position of the first base, which fixes the reading frame).
    """
    rows = []
    for clone, g in cells.groupby("clone"):
        r_tot = s_tot = 0
        germ, frame = None, 0
        for _, c in g.iterrows():
            start = int(c["start"])
            frame = (3 - start % 3) % 3            # first whole codon of the germline V
            r, s = replacement_fraction(c["seq"], c["germline"], frame)
            r_tot += r
            s_tot += s
            germ = c["germline"]
        n_mut = r_tot + s_tot
        if n_mut < min_mutations or germ is None:
            continue
        expected = expected_replacement_fraction(germ, frame, min(n_mut, 40))
        rows.append({"clone": clone, "n_cells": len(g), "replacement": r_tot, "silent": s_tot,
                     "observed_fraction": r_tot / n_mut, "expected_fraction": expected,
                     "selection": r_tot / n_mut - expected})
    return pd.DataFrame(rows).set_index("clone")


def main(dataset: str):
    from lineage_trees import _sequences
    from run_case_study import prepare

    adata, *_ = prepare(dataset, stamp=lambda m: None)
    cells = _sequences(adata, dataset)
    sel = clone_selection(cells)
    out = HERE / "results" / dataset
    sel.to_csv(out / "sequence_selection.csv")
    print(f"{dataset}: {len(sel)} clones with enough mutations to score")
    print(f"  replacement share: observed {sel['observed_fraction'].mean():.3f}, "
          f"expected by chance {sel['expected_fraction'].mean():.3f}")

    # does sequence-level selection track what the clone's cells were doing?
    import threadfin as tf
    from scipy.stats import spearmanr

    table = pd.read_csv(out / "clone_table.csv", index_col=0)
    joined = table.join(sel.drop(columns=["n_cells"]), how="inner")
    rows = []
    for col in [c for c in joined.columns if c.startswith(("division_gate", "zone_gate", "rbd_bait", "fate"))]:
        v = pd.to_numeric(joined[col], errors="coerce")
        ok = v.notna() & joined["selection"].notna()
        if ok.sum() < 20:
            continue
        rho, p = spearmanr(joined.loc[ok, "selection"], v[ok])
        rows.append({"label": col, "n_clones": int(ok.sum()), "spearman_rho": float(rho), "p_value": float(p)})
    if rows:
        res = pd.DataFrame(rows)
        res.to_csv(out / "sequence_selection_vs_labels.csv", index=False)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
