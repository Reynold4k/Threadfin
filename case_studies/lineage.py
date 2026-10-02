"""Shared helpers for somatic-hypermutation lineage trees (GCtree).

Input: one row per cell with its clone, an aligned V-region sequence and the
matching germline (same length, gaps removed consistently). For every clone
with enough cells a FASTA file is written whose first record ("naive") is the
germline root; GCtree (DeWitt et al. 2018) then infers an abundance-aware
parsimony tree, and each cell is placed on it.

Running GCtree needs a separate environment (``pip install gctree`` plus
PHYLIP ``dnapars``); see ``run_gctree.sh``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_clone_fastas(cells: pd.DataFrame, out: Path, *, min_cells: int = 3, min_length: int = 150) -> pd.DataFrame:
    """Write ``<clone>.fasta`` per clone, rooted at the germline; returns the cell table with FASTA ids.

    ``cells`` needs ``clone``, ``seq`` and ``germline`` (the V region of one
    cell and the matching germline, same length). Cells of one clone can be
    reported over different parts of the V region, and files differ in where
    their germline string starts, so each cell is first located inside the
    clone's longest germline (they carry no mutations, so an exact match
    finds them) and the clone is then trimmed to the stretch covered by all
    of its cells. Without this the same position would be compared across
    cells that are not aligned to each other.

    GCtree needs ids of at most 10 characters, so cells are renamed ``c1``,
    ``c2`` ... within each clone (column ``fasta_id``).
    """
    out.mkdir(parents=True, exist_ok=True)
    keep = []
    for clone, g in cells.groupby("clone"):
        g = g[g["seq"].str.len() == g["germline"].str.len()]
        if len(g) < min_cells:
            continue
        ref = max(g["germline"], key=len)                       # the clone's germline V, as fully as it was seen
        offset = g["germline"].map(ref.find)
        g, offset = g[offset >= 0], offset[offset >= 0]         # drop cells whose germline is a different allele
        if len(g) < min_cells:
            continue
        lo, hi = int(offset.max()), int((offset + g["seq"].str.len()).min())
        if hi - lo < min_length:
            continue
        root = ref[lo:hi]
        name = str(clone).replace("|", "_")
        g = g.assign(fasta_id=[f"c{i + 1}" for i in range(len(g))], file=name)
        seqs = [r["seq"][lo - o:hi - o] for (_, r), o in zip(g.iterrows(), offset)]
        with open(out / f"{name}.fasta", "w") as fh:
            fh.write(f">naive\n{root}\n")
            for fid, seq in zip(g["fasta_id"], seqs):
                fh.write(f">{fid}\n{seq}\n")
        g["mutations"] = [sum(a != b for a, b in zip(seq, root) if a not in "N-." and b not in "N-.")
                          for seq in seqs]
        g["window_length"] = hi - lo
        keep.append(g.drop(columns=[c for c in ("seq", "germline", "start") if c in g.columns]))
    return pd.concat(keep) if keep else pd.DataFrame()


def place_cells(out: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Position of every cell on its clone's best GCtree tree.

    Returns one row per cell: ``depth`` (mutations from the germline root to the
    cell's genotype), ``ancestral`` (the cell's genotype has observed or inferred
    descendants), ``germline`` (identical to the root).
    """
    from ete3 import Tree

    by_file = {f: g.set_index("fasta_id") for f, g in cells.groupby("file")}
    rows = []
    for d in sorted((out / "trees").glob("*")):
        nk = d / "gctree.out.inference.1.nk"
        if not nk.exists() or d.name not in by_file:
            continue
        idmap = {}
        for line in (d / "idmap.txt").read_text().splitlines():
            node, members = line.split(",", 1)
            idmap[node] = members.replace(":", ",").split(",")
        tree = Tree(str(nk), format=1)
        meta = by_file[d.name]
        for node in tree.traverse():
            for fid in idmap.get(node.name, []):
                if fid not in meta.index:
                    continue
                row = meta.loc[fid].to_dict()
                row.update(depth=node.get_distance(tree), ancestral=not node.is_leaf() and not node.is_root(),
                           germline=node.is_root(), n_descendant_genotypes=len(node.get_descendants()))
                rows.append(row)
    return pd.DataFrame(rows)


def within_clone_difference(tab: pd.DataFrame, group_col: str, a, b, measure: str = "depth") -> dict:
    """Mean of ``measure`` in group(s) ``a`` minus group(s) ``b``, within clones having both; Wilcoxon test."""
    from scipy.stats import wilcoxon

    a, b = list(np.atleast_1d(a)), list(np.atleast_1d(b))
    g = tab[tab[group_col].isin(a + b)].assign(_grp=lambda t: np.where(t[group_col].isin(a), "a", "b"))
    m = g.groupby(["clone", "_grp"])[measure].mean().unstack().dropna()
    diff = m["a"] - m["b"] if not m.empty else pd.Series(dtype=float)
    p = wilcoxon(diff).pvalue if (diff != 0).sum() >= 5 else np.nan
    return {"comparison": f"{'+'.join(map(str, a))} vs {'+'.join(map(str, b))}", "measure": measure,
            "n_clones": int(len(m)), "mean_difference": float(diff.mean()) if len(m) else np.nan,
            "median_difference": float(diff.median()) if len(m) else np.nan, "wilcoxon_p": float(p)}
