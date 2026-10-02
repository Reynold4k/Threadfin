"""One-call entry point: :func:`run`.

``threadfin.run`` answers three questions about a paired single-cell
RNA-seq + BCR-seq dataset, in this order:

1. **Does clone identity shape cell state?** - the *clonal coherence*: how
   much of the transcriptional variation inside each sample is explained by
   which clone a cell belongs to, tested against shuffled clone labels.
2. **Which clones behave alike?** - genetically distinct clones are grouped
   into *clonal programmes* by the state bias of their cells (for example
   "plasmablast-biased" or "germinal-centre-retained" clones), each with a
   bootstrap stability score.
3. **What distinguishes the programmes?** - optional clone-level tests
   against any labels you have (antigen specificity, isotype, infection,
   disease group), and clonal memory across time points.

Every step is also available on its own in :mod:`threadfin.tl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._utils import log

# --------------------------------------------------------------------------- reading BCR files


def _read_airr_cells(path) -> pd.DataFrame:
    """AIRR rearrangement TSV -> one row per cell (heavy + best light chain)."""
    from .clones import mutation_frequency

    tab = pd.read_csv(path, sep="\t", low_memory=False)
    cell_col = "cell_id" if "cell_id" in tab.columns else None
    if cell_col is None:
        if "sequence_id" not in tab.columns:
            raise ValueError(f"{path}: no 'cell_id' or 'sequence_id' column.")
        tab["cell_id"] = tab["sequence_id"].astype(str).str.replace(r"_contig_\d+$", "", regex=True)
    if "productive" in tab.columns:
        prod = tab["productive"].astype(str).str.upper().isin(("T", "TRUE", "1"))
        tab = tab[prod]
    locus = tab["locus"].astype(str) if "locus" in tab.columns else tab["v_call"].astype(str).str[:3]
    if {"sequence_alignment", "germline_alignment"} <= set(tab.columns):
        tab["mutation_frequency"] = mutation_frequency(tab)
    count_col = next((c for c in ("umi_count", "duplicate_count", "consensus_count") if c in tab.columns), None)
    if count_col is not None:
        tab = tab.sort_values(count_col, ascending=False, kind="mergesort")
    keep = [c for c in ("v_call", "d_call", "j_call", "c_call", "junction", "junction_aa", "cdr3",
                        "cdr3_aa", "clone_id", "mutation_frequency") if c in tab.columns]
    heavy = tab[locus.str.upper().eq("IGH").to_numpy()].drop_duplicates("cell_id").set_index("cell_id")[keep]
    light = tab[locus.str.upper().isin(("IGK", "IGL")).to_numpy()].drop_duplicates("cell_id").set_index("cell_id")
    out = heavy.copy()
    for col in ("v_call", "j_call", "junction", "junction_aa", "cdr3_aa"):
        if col in light.columns:
            out[f"{col}_light"] = light[col].reindex(out.index)
    if "cdr3" not in out.columns:
        src = "cdr3_aa" if "cdr3_aa" in out.columns else ("junction_aa" if "junction_aa" in out.columns else None)
        if src:
            out["cdr3"] = out[src]
    out.index = out.index.astype(str)
    out.index.name = "barcode"
    return out


def read_bcr(path, *, barcode_prefix: str = "", barcode_suffix: str = "") -> pd.DataFrame:
    """Read a BCR file into one row per cell.

    Accepts the two common formats and detects which one by extension:

    * Cell Ranger ``filtered_contig_annotations.csv`` (``.csv`` / ``.csv.gz``);
    * AIRR rearrangement table (``.tsv`` / ``.tsv.gz``), e.g. Cell Ranger's
      ``airr_rearrangement.tsv`` or an Immcantation ``*_db-pass.tsv``.

    The table is indexed by cell barcode (with optional prefix/suffix, to
    match ``adata.obs_names`` of multi-sample objects) and contains the
    heavy-chain V/D/J/C calls, junction or CDR3 sequences, light-chain
    calls and, for AIRR input with alignments, ``mutation_frequency``.
    """
    from .io import read_10x_vdj

    name = str(path).lower()
    if name.endswith((".csv", ".csv.gz")):
        out = read_10x_vdj(str(path))
    elif name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        out = _read_airr_cells(path)
    else:
        raise ValueError(f"Cannot tell the BCR format of {path}; expected .csv (10x) or .tsv (AIRR).")
    out.index = [f"{barcode_prefix}{b}{barcode_suffix}" for b in out.index.astype(str)]
    out.index.name = "barcode"
    return out


# --------------------------------------------------------------------------- result object


@dataclass
class ThreadfinResult:
    """Everything :func:`run` computed. Print :meth:`summary` for a readable report."""

    adata: object = field(repr=False)
    coherence: dict = field(repr=False)
    programmes: pd.DataFrame = field(repr=False)
    clones: pd.DataFrame = field(repr=False)
    composition: pd.DataFrame | None = field(default=None, repr=False)
    tests: dict = field(default_factory=dict, repr=False)
    memory: dict | None = field(default=None, repr=False)
    settings: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (f"ThreadfinResult(clonal ICC={self.coherence['icc']:.3f}, "
                f"p={self.coherence['p_value']:.3g}, programmes={self.programmes.shape[0]}, "
                f"clones={int(self.clones['clone_programme'].notna().sum())})")

    def summary(self) -> str:
        """Plain-language summary of the results."""
        from .programmes import get_programmes

        ad, coh = self.adata, self.coherence
        clone_key = self.settings["clone_key"]
        clones = ad.obs[clone_key].dropna()
        sizes = clones.value_counts()
        lines = [
            "Threadfin summary",
            "=================",
            f"Cells with a BCR: {clones.size:,} of {ad.n_obs:,}, in {sizes.size:,} clones "
            f"({int((sizes >= 3).sum()):,} clones have >= 3 cells).",
            "",
            "1. Does clone identity shape cell state? (clonal coherence)",
            f"   Clone identity explains {100 * coh['icc']:.1f}% of transcriptional variance within "
            f"'{coh['strata_key']}' strata; shuffled clones explain {100 * coh['null_mean']:.1f}% "
            f"(p = {coh['p_value']:.3g}, {coh['n_perm']} permutations).",
        ]
        prof_vc = ad.uns["threadfin"]["profiles"]["variance_components"]
        from .profiles import VarianceComponents

        vc = VarianceComponents(np.asarray(prof_vc["sigma2"]), np.asarray(prof_vc["tau2"]),
                                prof_vc["n0"], prof_vc["n_clones"], prof_vc["n_cells"],
                                np.zeros(len(prof_vc["tau2"])))
        lines.append(f"   A clone's profile is reliable (>= 0.5) once it has >= {vc.min_cells_for(0.5)} cells.")
        prog = self.programmes
        params = get_programmes(ad)["params"]
        n_stable = int((prog["stability"] >= params["stability_threshold"]).sum())
        lines += [
            "",
            "2. Which clones behave alike? (clonal programmes)",
            f"   {prog.shape[0]} programmes from {int(prog['n_clones'].sum()):,} clones; {n_stable} are "
            f"stable (bootstrap Jaccard >= {params['stability_threshold']}).",
        ]
        for name, row in prog.iterrows():
            desc = ""
            if self.composition is not None and name in self.composition.index:
                comp = self.composition.loc[name].sort_values(ascending=False)
                desc = "; " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in comp.head(2).items())
            flag = "" if row["stability"] >= 0.6 else "  [unstable - do not interpret]"
            lines.append(f"   {name}: {int(row['n_clones']):,} clones, {int(row['n_cells']):,} cells, "
                         f"stability {row['stability']:.2f}{desc}{flag}")
        if self.tests:
            lines += ["", "3. What distinguishes the programmes? (clone-level tests, FDR < 0.05)"]
            for label, tab in self.tests.items():
                sig = tab[tab["fdr"] < 0.05]
                if sig.empty:
                    lines.append(f"   {label}: no significant programme effect.")
                    continue
                for _, r in sig.sort_values("pvalue").head(6).iterrows():
                    if "level" in r:
                        lines.append(
                            f"   {label}: {r['programme']} {'enriched' if r['odds_ratio'] > 1 else 'depleted'} "
                            f"for '{r['level']}' ({100 * r['frac_in_programme']:.0f}% vs "
                            f"{100 * r['frac_elsewhere']:.0f}% of clones; OR {r['odds_ratio']:.2f}, FDR {r['fdr']:.2g})")
                    else:
                        lines.append(
                            f"   {label}: {r['programme']} median {r['median_in_programme']:.3g} vs "
                            f"{r['median_elsewhere']:.3g} elsewhere (FDR {r['fdr']:.2g})")
        if self.memory is not None:
            m = self.memory
            lo, hi = m["memory_index_ci"]
            lines += [
                "",
                f"4. Do clones keep their state over '{m['time_key']}'? (clonal memory)",
                f"   {m['n_clones']} clones seen at >= 2 levels: memory index {m['memory_index']:.2f} "
                f"[{lo:.2f}, {hi:.2f}], p = {m['p_value']:.3g} (1 = clones stay as distinct as they were, "
                "0 = no more similar to their earlier self than to a random clone).",
            ]
            if "persistence" in m:
                lines.append(f"   Same programme at the next level: {100 * m['persistence']:.0f}% of clones vs "
                             f"{100 * m['persistence_null']:.0f}% expected for random clones.")
        lines += ["", "Per-cell programme labels: adata.obs['clone_programme']; "
                      "per-clone table: result.clones."]
        return "\n".join(lines)

    def plot(self, save: str | Path | None = None):
        """Overview figure (clone map, coherence test, programme composition, stability)."""
        from .plotting import overview

        return overview(self.adata, state_key=self.settings.get("state_key"), save=save)


# --------------------------------------------------------------------------- the pipeline


def run(
    adata,
    bcr=None,
    *,
    donor_key: str | None = None,
    sample_key: str | None = None,
    batch_key: str | None = None,
    state_key: str | None = None,
    time_key: str | None = None,
    test=None,
    clone_key: str = "clone_id",
    basis: str | None = None,
    representation: str = "mean",
    n_perm: int = 200,
    n_boot: int = 30,
    random_state: int = 0,
    verbose: bool = True,
) -> ThreadfinResult:
    """Run the complete Threadfin analysis in one call.

    Parameters
    ----------
    adata
        AnnData of B cells (one row per cell). Needs raw counts in ``X`` (or
        ``layers['counts']``) unless you pass a ready-made ``basis``.
    bcr
        BCR data: a path to a Cell Ranger ``filtered_contig_annotations.csv``
        or an AIRR ``.tsv``, or a per-cell DataFrame indexed like
        ``adata.obs_names``. Clones are then defined from the sequences
        (within each donor). Leave ``None`` if ``adata.obs[clone_key]``
        already holds clone ids.
    donor_key
        ``obs`` column naming the individual each cell came from. Strongly
        recommended: clones are defined within donors and tests are
        stratified by donor.
    sample_key
        ``obs`` column naming the sample/library of each cell. Clones are
        compared with the other cells of their own sample, and the null
        model shuffles clone labels only within samples.
    batch_key
        Optional ``obs`` column to integrate over with Harmony when the
        embedding is built (technical batches only).
    state_key
        Optional cell-type/state annotation, used to describe programmes.
    time_key
        Optional time point (or tissue) column; enables the clonal-memory test.
    test
        Optional list of ``obs`` columns to test against programmes at clone
        level (e.g. ``["isotype", "antigen_binding"]``).
    clone_key
        Name of the clone-id column (created when ``bcr`` is given).
    basis
        Existing ``obsm`` embedding to use instead of building one. It
        should exclude immunoglobulin genes (see
        :func:`threadfin.pp.ig_gene_mask`).
    representation
        ``"mean"`` (clone centroid, default) or ``"kernel"`` (whole cell
        distribution of each clone).
    n_perm, n_boot
        Permutations for the coherence test; bootstrap replicates for
        programme stability.

    Returns
    -------
    :class:`ThreadfinResult`. Also adds ``obs[clone_key]`` (if ``bcr`` is
    given), ``obs['clone_programme']`` and results under
    ``adata.uns['threadfin']``.

    Examples
    --------
    >>> import threadfin as tf
    >>> result = tf.run(adata, bcr="filtered_contig_annotations.csv",
    ...                 donor_key="donor", sample_key="sample")
    >>> print(result.summary())
    >>> result.plot("threadfin_overview.pdf")
    """
    from .clones import define_clones
    from .dynamics import clonal_memory
    from .io import attach_bcr
    from .pp import prepare_embedding
    from .profiles import clone_profiles
    from .programmes import find_programmes, programme_composition
    from .stats import association_test, clonal_coherence

    # Step 1 - clones from BCR sequences (within each donor)
    if bcr is not None:
        table = bcr.copy() if isinstance(bcr, pd.DataFrame) else read_bcr(bcr)
        table.index = table.index.astype(str)
        table = table[table.index.isin(adata.obs_names.astype(str))]
        if table.empty:
            raise ValueError("No BCR barcodes match adata.obs_names; check barcode prefixes/suffixes.")
        if donor_key is not None:
            table["donor"] = adata.obs[donor_key].astype(str).reindex(table.index).values
        table = define_clones(table, donor_key="donor" if donor_key else None, out_col=clone_key,
                              verbose=verbose)
        attach_bcr(adata, table.drop(columns=["donor"], errors="ignore"), clone_col=clone_key,
                   summarize=False, verbose=verbose)
    elif clone_key not in adata.obs.columns:
        raise ValueError(f"Pass `bcr`, or provide clone ids in adata.obs['{clone_key}'].")

    # Step 2 - expression embedding without immunoglobulin genes
    if basis is None:
        basis = "X_threadfin"
        if basis not in adata.obsm:
            prepare_embedding(adata, batch_key=batch_key, random_state=random_state, verbose=verbose)

    # Step 3 - one profile per clone, relative to its own sample
    clone_profiles(adata, clone_key=clone_key, basis=basis, context_key=sample_key,
                   donor_key=donor_key, representation=representation,
                   random_state=random_state, verbose=verbose)

    # Step 4 - is clone identity informative at all?
    coherence = clonal_coherence(adata, strata_key=sample_key, n_perm=n_perm,
                                 random_state=random_state, verbose=verbose)

    # Step 5 - group clones into programmes, with bootstrap stability
    programmes = find_programmes(adata, n_boot=n_boot, random_state=random_state, verbose=verbose)

    # Step 6 - describe and test the programmes
    composition = programme_composition(adata, state_key) if state_key else None
    tests = {}
    for col in ([test] if isinstance(test, str) else (test or [])):
        tests[col] = association_test(adata, col, random_state=random_state, verbose=verbose)
    memory = None
    if time_key is not None:
        memory = clonal_memory(adata, time_key, random_state=random_state, verbose=verbose)

    clones = adata.uns["threadfin"]["profiles"]["clone_table"]
    result = ThreadfinResult(
        adata=adata, coherence=coherence, programmes=programmes, clones=clones,
        composition=composition, tests=tests, memory=memory,
        settings={"clone_key": clone_key, "donor_key": donor_key, "sample_key": sample_key,
                  "state_key": state_key, "time_key": time_key, "basis": basis,
                  "representation": representation},
    )
    log("done. print(result.summary()) for a readable report.", verbose)
    return result
