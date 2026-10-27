"""One-call entry point: :func:`run`.

``threadfin.run`` answers three questions about a paired single-cell
RNA-seq + BCR-seq dataset, in this order:

1. **Is cell state inherited within clones?** - the *clonal coherence*: how
   much of the transcriptional variation inside each sample is explained by
   which clone a cell belongs to, tested against shuffled clone labels.
2. **Which clones behave alike?** - genetically distinct clones are grouped
   into *clonal programmes* by the state bias of their cells (for example
   "plasmablast-biased" or "germinal-centre-retained" clones). Every split
   between programmes must pass a significance test, and every programme
   gets a bootstrap stability score; if no split is significant, a single
   programme is reported (the clones differ, but not as groups the test can
   separate).
3. **Do your labels explain how clones differ?** - optional clone-level
   tests against any labels you have (antigen specificity, isotype,
   infection, disease group): first whether the label explains clone
   profiles at all, then which programmes differ; and clonal memory across
   time points.

Every step is also available on its own in :mod:`threadfin.tl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._utils import ensure_actual, log, require_complete_groups, require_obs, require_positive_int

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
    locus = locus.reindex(tab.index)  # UMI sorting changes row order; keep chain labels aligned.
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
    profile_tests: dict = field(default_factory=dict, repr=False)
    memory: dict | None = field(default=None, repr=False)
    settings: dict = field(default_factory=dict, repr=False)
    #: analyses that were requested but could not be run on these data, with the reason
    skipped: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        n_prog = self.programmes.shape[0]
        n_cl = int(self.clones["clone_programme"].notna().sum()) if "clone_programme" in self.clones else 0
        return (f"ThreadfinResult(clonal ICC={self.coherence['icc']:.3f}, "
                f"p={self.coherence['p_value']:.3g}, programmes={n_prog}, clones in programmes={n_cl})")

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
            f"Cells with a {self.settings.get('receptor') or 'clone id'}: {clones.size:,} of {ad.n_obs:,}, "
            f"in {sizes.size:,} clones "
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
        n_half = vc.min_cells_for(0.5)
        lines.append(f"   Profiles are shrunk towards the mean, so a clone reaches reliability 0.5 at "
                     f"{n_half} {'cell' if n_half == 1 else 'cells'}. No clone is filtered out on reliability.")
        n_profiled = int(prof_vc["n_clones"])
        if n_profiled < 10:
            lines.append(f"   Caution: only {n_profiled} clones have two or more cells. Estimates from so few clones "
                         "are unstable; Threadfin's power analyses need about 10 clones to detect a clonal ICC "
                         "of 0.1, and more for weaker effects.")
        prog = self.programmes
        lines += ["", "2. Which clones behave alike? (clonal programmes)"]
        if prog.empty:
            lines.append("   Not enough reliably profiled clones to define programmes in this dataset.")
        elif prog.shape[0] == 1:
            tests = get_programmes(ad).get("split_tests", pd.DataFrame())
            p_root = f" (split test p = {tests['pvalue'].iloc[0]:.2g})" if not tests.empty else ""
            lines.append(f"   No distinct programmes: the {int(prog['n_clones'].sum()):,} reliable clones differ, "
                         f"but no split into separate groups was significant{p_root}.")
        else:
            params = get_programmes(ad)["params"]
            n_stable = int((prog["stability"] >= params["stability_threshold"]).sum())
            verb = "is" if n_stable == 1 else "are"
            lines.append(f"   {prog.shape[0]} programmes from {int(prog['n_clones'].sum()):,} reliable clones "
                         f"(every split significant); {n_stable} {verb} stable "
                         f"(bootstrap Jaccard >= {params['stability_threshold']}).")
        for name, row in (prog.iterrows() if prog.shape[0] > 1 else []):
            desc = ""
            if self.composition is not None and name in self.composition.index:
                comp = self.composition.loc[name].sort_values(ascending=False)
                desc = "; " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in comp.head(2).items())
            flag = "" if row["stability"] >= 0.6 else "  [unstable - do not interpret]"
            lines.append(f"   {name}: {int(row['n_clones']):,} clones, {int(row['n_cells']):,} cells, "
                         f"stability {row['stability']:.2f}{desc}{flag}")
        section = 3                     # the optional sections are numbered in the order they appear
        if self.profile_tests or self.tests:
            lines += ["", f"{section}. Do your labels explain how clones differ? (clone-level tests)"]
            section += 1
        for label, r in self.profile_tests.items():
            verdict = "explains" if r["p_value"] < 0.05 else "does not significantly explain"
            lines.append(f"   {label} {verdict} clone states: {100 * r['r2']:.1f}% of clone-profile variance "
                         f"vs {100 * r['null_mean']:.1f}% for shuffled labels (p = {r['p_value']:.3g}, "
                         f"{r['n_clones']} clones).")
        if self.tests:
            lines.append("   Programme differences (FDR < 0.05):")
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
                f"{section}. Do clones keep their state over '{m['time_key']}'? (clonal memory)",
                f"   {m['n_clones']} clones seen at >= 2 levels: memory index {m['memory_index']:.2f} "
                f"[{lo:.2f}, {hi:.2f}], p = {m['p_value']:.3g} (1 = clones stay as distinct as they were, "
                "0 = no more similar to their earlier self than to a random clone).",
            ]
            if "persistence" in m:
                lines.append(f"   Same programme at the next level: {100 * m['persistence']:.0f}% of clones vs "
                             f"{100 * m['persistence_null']:.0f}% expected for random clones.")
        if self.skipped:
            lines += ["", "Not run (everything above is complete):"]
            lines += [f"   {what}: {why}" for what, why in self.skipped.items()]
        lines += ["", "Per-cell programme labels: adata.obs['clone_programme']; "
                      "per-clone table: result.clones."]
        return "\n".join(lines)

    def methods(self, clone_ids: pd.Series | None = None) -> str:
        """A Methods paragraph for this analysis, with the settings and package versions that were used.

        Written from the settings recorded by :func:`run`, followed by the references it cites. Check it
        against your own description of the data before using it.

        Parameters
        ----------
        clone_ids
            When the clones were called before :func:`run` (``adata.obs['clone_id']`` filled from
            :func:`threadfin.define_clones_dandelion`), pass the Series that call returned: its recorded
            settings are then described too.
        """
        import platform
        from importlib.metadata import PackageNotFoundError, version

        from . import __version__

        def ver(dist):
            try:
                return version(dist)
            except PackageNotFoundError:
                return "unknown"

        def n(x):
            return f"{int(x):,}"

        ad, st = self.adata, self.settings
        uns = ad.uns.get("threadfin", {})
        refs = {"threadfin": "Threadfin (https://github.com/Reynold4k/Threadfin)"}
        text = [f"Clonal analyses used Threadfin {__version__} (Python {platform.python_version()}, "
                f"scanpy {ver('scanpy')}, anndata {ver('anndata')})."]

        # clones
        cc = st.get("clone_calling")
        if cc is None and clone_ids is not None and clone_ids.attrs.get("method") == "dandelion":
            cc = dict(clone_ids.attrs)
        clones = ad.obs[st["clone_key"]].dropna().astype(str)
        sizes = clones.value_counts()
        if cc:
            ident = cc.get("identity")

            def identity_for(kind, default):
                value = float(ident.get(kind, default)) if isinstance(ident, dict) else float(ident)
                return ("identical junction amino-acid sequences" if value >= 1
                        else f"a junction amino-acid identity of at least {value:g}")

            light = cc.get("use_light_chain", True)
            receptor = st.get("receptor") or cc.get("receptor") or "BCR"
            if receptor == "TCR":
                chains, ident_txt, vdj = ("the beta and alpha chains" if light else "the beta chain",
                                          identity_for("tr-ab", 1.0), "a beta (or delta) chain")
            elif receptor == "BCR":
                chains, ident_txt, vdj = ("the heavy and light chains" if light else "the heavy chain",
                                          identity_for("ig", 0.85), "a heavy chain")
            else:
                chains = ("both chains (heavy and light; beta and alpha)" if light else "the heavy (beta) chain")
                ident_txt = f"{identity_for('ig', 0.85)} for B-cell and {identity_for('tr-ab', 1.0)} for T-cell receptors"
                vdj = "a heavy (beta) chain"
            # run() passes its own column of donors to the clone caller; an earlier call names the user's column
            donor = st.get("donor_key") if st.get("clone_calling") else cc.get("donor_key")
            text.append(
                f"Clones were called with Dandelion {cc.get('dandelion_version', '')} (ref. dandelion)"
                + (f" within each donor ({donor!r})" if donor else "")
                + f", from productive contigs, requiring the same V and J genes and junction length on {chains} and "
                f"{ident_txt}" + (", after Dandelion's contig check" if cc.get("check_contigs") else "") + ". "
                + (f"Cells without {vdj} were not assigned to a clone. " if cc.get("require_heavy_chain", True) else "")
                + (f"Python's hash seed was fixed ({cc['python_hash_seed']}) so that clone calls are reproducible."
                   if cc.get("python_hash_seed") is not None else
                   "Python's hash seed was not fixed, so clone calls can differ slightly between sessions."))
            refs["dandelion"] = ("Suo C et al. Dandelion uses the single-cell adaptive immune receptor repertoire "
                                 "to explore lymphocyte developmental origins. Nat Biotechnol 2024;42:40-51.")
        else:
            text.append(f"Clone identities were taken from adata.obs['{st['clone_key']}'].")
        text.append(f"{n(sizes.sum())} of {n(ad.n_obs)} cells belonged to {n(sizes.size)} clones, "
                    f"{n((sizes >= 2).sum())} of them with two or more cells.")

        # embedding
        emb = uns.get("prepare_embedding")
        if emb and st.get("basis") == emb.get("key_added", "X_threadfin"):
            batch = emb.get("batch_key")
            text.append(
                f"Counts were normalised to 10,000 per cell and log-transformed; {n(emb['n_top_genes'])} highly "
                f"variable genes (Seurat flavour{f', batch-aware over {batch!r}' if batch else ''}) were selected "
                + ("after excluding immunoglobulin and T-cell receptor genes, " if emb.get("exclude_receptor_genes")
                   else "") + f"scaled and reduced to {emb['n_comps']} principal components"
                + (f", integrated over {batch!r} with Harmony (ref. harmony)" if emb.get("integrate") == "harmony"
                   else "") + ".")
            if emb.get("integrate") == "harmony":
                refs["harmony"] = ("Korsunsky I et al. Fast, sensitive and accurate integration of single-cell "
                                   "data with Harmony. Nat Methods 2019;16:1289-1296.")
        else:
            text.append(f"Cell states were the coordinates of the embedding '{st.get('basis')}'.")

        # clone profiles
        prof = uns.get("profiles", {})
        pp_ = prof.get("params", {})
        ctx = pp_.get("context_key")
        centring = (f"Each cell's coordinates were centred on the mean of its sampling context ({ctx!r}). " if ctx
                    else "")
        if pp_.get("representation") == "kernel":
            smooth = int(pp_.get("smooth", 0))
            text.append(
                centring
                + (f"Each cell was averaged with its {smooth} nearest neighbours and " if smooth else "Cells were ")
                + f"mapped to {pp_.get('n_features', 256)} random Fourier features of a Gaussian kernel (bandwidth "
                f"{float(pp_.get('bandwidth', float('nan'))):.3g}, the median distance between cells; ref. rff); every "
                f"clone with at least {pp_.get('min_cells', 2)} cells was described by the mean of its cells' "
                "features (kernel mean embedding; ref. kme), shrunk towards the context average the more, the fewer "
                "cells it has (best linear unbiased prediction under a random-effects model; ref. searle), and the "
                f"profiles were reduced to {pp_.get('n_components', 30)} principal components.")
            refs["rff"] = "Rahimi A, Recht B. Random features for large-scale kernel machines. NeurIPS 2007."
            refs["kme"] = ("Muandet K, Fukumizu K, Sriperumbudur B, Schoelkopf B. Kernel mean embedding of "
                           "distributions: a review and beyond. Found Trends Mach Learn 2017;10:1-141.")
        else:
            text.append(centring + f"Every clone with at least {pp_.get('min_cells', 2)} cells was described by the "
                        "mean of its cells' coordinates, shrunk towards the context average the more, the fewer cells "
                        "it has (best linear unbiased prediction under a random-effects model; ref. searle).")
        refs["searle"] = "Searle SR, Casella G, McCulloch CE. Variance Components. Wiley, 1992."

        # coherence
        coh = self.coherence
        floor = np.isclose(coh["p_value"], 1 / (coh["n_perm"] + 1))
        text.append(
            "Clonal coherence was the intraclass correlation of cell states within clones (the share of variance "
            "explained by clone, on unsmoothed, context-centred coordinates), tested against "
            f"{n(coh['n_perm'])} permutations of clone labels"
            + (f" within {coh['strata_key']!r}" if coh.get("strata_key") else "")
            + f" (observed {coh['icc']:.3f}, shuffled {coh['null_mean']:.3f}, P = {coh['p_value']:.2g}"
            + (", the smallest possible with this number of permutations" if floor else "") + ").")

        # clone map
        mp = prof.get("map_params")
        if mp:
            text.append(f"The clone map is a UMAP (ref. umap) of the clone profiles with n_neighbors "
                        f"{mp['n_neighbors']} and min_dist {mp['min_dist']:g}"
                        + (", chosen by Threadfin's layout search" if mp.get("chosen_by") == "tune_clone_map" else "")
                        + "; it is a display, and no statistic depends on it.")
            refs["umap"] = ("McInnes L, Healy J, Saul N, Grossberger L. UMAP: uniform manifold approximation and "
                            "projection. J Open Source Softw 2018;3:861.")

        # programmes
        prog = uns.get("programmes")
        if prog and not self.programmes.empty:
            pr = prog["params"]
            k = self.programmes.shape[0]
            text.append(
                f"Clonal programmes were Leiden communities (ref. leiden) of the {pr['n_neighbors']}-nearest-neighbour "
                "graph of clone profiles"
                + (" (edges weighted by the clones' reliabilities), " if pr.get("weight_by_reliability", True) else ", ")
                + ("at the resolution chosen for bootstrap stability" if pr.get("auto", True)
                   else f"at resolution {pr['resolution']:g}")
                + f"; every split was tested against a single Gaussian group (alpha {pr['alpha']:g}) and splits "
                "that failed were merged; "
                f"stability was the mean Jaccard overlap with the best-matching group over {pr['n_boot']} bootstrap "
                "replicates (ref. hennig). "
                + (f"This gave {k} programmes." if k > 1 else "No split was significant: one programme."))
            refs["leiden"] = ("Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected "
                              "communities. Sci Rep 2019;9:5233.")
            refs["hennig"] = ("Hennig C. Cluster-wise assessment of cluster stability. Comput Stat Data Anal "
                              "2007;52:258-271.")

        # label tests and memory
        if self.profile_tests:
            r0 = next(iter(self.profile_tests.values()))
            labels = ", ".join(repr(k) for k in self.profile_tests)
            text.append(
                f"For {labels}, the share of the variance between clone profiles explained by the clone-level label "
                f"(as in PERMANOVA; ref. anderson) was tested with {n(r0.get('n_perm', 2000))} permutations of the label "
                + (f"among clones of the same {r0['strata_key']!r}" if r0.get("strata_key") else "among clones")
                + ".")
            refs["anderson"] = ("Anderson MJ. A new method for non-parametric multivariate analysis of variance. "
                                "Austral Ecol 2001;26:32-46.")
        if self.memory is not None:
            text.append(f"Clonal memory over {self.memory['time_key']!r} compared the change of each clone's profile "
                        "between levels, corrected for sampling, with the distance to a random clone of the same donor.")

        body = " ".join(text)
        order = list(dict.fromkeys(k for k in refs if k != "threadfin"))
        for i, key in enumerate(order, 1):
            body = body.replace(f"ref. {key}", f"ref. {i}")
        lines = [body, "", "References"] + [f"{i}. {refs[key]}" for i, key in enumerate(order, 1)]
        return "\n".join(lines)

    def plot(self, save: str | Path | None = None):
        """Overview figure (clone map, coherence test, programme composition, stability)."""
        from .plotting import overview

        return overview(self.adata, state_key=self.settings.get("state_key"), coherence_result=self.coherence, save=save)


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
    counts_layer: str | None = None,
    representation: str = "kernel",
    map_n_neighbors: int | str = "auto",
    map_min_dist: float | str = "auto",
    n_perm: int = 200,
    n_boot: int = 30,
    random_state: int = 0,
    n_jobs: int | None = 1,
    clone_options: dict | None = None,
    verbose: bool = True,
) -> ThreadfinResult:
    """Run the complete Threadfin analysis in one call.

    Parameters
    ----------
    adata
        AnnData of B cells, or T cells (one row per cell). Needs raw counts in
        ``X`` (or in the layer named by ``counts_layer``) unless you pass a
        ready-made ``basis``.
    bcr
        Receptor data, BCR or TCR: a path to a Cell Ranger
        ``filtered_contig_annotations.csv`` or an AIRR ``.tsv``, a contig-level
        DataFrame (one row per receptor chain), or a scirpy AnnData/MuData.
        Clones are defined with Dandelion's paired definition (heavy + light
        chain; TCR beta + alpha) within each donor
        (:func:`threadfin.define_clones_dandelion`); this needs the
        sc-dandelion package. Leave ``None`` if ``adata.obs[clone_key]``
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
    counts_layer
        ``layers`` key holding the raw counts, when ``X`` holds normalised
        values (for example ``"counts"`` after Scanpy's preprocessing). Used
        only to build the embedding (:func:`threadfin.pp.prepare_embedding`).
    representation
        How each clone is summarised for grouping into programmes:
        ``"kernel"`` (default; the whole distribution of the clone's cells,
        so two clones with the same average but different spreads differ)
        or ``"mean"`` (the clone centroid only). The coherence test always
        uses clone means.
    map_n_neighbors, map_min_dist
        UMAP settings of the clone map. ``"auto"`` (default) chooses them
        for this dataset by optimisation (:func:`threadfin.tl.tune_clone_map`),
        so that clones that vary continuously are drawn as one connected
        picture; pass numbers to set them by hand. The map is a display only:
        no statistic depends on it.
    n_perm, n_boot
        Permutations for the coherence test; bootstrap replicates for
        programme stability.
    n_jobs
        Processes for the two most time-consuming steps, the automatic choice
        of the clone-map settings and the bootstrap of programme stability
        (``-1``: all available cores). Results are identical for any value;
        only the run time changes.
    clone_options
        Settings for calling clones from ``bcr``, passed to
        :func:`threadfin.define_clones_dandelion`, e.g. ``{"identity": 0.9}`` or
        ``{"check_contigs": True}`` (Dandelion's check for cells with several chains of one kind).

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
    ensure_actual(adata, verbose)
    from .dynamics import clonal_memory
    from .io import attach_bcr
    from .pp import prepare_embedding
    from .profiles import clone_profiles
    from .programmes import embed_clones, find_programmes, programme_composition
    from .stats import association_test, clonal_coherence, profile_association

    # Check user-supplied column names before mutating ``adata`` or entering a
    # downstream statistical routine.  In particular, a misspelled requested
    # test label must not be silently reported as a skipped analysis.
    test_cols = [test] if isinstance(test, str) else list(test or [])
    require_obs(adata, donor_key, sample_key, state_key, time_key, batch_key, *test_cols,
                context="threadfin.run()")
    require_complete_groups(adata, donor_key, sample_key, batch_key, context="threadfin.run()")
    if counts_layer is not None and counts_layer not in adata.layers:
        raise KeyError(f"counts_layer '{counts_layer}' is not in adata.layers (layers: "
                       f"{', '.join(map(repr, adata.layers.keys())) or 'none'}).")
    require_positive_int(n_perm, "n_perm")
    require_positive_int(n_boot, "n_boot")
    if representation not in {"mean", "kernel"}:
        raise ValueError("representation must be 'mean' or 'kernel'.")

    # Step 1 - clones from receptor sequences (within each donor).
    # Threadfin's clone definition is Dandelion's paired heavy + light definition, so the
    # contig-level table (one row per receptor chain) is required — a per-cell summary cannot
    # recover the light-chain pairing.
    if bcr is not None:
        if not isinstance(bcr, (pd.DataFrame, str, Path)) and not (hasattr(bcr, "mod") or hasattr(bcr, "obsm")):
            raise TypeError("bcr must be a contig-level pandas DataFrame, a path to a 10x "
                            "filtered_contig_annotations.csv / AIRR .tsv file, or a scirpy AnnData/MuData.")
        from .dandelion import _locus, _receptor_name, define_clones_dandelion, read_contigs

        contigs = read_contigs(bcr, obs_names=adata.obs_names)
        contigs["cell_id"] = contigs["cell_id"].astype(str)
        contigs = contigs[contigs["cell_id"].isin(adata.obs_names.astype(str))]
        receptor = _receptor_name(_locus(contigs))
        if contigs["cell_id"].nunique() == 0:
            raise ValueError(
                "No barcode of the receptor data matches adata.obs_names. Check that both use the same barcode "
                "format; for multi-sample data, harmonise the barcodes first (see "
                "tutorial 2, section 4: prefix the cell_id column of the contig table)."
            )
        if donor_key is not None:
            donor_of_cell = pd.Series(adata.obs[donor_key].astype(str).to_numpy(),
                                      index=adata.obs_names.astype(str))
            contigs["donor"] = contigs["cell_id"].map(donor_of_cell)
        ids = define_clones_dandelion(contigs, cell_key="cell_id",
                                      donor_key="donor" if donor_key else None,
                                      out_col=clone_key, verbose=verbose, **(clone_options or {}))
        clone_calling = dict(ids.attrs)
        if int(ids.notna().sum()) == 0:
            raise ValueError(
                "Matched receptor contigs produced no clone calls. Each retained cell needs "
                "heavy-chain (TCR: beta or delta) V and J calls plus a non-empty junction/CDR3 sequence."
            )
        if not isinstance(bcr, (str, Path)) or receptor != "BCR":
            # a DataFrame or a scirpy object, or T-cell contigs (read_bcr summarises B-cell chains only):
            # clone ids only
            table = ids.rename(clone_key).to_frame()
            table.index.name = "barcode"
        else:
            # per-cell V/J/C/SHM summary attached beside the Dandelion clone ids
            table = read_bcr(bcr)
            table.index = table.index.astype(str)
            table = table[table.index.isin(ids.index)]
            table[clone_key] = ids.reindex(table.index)
        attach_bcr(adata, table, clone_col=clone_key, summarize=False, verbose=verbose)
    elif clone_key not in adata.obs.columns:
        raise ValueError(f"Pass `bcr`, or provide clone ids in adata.obs['{clone_key}'].")
    else:
        receptor, clone_calling = None, None

    # Threadfin needs clones observed more than once; say so plainly instead of failing deep inside the model
    sizes = adata.obs[clone_key].dropna().astype(str).value_counts()
    if int((sizes >= 2).sum()) < 2:
        raise ValueError(
            f"Only {int((sizes >= 2).sum())} clone(s) have two or more cells ({sizes.size:,} clones, "
            f"{int(sizes.sum()):,} cells with a clone id). Threadfin compares the cells within a clone, so it needs "
            "clones seen at least twice. Check that the receptor barcodes match adata.obs_names (the share of cells with a "
            "clone id), that clones were called within donors, and that the data contain expanded B cells.")

    # Step 2 - expression embedding without immunoglobulin genes
    if basis is None:
        basis = "X_threadfin"
        if basis not in adata.obsm:
            prepare_embedding(adata, counts_layer=counts_layer, batch_key=batch_key, random_state=random_state,
                              verbose=verbose)

    # Step 3 - one profile per clone, relative to its own sample
    clone_profiles(adata, clone_key=clone_key, basis=basis, context_key=sample_key,
                   donor_key=donor_key, representation=representation,
                   random_state=random_state, verbose=verbose)

    # Step 4 - is clone identity informative at all?
    coherence = clonal_coherence(adata, strata_key=sample_key, n_perm=n_perm,
                                 random_state=random_state, verbose=verbose)

    # Step 5 - group clones into programmes (significant splits only), with bootstrap stability
    try:
        # the clone map is drawn separately from programme discovery, over every profiled clone;
        # find_programmes used to do it over its reliability-gated subset, which silently left some
        # datasets with no map at all
        embed_clones(adata, n_neighbors=map_n_neighbors, min_dist=map_min_dist,
                     random_state=random_state, n_jobs=n_jobs, verbose=verbose)
        programmes = find_programmes(adata, n_boot=n_boot, random_state=random_state, n_jobs=n_jobs,
                                     verbose=verbose)
    except ValueError as e:  # too few expanded clones: keep the coherence result
        log(f"no programmes: {e}", verbose)
        programmes = pd.DataFrame(columns=["n_clones", "n_cells", "mean_reliability", "mean_confidence",
                                           "stability"])

    # Step 6 - describe and test the programmes
    composition = None
    tests, profile_tests, skipped = {}, {}, {}
    if not programmes.empty:
        composition = programme_composition(adata, state_key) if state_key else None
    for col in test_cols:
        # does the label explain clone states at all? (needs no programmes)
        try:
            profile_tests[col] = profile_association(adata, col, random_state=random_state, verbose=verbose)
        except ValueError as e:
            skipped[f"clone-level test of '{col}'"] = str(e)
            log(f"profile test '{col}' skipped: {e}", verbose)
        if programmes.shape[0] >= 2:  # which programmes differ? (needs >= 2 programmes)
            try:
                tests[col] = association_test(adata, col, random_state=random_state, verbose=verbose)
            except ValueError as e:
                if str(e) not in skipped.values():  # the profile test already reported the same reason
                    skipped[f"programme test of '{col}'"] = str(e)
                log(f"programme test '{col}' skipped: {e}", verbose)
    memory = None
    if time_key is not None:
        # optional like the label tests: a time point or tissue column that cannot be tested (e.g. no clone seen at
        # two of its levels) is reported, not allowed to discard the analysis that is already complete
        try:
            memory = clonal_memory(adata, time_key, random_state=random_state, verbose=verbose)
        except ValueError as e:
            skipped[f"clonal memory over '{time_key}'"] = str(e)
            log(f"clonal memory over '{time_key}' skipped: {e}", verbose)

    clones = adata.uns["threadfin"]["profiles"]["clone_table"]
    result = ThreadfinResult(
        adata=adata, coherence=coherence, programmes=programmes, clones=clones,
        composition=composition, tests=tests, profile_tests=profile_tests, memory=memory, skipped=skipped,
        settings={"clone_key": clone_key, "donor_key": donor_key, "sample_key": sample_key,
                  "state_key": state_key, "time_key": time_key, "basis": basis,
                  "representation": representation, "receptor": receptor, "clone_calling": clone_calling,
                  "n_perm": int(n_perm), "n_boot": int(n_boot), "random_state": int(random_state)},
    )
    log("done. print(result.summary()) for a readable report.", verbose)
    return result
