# Threadfin v3 design contract — the discovery layer

*v2 delivered clone-level state inference with validation (communities,
nulls, held-out replication, joint GEX+BCR embedding). v3 adds the
**discovery layer**: functions that connect clone communities to the
biological axes a clonal researcher actually cares about — antigen
specificity, affinity maturation (SHM), tissue/time migration, gene
programmes, and public clones. Rationale and evidence: `docs/GAP_ANALYSIS.md`.*

Design rules (unchanged from v2):

* AnnData in → AnnData out (or tidy `pandas.DataFrame`); no new hard
  dependencies beyond numpy/pandas/scipy/scanpy; optional accelerators stay
  optional.
* Clone = row of `obs[clone_key]` (default `clone_id`); community =
  `obs[cluster_key]` (default `clone_cluster`).
* Every discovery statistic ships with a null model or an explicit
  caveat in the docstring.
* v1/v2 API stays untouched; all additions are new functions/modules.

## Module: `threadfin/specificity.py`

Antigen specificity is the strongest external ground truth in clonal
immunology (LIBRA-seq; author-validated mAb panels; CoV-AbDab). Two routes:

* `annotate_specificity(adata, *, clone_key="clone_id", labels=None,
  label_col=None, reference=None, ref_cols=None, cdr3_col="bcr_cdr3_aa",
  v_col="bcr_v_call", j_col="bcr_j_call", identity=0.85,
  key_added="specificity") -> AnnData`
  * **Label mode** (`labels`: DataFrame with `clone_key` and `label_col`):
    per-clone specificity labels (e.g. author `s_pos_clone`) mapped onto
    `obs[key_added]` via clone identity.
  * **Reference-matching mode** (`reference`: path or DataFrame of an
    antibody database such as CoV-AbDab): per clone, take the modal
    (v_call, j_call, cdr3_aa); match reference entries with the same heavy
    V gene and `sequence.cdr3_similarity >= identity`; assign the hit's
    specificity (e.g. "SARS-CoV-2 S"). Per-clone match table stored in
    `adata.uns[key_added + "_clones"]` (clone, n_cells, best score, hit
    name, specificity).
* `specificity_enrichment(adata, *, cluster_key="clone_cluster",
  specificity_key="specificity") -> DataFrame` — Fisher exact per
  community × specificity with BH FDR; tidy columns `[cluster_key,
  specificity, n_cells, fraction, odds_ratio, pvalue, fdr]`.

## Module: `threadfin/migration.py`

STARTRAC-style clonal dynamics indices (Zhang et al., *Nature* 2018),
re-implemented cleanly at clone level with formulas in the docstrings — the
feature scirpy issue #36 has had open since 2020.

* `clone_distribution(adata, *, clone_key="clone_id", group_key) ->
  DataFrame` — clones × groups matrix of within-clone cell fractions.
* `migration_index(adata, *, clone_key="clone_id", group_key) ->
  DataFrame` — pairwise group migration: migr(g1, g2) = Σ_c p_c,g1 · p_c,g2
  over clones c, where p_c,g is the fraction of clone c's cells in group g.
  Square DataFrame.
* `transition_index(adata, *, clone_key="clone_id", state_key) ->
  DataFrame` — same formula across states.
* `expansion_index(adata, *, clone_key="clone_id", group_key) ->
  Series` — per group: 1 − H/ log n (normalized Shannon entropy of the
  clone-size distribution), STARTRAC-expa analogue.

## Module: `threadfin/programs.py`

* `community_markers(adata, *, cluster_key="clone_cluster", method=
  "wilcoxon", n_genes=25, layer=None) -> DataFrame` — scanpy
  `rank_genes_groups` over cells grouped by community; tidy
  `[community, gene, score, logfoldchange, pval_adj]`.
* `community_score(adata, signatures, *, cluster_key="clone_cluster",
  score_kwargs=None) -> DataFrame` — `scanpy.tl.score_genes` per named
  gene programme, mean score per community; tidy `[community, signature,
  score, n_cells]`.

## Additions to `threadfin/clones.py`

* `shm_gradient_test(adata, *, cluster_key="clone_cluster", order=None,
  mut_col="bcr_mu_count") -> dict` — Kruskal–Wallis across communities; if
  `order` (expected maturation order) is given, also Spearman between
  community median SHM and the order.
* `public_clone_summary(adata, *, clone_key="clone_id", donor_key,
  cluster_key="clone_cluster") -> DataFrame` — clones observed in ≥2
  donors: n_donors, n_cells, dominant community, state purity; enables
  public-vs-private comparisons.

## Validation targets (biological phenomena to reconstruct)

1. **LN vaccine (Kim 2022)** — author labels `s_pos_clone` / ELISA-validated
   mAbs / `nuc_RS_freq_19_312` (SHM) / `tissue` (LN vs blood) already local:
   (i) spike-specific clones concentrate in specific communities;
   (ii) SHM rises d28→d201 within S+ clones (affinity maturation);
   (iii) LN→blood migration index is higher for S+ clones (GC emigrants).
2. **Stephenson COVID** — plasmablast community enriched for
   severe/critical donors; CoV-AbDab-matched clones map to the plasmablast
   community; fine-grained `full_clustering` states.
3. **Flu vaccine** — d0→d7 transition + expansion indices; young vs older.
4. **EBV organoid / tonsil** — community gene programmes recover known
   biology (plasmablast XBP1/JCHAIN/MZB1; GC AICDA/BCL6; IFN programme).

## Deliverables

* 3 new modules + clones.py additions, each with `tests/test_*.py` (small
  synthetic AnnData, same conventions as existing tests).
* `benchmarks/discovery/` pipeline + sbatch wrappers producing committed
  figures/tables for the README.
* README "gap" section condensed from `docs/GAP_ANALYSIS.md` with the new
  biological findings.
