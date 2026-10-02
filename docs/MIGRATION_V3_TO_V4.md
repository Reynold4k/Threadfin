# Upgrading from Threadfin v3 to v4

Version 4 is a redesign around one idea: **the clone, not the cell, is the
unit of analysis**. The v3 functions are still importable, so existing scripts
keep running, but new analyses should use `threadfin.run()` or the
`threadfin.tl` functions below. The v3 code and its results are preserved at
the git tag `v3.0.0`.

## What replaces what

| v3 | v4 | why it changed |
|---|---|---|
| `build_clone_key` (exact V + J + CDR3 match) | `define_clones` | hypermutated relatives belong to one clone; clones are defined within each donor |
| `clonotype_recluster` / `clone_centroids` | `tl.clone_profiles` + `tl.find_programmes` | clones are compared with the cells sampled alongside them, small clones are shrunk, and groups of clones are reported only when they are genuinely distinct |
| `joint_embedding` with all variable genes | `pp.prepare_embedding` | immunoglobulin genes are excluded, so receptor transcripts cannot make clonal relatives look alike |
| `metrics.state_enrichment` (counts cells) | `tl.profile_association`, `tl.association_test` | clones, not cells, are the replicates; labels are shuffled within donors |
| `clones.community_transition` | `tl.clonal_memory` | a programme label computed from all of a clone's cells is the same at every time point, so its "transitions" cannot change; v4 compares each clone's state at the two time points directly |
| `metrics.clone_state_purity`, `metrics.state_concordance` | `tl.clonal_coherence` | clone labels are shuffled within samples, so differences between samples are not mistaken for clonality |
| (none) | `tl.gene_heritability` | which genes are clonally inherited |
| (none) | `threadfin.run` | the whole analysis in one call |

## Interpreting old results

Results from v3 that relied on cell-level enrichment tests, on community
"transition" matrices over time, or on embeddings that included
immunoglobulin genes should be recomputed with v4 before they are used.
