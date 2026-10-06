# Historical Top2a provenance review

The public input identity is verified; the exact saved Top2a figure input chain remains unresolved.

The official [GSE180920 record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180920)
and [supplementary-file directory](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180920/suppl/)
provide separate day-7 and day-14 GC B-cell RNA counts and metadata from NP-haptenated-antigen immunisation.
The deposited metadata identify two samples per day and the measured cell-cycle/zone annotations.
Downloaded count-column IDs exactly match metadata cell IDs in order.

| Input | Cells | Cells with nonzero raw Top2a | RNA counts SHA-256 |
| --- | ---: | ---: | --- |
| Day 7 | 3,542 | 2,719 | `4a1c34f16e8943681947d103390a85f64a77b7f6179e256a101c321fc9a6149e` |
| Day 14 | 19,757 | 11,388 | `4966bf6e2295651dbe9d8e943adf8dc07a2a0bcb14d13c71906afe7adad9e732` |

At [notebook commit 57a9565](https://github.com/Reynold4k/Threadfin/blob/57a956597df762968ca9fe542449b73c95d0f42a/manuscript.ipynb),
zero-based cells 66–71 explicitly read both days, prefix cluster labels,
concatenate RNA objects with a day label and write `wanquan_mixed7_14.h5ad`.
Cell 80 also concatenates day-7/day-14 IgBLAST receptor-call tables.
Later cells alternatively load day-specific objects. Thus a combined-day
analysis is documented, but it is not the only available notebook state.

Cell 33 stores the Top2a plot but reads `integrated_df`, `umap_df` and
`max_size` from reused variables. Saved execution counts are non-sequential
and reused across notebook states. The processed Windows h5ad objects and
`day7bcr_fmt19.tsv` / `day14bcr_fmt19.tsv` calls are not available here, and
the GEO expression supplement does not supply those reconstructed calls.
The deposited RNA alone cannot identify which saved state produced the plot.
Shape resemblance to report Figure 2.7 is supporting context, not an ID-level match.

Supplementary Figure 8A therefore preserves the image as a historical output
with unresolved exact cohort/day attribution. It is excluded from GSE246382
state/fate claims. Current Figure 2B instead comes from a new audited
GSE246382 raw-RNA run using frozen same-mouse sequence-defined families.
Original source image hashes and the separate new-analysis paths are in
[the provenance manifest](figure_plan/assets/legacy_gc_provenance.json).
