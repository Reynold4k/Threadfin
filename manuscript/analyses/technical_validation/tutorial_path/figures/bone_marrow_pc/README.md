# GSE253857 bone-marrow plasma cells (`bone_marrow_pc`)

threadfin 4.6.0 · 115,144 cells · 6,583 clones with two or more cells · clonal ICC 0.427 (shuffled 0.245, P = 0.005) · 3 programmes

The run: `tf.run(adata, bcr=contigs, donor_key="donor", sample_key="donor", strata_key="sample", batch_key="donor", time_key="tissue", test=["sorted_as", "tissue", "antigen", "isotype"], n_jobs=8)` ([`run_dataset.py`](../../run_dataset.py)); the figures are drawn from the kept result by [`make_gallery.py`](../../make_gallery.py).

| figure | what it shows |
|---|---|
| [Overview (`result.plot()`)](#overview-resultplot) | The figure that `result.plot()` saves: (a) clonal programmes on the clone map, (b) clonal coherence, (c) programme stability, (d) programme sizes. |
| [Clonal coherence](#clonal-coherence) | Share of transcriptional variance explained by clone identity (clonal ICC), observed and with clone labels shuffled within each `sample` (bar: mean of 200 permutations; whisker: their central 95 %). |
| [Clone map: programmes](#clone-map-programmes) | The clone map (UMAP of clone profiles; one point per clone, area by its number of cells) coloured by clonal programme, one panel per programme, every other clone in grey. |
| [Clone map: clone size](#clone-map-clone-size) | The clone map coloured by the number of cells of each clone. |
| [Clone map: profile reliability](#clone-map-profile-reliability) | The clone map coloured by the reliability of each clone's profile (0 to 1; it grows with the number of cells sampled from the clone). |
| [Clone map: donor](#clone-map-donor) | The clone map coloured by donor, one panel per donor, every other clone in grey. |
| [Clone map: sorted_as](#clone-map-sorted_as) | The clone map coloured by `sorted_as`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: tissue](#clone-map-tissue) | The clone map coloured by `tissue`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: antigen](#clone-map-antigen) | The clone map coloured by `antigen`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: isotype](#clone-map-isotype) | The clone map coloured by `isotype`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Programme stability](#programme-stability) | Bootstrap stability of each programme (mean Jaccard overlap with the best-matching group over 30 bootstrap replicates), with the 0.75 (stable) and 0.6 guides. |
| [Programme composition: sorted_as (cells)](#programme-composition-sorted_as-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `sorted_as` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: sorted_as (clones)](#programme-composition-sorted_as-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `sorted_as`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Programme composition: tissue (cells)](#programme-composition-tissue-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `tissue` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: tissue (clones)](#programme-composition-tissue-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `tissue`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Programme composition: antigen (cells)](#programme-composition-antigen-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `antigen` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: antigen (clones)](#programme-composition-antigen-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `antigen`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Programme composition: isotype (cells)](#programme-composition-isotype-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `isotype` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: isotype (clones)](#programme-composition-isotype-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `isotype`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Label effects](#label-effects) | For each tested label, the share of the variance between clone profiles that it explains (blue) and the same with the label shuffled among clones of the same donor and of similar size (grey), with the permutation P value. |
| [Programme association: sorted_as](#programme-association-sorted_as) | Clone-level association of each programme with `sorted_as`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level. Filled markers: FDR < 0.05. |
| [Programme association: tissue](#programme-association-tissue) | Clone-level association of each programme with `tissue`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level. Filled markers: FDR < 0.05. |
| [Programme association: antigen](#programme-association-antigen) | Clone-level association of each programme with `antigen`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level. Filled markers: FDR < 0.05. |
| [Programme association: isotype](#programme-association-isotype) | Clone-level association of each programme with `isotype`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level; levels whose odds ratios are all 0 or infinite cannot be placed on a log axis and are left out (IGHE). Filled markers: FDR < 0.05. |
| [Clonal memory over tissue](#clonal-memory-over-tissue) | How much a clone's profile changes between levels of `tissue`, against other clones of the same donor seen at the same two levels; the title gives the memory index with its 95 % CI. |
| [Gene heritability](#gene-heritability) | Clonal ICC of every gene expressed in at least 5 % of the cells of clones with two or more cells (receptor genes excluded), by rank; the 12 highest are named (`tl.gene_heritability` with its defaults). |
| [Cells: programmes](#cells-programmes) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each programme's clones, one panel per programme, on the grey of all cells. |
| [Cells: sorted_as](#cells-sorted_as) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `sorted_as`, one panel per level, on the grey of all cells. |
| [Cells: tissue](#cells-tissue) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `tissue`, one panel per level, on the grey of all cells. |
| [Cells: antigen](#cells-antigen) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `antigen`, one panel per level, on the grey of all cells. |
| [Cells: isotype](#cells-isotype) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `isotype`, one panel per level, on the grey of all cells. |
| [Marker signatures by programme](#marker-signatures-by-programme) | Mean log-normalised expression of the case-study marker sets (plasma cell, germinal centre, dark zone / cycling, light zone, memory, naive, interferon) in the cells of each programme, z-scored across programmes. |

## Overview (`result.plot()`)

![Overview (`result.plot()`)](overview.png)

## Clonal coherence

![Clonal coherence](coherence.png)

## Clone map: programmes

![Clone map: programmes](clone_map_programmes.png)

## Clone map: clone size

![Clone map: clone size](clone_map_clone_size.png)

## Clone map: profile reliability

![Clone map: profile reliability](clone_map_reliability.png)

## Clone map: donor

![Clone map: donor](clone_map_donor.png)

## Clone map: sorted_as

![Clone map: sorted_as](clone_map_sorted_as.png)

## Clone map: tissue

![Clone map: tissue](clone_map_tissue.png)

## Clone map: antigen

![Clone map: antigen](clone_map_antigen.png)

## Clone map: isotype

![Clone map: isotype](clone_map_isotype.png)

## Programme stability

![Programme stability](stability.png)

## Programme composition: sorted_as (cells)

![Programme composition: sorted_as (cells)](composition_sorted_as_cells.png)

## Programme composition: sorted_as (clones)

![Programme composition: sorted_as (clones)](composition_sorted_as_clones.png)

## Programme composition: tissue (cells)

![Programme composition: tissue (cells)](composition_tissue_cells.png)

## Programme composition: tissue (clones)

![Programme composition: tissue (clones)](composition_tissue_clones.png)

## Programme composition: antigen (cells)

![Programme composition: antigen (cells)](composition_antigen_cells.png)

## Programme composition: antigen (clones)

![Programme composition: antigen (clones)](composition_antigen_clones.png)

## Programme composition: isotype (cells)

![Programme composition: isotype (cells)](composition_isotype_cells.png)

## Programme composition: isotype (clones)

![Programme composition: isotype (clones)](composition_isotype_clones.png)

## Label effects

![Label effects](label_effects.png)

## Programme association: sorted_as

![Programme association: sorted_as](association_sorted_as.png)

## Programme association: tissue

![Programme association: tissue](association_tissue.png)

## Programme association: antigen

![Programme association: antigen](association_antigen.png)

## Programme association: isotype

![Programme association: isotype](association_isotype.png)

## Clonal memory over tissue

![Clonal memory over tissue](memory_tissue.png)

## Gene heritability

![Gene heritability](heritability.png)

## Cells: programmes

![Cells: programmes](cells_programmes.png)

## Cells: sorted_as

![Cells: sorted_as](cells_sorted_as.png)

## Cells: tissue

![Cells: tissue](cells_tissue.png)

## Cells: antigen

![Cells: antigen](cells_antigen.png)

## Cells: isotype

![Cells: isotype](cells_isotype.png)

## Marker signatures by programme

![Marker signatures by programme](signatures.png)
