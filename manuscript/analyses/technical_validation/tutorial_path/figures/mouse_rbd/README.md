# GSE287123 RBD reporter (Tutorial 5) (`mouse_rbd`)

threadfin 4.6.0 · 36,188 cells · 1,881 clones with two or more cells · clonal ICC 0.169 (shuffled 0.050, P = 0.005) · 3 programmes

The run: `tf.run(adata, bcr=contigs, donor_key="donor", sample_key="donor", strata_key="sample", time_key="division_gate", test=["rbd_bait", "division_gate", "zone_gate", "isotype"], n_jobs=8)` ([`run_dataset.py`](../../run_dataset.py)); the figures are drawn from the kept result by [`make_gallery.py`](../../make_gallery.py).

| figure | what it shows |
|---|---|
| [Overview (`result.plot()`)](#overview-resultplot) | The figure that `result.plot()` saves: (a) clonal programmes on the clone map, (b) clonal coherence, (c) programme stability, (d) programme sizes. |
| [Clonal coherence](#clonal-coherence) | Share of transcriptional variance explained by clone identity (clonal ICC), observed and with clone labels shuffled within each `sample` (bar: mean of 200 permutations; whisker: their central 95 %). |
| [Clone map: programmes](#clone-map-programmes) | The clone map (UMAP of clone profiles; one point per clone, area by its number of cells) coloured by clonal programme, one panel per programme, every other clone in grey. |
| [Clone map: clone size](#clone-map-clone-size) | The clone map coloured by the number of cells of each clone. |
| [Clone map: profile reliability](#clone-map-profile-reliability) | The clone map coloured by the reliability of each clone's profile (0 to 1; it grows with the number of cells sampled from the clone). |
| [Clone map: donor](#clone-map-donor) | The clone map coloured by donor, one panel per donor, every other clone in grey. |
| [Clone map: rbd_bait](#clone-map-rbd_bait) | The clone map coloured by `rbd_bait`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: division_gate](#clone-map-division_gate) | The clone map coloured by `division_gate`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: zone_gate](#clone-map-zone_gate) | The clone map coloured by `zone_gate`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: isotype](#clone-map-isotype) | The clone map coloured by `isotype`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Programme stability](#programme-stability) | Bootstrap stability of each programme (mean Jaccard overlap with the best-matching group over 30 bootstrap replicates), with the 0.75 (stable) and 0.6 guides. |
| [Programme composition: rbd_bait (cells)](#programme-composition-rbd_bait-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `rbd_bait` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: rbd_bait (clones)](#programme-composition-rbd_bait-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `rbd_bait`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Programme composition: division_gate (cells)](#programme-composition-division_gate-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `division_gate` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: division_gate (clones)](#programme-composition-division_gate-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `division_gate`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Programme composition: zone_gate (cells)](#programme-composition-zone_gate-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `zone_gate` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: zone_gate (clones)](#programme-composition-zone_gate-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `zone_gate`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Programme composition: isotype (cells)](#programme-composition-isotype-cells) | For each programme (one bar), the fraction of the programme's cells in each level of `isotype` (levels beyond the seventh pooled as 'other'). |
| [Programme composition: isotype (clones)](#programme-composition-isotype-clones) | For each programme (one bar), the mean over the programme's clones of each clone's fraction of cells in each level of `isotype`, so that every clone counts once (levels beyond the seventh pooled as 'other'). |
| [Label effects](#label-effects) | For each tested label, the share of the variance between clone profiles that it explains (blue) and the same with the label shuffled among clones of the same donor and of similar size (grey), with the permutation P value. |
| [Programme association: rbd_bait](#programme-association-rbd_bait) | Clone-level association of each programme with `rbd_bait`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level. Filled markers: FDR < 0.05. |
| [Programme association: division_gate](#programme-association-division_gate) | Clone-level association of each programme with `division_gate`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level. Filled markers: FDR < 0.05. |
| [Programme association: zone_gate](#programme-association-zone_gate) | Clone-level association of each programme with `zone_gate`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level. Filled markers: FDR < 0.05. |
| [Programme association: isotype](#programme-association-isotype) | Clone-level association of each programme with `isotype`: Mantel-Haenszel odds ratio within donor x clone-size bins with its 95 % CI, one panel per level; levels whose odds ratios are all 0 or infinite cannot be placed on a log axis and are left out (IGHD). Filled markers: FDR < 0.05. |
| [Clonal memory over division_gate](#clonal-memory-over-division_gate) | How much a clone's profile changes between levels of `division_gate`, against other clones of the same donor seen at the same two levels; the title gives the memory index with its 95 % CI. |
| [Gene heritability](#gene-heritability) | Clonal ICC of every gene expressed in at least 5 % of the cells of clones with two or more cells (receptor genes excluded), by rank; the 12 highest are named (`tl.gene_heritability` with its defaults). |
| [Cells: programmes](#cells-programmes) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each programme's clones, one panel per programme, on the grey of all cells. |
| [Cells: rbd_bait](#cells-rbd_bait) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `rbd_bait`, one panel per level, on the grey of all cells. |
| [Cells: division_gate](#cells-division_gate) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `division_gate`, one panel per level, on the grey of all cells. |
| [Cells: zone_gate](#cells-zone_gate) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `zone_gate`, one panel per level, on the grey of all cells. |
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

## Clone map: rbd_bait

![Clone map: rbd_bait](clone_map_rbd_bait.png)

## Clone map: division_gate

![Clone map: division_gate](clone_map_division_gate.png)

## Clone map: zone_gate

![Clone map: zone_gate](clone_map_zone_gate.png)

## Clone map: isotype

![Clone map: isotype](clone_map_isotype.png)

## Programme stability

![Programme stability](stability.png)

## Programme composition: rbd_bait (cells)

![Programme composition: rbd_bait (cells)](composition_rbd_bait_cells.png)

## Programme composition: rbd_bait (clones)

![Programme composition: rbd_bait (clones)](composition_rbd_bait_clones.png)

## Programme composition: division_gate (cells)

![Programme composition: division_gate (cells)](composition_division_gate_cells.png)

## Programme composition: division_gate (clones)

![Programme composition: division_gate (clones)](composition_division_gate_clones.png)

## Programme composition: zone_gate (cells)

![Programme composition: zone_gate (cells)](composition_zone_gate_cells.png)

## Programme composition: zone_gate (clones)

![Programme composition: zone_gate (clones)](composition_zone_gate_clones.png)

## Programme composition: isotype (cells)

![Programme composition: isotype (cells)](composition_isotype_cells.png)

## Programme composition: isotype (clones)

![Programme composition: isotype (clones)](composition_isotype_clones.png)

## Label effects

![Label effects](label_effects.png)

## Programme association: rbd_bait

![Programme association: rbd_bait](association_rbd_bait.png)

## Programme association: division_gate

![Programme association: division_gate](association_division_gate.png)

## Programme association: zone_gate

![Programme association: zone_gate](association_zone_gate.png)

## Programme association: isotype

![Programme association: isotype](association_isotype.png)

## Clonal memory over division_gate

![Clonal memory over division_gate](memory_division_gate.png)

## Gene heritability

![Gene heritability](heritability.png)

## Cells: programmes

![Cells: programmes](cells_programmes.png)

## Cells: rbd_bait

![Cells: rbd_bait](cells_rbd_bait.png)

## Cells: division_gate

![Cells: division_gate](cells_division_gate.png)

## Cells: zone_gate

![Cells: zone_gate](cells_zone_gate.png)

## Cells: isotype

![Cells: isotype](cells_isotype.png)

## Marker signatures by programme

![Marker signatures by programme](signatures.png)
