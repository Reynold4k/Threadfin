# GSE317692 influenza lung (`flu_lung`)

threadfin 4.6.0 · 4,121 cells · 204 clones with two or more cells · clonal ICC 0.162 (shuffled 0.051, P = 0.005) · 1 programme

The run: `tf.run(adata, bcr=contigs, donor_key="donor", sample_key="donor", strata_key="sample", time_key="tissue", test=["ha_binding", "tissue", "isotype"], n_jobs=8)` ([`run_dataset.py`](../../run_dataset.py)); the figures are drawn from the kept result by [`make_gallery.py`](../../make_gallery.py).

| figure | what it shows |
|---|---|
| [Overview (`result.plot()`)](#overview-resultplot) | The figure that `result.plot()` saves: (a) clonal programmes on the clone map, (b) clonal coherence, (c) programme stability, (d) programme sizes. |
| [Clonal coherence](#clonal-coherence) | Share of transcriptional variance explained by clone identity (clonal ICC), observed and with clone labels shuffled within each `sample` (bar: mean of 200 permutations; whisker: their central 95 %). |
| [Clone map: clone size](#clone-map-clone-size) | The clone map coloured by the number of cells of each clone. |
| [Clone map: profile reliability](#clone-map-profile-reliability) | The clone map coloured by the reliability of each clone's profile (0 to 1; it grows with the number of cells sampled from the clone). |
| [Clone map: donor](#clone-map-donor) | The clone map coloured by donor, one panel per donor, every other clone in grey. |
| [Clone map: ha_binding](#clone-map-ha_binding) | The clone map coloured by `ha_binding`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: tissue](#clone-map-tissue) | The clone map coloured by `tissue`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Clone map: isotype](#clone-map-isotype) | The clone map coloured by `isotype`, the level held by at least half of each clone's cells, one panel per level, every other clone in grey (clones without such a majority appear only in grey). |
| [Label effects](#label-effects) | For each tested label, the share of the variance between clone profiles that it explains (blue) and the same with the label shuffled among clones of the same donor and of similar size (grey), with the permutation P value. |
| [Clonal memory over tissue](#clonal-memory-over-tissue) | How much a clone's profile changes between levels of `tissue`, against other clones of the same donor seen at the same two levels; the title gives the memory index with its 95 % CI. |
| [Gene heritability](#gene-heritability) | Clonal ICC of every gene expressed in at least 5 % of the cells of clones with two or more cells (receptor genes excluded), by rank; the 12 highest are named (`tl.gene_heritability` with its defaults). |
| [Cells: ha_binding](#cells-ha_binding) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `ha_binding`, one panel per level, on the grey of all cells. |
| [Cells: tissue](#cells-tissue) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `tissue`, one panel per level, on the grey of all cells. |
| [Cells: isotype](#cells-isotype) | Cells on a UMAP of the embedding the run used (`X_threadfin`): the cells of each level of `isotype`, one panel per level, on the grey of all cells. |

## Overview (`result.plot()`)

![Overview (`result.plot()`)](overview.png)

## Clonal coherence

![Clonal coherence](coherence.png)

## Clone map: clone size

![Clone map: clone size](clone_map_clone_size.png)

## Clone map: profile reliability

![Clone map: profile reliability](clone_map_reliability.png)

## Clone map: donor

![Clone map: donor](clone_map_donor.png)

## Clone map: ha_binding

![Clone map: ha_binding](clone_map_ha_binding.png)

## Clone map: tissue

![Clone map: tissue](clone_map_tissue.png)

## Clone map: isotype

![Clone map: isotype](clone_map_isotype.png)

## Label effects

![Label effects](label_effects.png)

## Clonal memory over tissue

![Clonal memory over tissue](memory_tissue.png)

## Gene heritability

![Gene heritability](heritability.png)

## Cells: ha_binding

![Cells: ha_binding](cells_ha_binding.png)

## Cells: tissue

![Cells: tissue](cells_tissue.png)

## Cells: isotype

![Cells: isotype](cells_isotype.png)
