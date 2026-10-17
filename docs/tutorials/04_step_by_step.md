# Tutorial 4 — Step by step

`threadfin.run` chains the functions in `threadfin.tl`. Calling them yourself lets you change one step —
the sampling context, the number of permutations, the clone map — without rerunning the others. This
tutorial uses a simulated time course: four donors, each sampled at two time points, in which clones keep
their programme between time points with probability 0.8.

```python
import threadfin as tf

adata = tf.sim.simulate_repertoire(n_timepoints=2, memory=0.8, random_state=0)
```

## 1. Clone profiles

```python
table = tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor",
                             representation="kernel")
```

```
[threadfin] clone profiles: 962 clones (>= 2 cells); clonal ICC = 0.313 (962 clones, 5896 cells). ...
           n_cells  reliability donor  n_contexts
clone_id
D0|C00055        7     0.775251    D0           2
D0|C00086        5     0.714856    D0           2
D0|C00091       15     0.877449    D0           2
```

Each clone is described relative to the cells of its sampling context (here, one sample per donor and
time point), so differences between samples are not mistaken for differences between clones.
`representation="kernel"` describes the whole distribution of a clone's cells — a clone split between two
states differs from a clone sitting between them — and is what `threadfin.run` uses; `"mean"` uses the
clone centroid only. Small clones are shrunk towards their context, and `reliability` says how well each
profile is determined. No clone is removed for low reliability; a clone needs two cells.

## 2. Clonal coherence

```python
coh = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=200)
```

```
[threadfin] clonal coherence: ICC 0.284 vs 0.003 under within-'context' permutation (p = 0.00498, 200 permutations).
```

The share of variance in cell state explained by clone identity, against clone labels shuffled among the
cells of each sample. The p-value cannot be smaller than 1/(permutations + 1); use more permutations for a
finer p-value.

## 3. The clone map

```python
tf.tl.embed_clones(adata)
```

```
[threadfin] clone-map settings: n_neighbors 100, min_dist 0.95; in every layout at least 99.7% of clones form one piece ...
```

See [Tutorial 3](03_clone_map.md).

## 4. Clonal programmes

```python
programmes = tf.tl.find_programmes(adata)
```

```
           n_clones  n_cells  mean_reliability  mean_confidence  stability  n_clones_with_assigned
programme
P1              539     3109          0.628033         0.519357   0.510800                     539
P2              244     1743          0.621933         0.946038   0.693330                     244
P3              179     1044          0.619189         0.772626   0.684919                     179
```

Programmes are groups of clones with similar profiles. A split between programmes is kept only if it
passes a test against a single group, and each programme's bootstrap stability says how reproducible it
is: here no programme reaches 0.75, and P1 (0.51) should not be interpreted. Programme labels are written
to `adata.obs["clone_programme"]`.

## 5. What distinguishes clones

```python
effect = tf.tl.profile_association(adata, "true_programme")
table = tf.tl.association_test(adata, "true_programme")
```

```
[threadfin] profile association 'true_programme': explains 55.3% of clone-profile variance vs 0.3% for permuted labels (p = 0.0005, 894 clones, within-'donor' permutation).
[threadfin] association 'true_programme': 894 clones, 12 programme effects at FDR < 0.05 (within-'donor' permutation).
```

`profile_association` asks how much of the difference between clone profiles a label explains, and works
whether or not distinct programmes exist. `association_test` then asks which programmes differ, with
per-programme odds ratios (or rank effects for numbers) and false-discovery-rate control. Both use clones
as the unit and shuffle labels among the clones of each donor.

## 6. Clonal memory

```python
memory = tf.tl.clonal_memory(adata, "timepoint")
```

```
[threadfin] clonal memory over 'timepoint': 677 clones, 677 snapshot pairs; memory index 0.63 [0.56, 0.71], p = 0.002; same programme 0.64 vs 0.38 for random clones.
```

For every clone seen at both time points, its change of profile is compared with the distance to a random
other clone of the same donor, after removing the change expected from sampling few cells. The memory
index is 1 when clones keep everything that distinguishes them and 0 when a clone is no more similar to its
earlier self than to a random clone. The same call works across tissues or sort gates.

## 7. Further analyses

| Function | Question |
|---|---|
| `tl.gene_heritability`, `tl.geneset_heritability` | Which genes are clonally inherited? (needs gene expression in `adata.X`) |
| `tl.lineage_forest`, `tl.lineage_heritability` | Within a clone, do close relatives on the mutation lineage tree share state? |
| `tl.export_lag` | How far behind its germinal-centre relatives is a clone's output, in somatic mutations? |
| `tl.programme_markers` | Which genes mark each programme, with clones as replicates? |
| `tl.programme_composition` | Which cell states make up each programme, with clones weighted equally? |

The statistical model behind each step is described in [Methods](../METHODS.md).
