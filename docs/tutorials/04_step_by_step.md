# Tutorial 4 — Step by step

`threadfin.run` chains the functions in `threadfin.tl`. Calling them yourself lets you change one step —
the sampling context, the number of permutations, the clone map — without rerunning the others, and makes
each decision explicit. This tutorial uses a simulated time course: four donors, each sampled at two time
points, in which clones keep their programme between time points with probability 0.8.

```python
import threadfin as tf

adata = tf.sim.simulate_repertoire(n_timepoints=2, memory=0.8, random_state=0)
```

**The order of the steps, and what each one needs.**

```
cells + clone ids + embedding
        │
        ▼
1. clone_profiles ──► 2. clonal_coherence        (is clone identity informative at all?)
        │
        ├──► 3. embed_clones                       (the clone map, a display)
        ├──► 4. find_programmes                    (groups of clones, if they exist)
        ├──► 5. profile_association / association_test   (do annotations explain clone differences?)
        └──► 6. clonal_memory                      (do clones keep their state over time or tissue?)
```

Every later step reads the profiles stored by step 1 in `adata.uns["threadfin"]["profiles"]`, so step 1
comes first; the others can be run in any order. If you change the cells, clone ids or embedding, rerun
step 1 — Threadfin checks this and refuses to mix results from different inputs.

---

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

**What it computes.** One profile per clone with at least two cells, in four stages:

1. *Centring.* Each cell's coordinates minus the average of its `context_key` (here, one sample per donor
   and time point). A profile therefore says how a clone's cells differ from the other cells sampled with
   them, so differences between samples are not mistaken for differences between clones.
2. *Smoothing* (kernel profiles only). Each cell is averaged with its 15 nearest cells, which de-noises
   clones of two or three cells.
3. *Representation.* `representation="kernel"` maps each cell into 256 random Fourier features of a Gaussian
   kernel; their mean over a clone's cells (a *kernel mean embedding*) describes the clone's whole
   distribution of states, so a clone split between two states differs from a clone sitting between them.
   `"mean"` uses the clone centroid only.
4. *Shrinkage.* A one-way random-effects model separates variance *between* clones from variance *within*
   clones. Each clone's profile is its best linear unbiased predictor: small clones are pulled towards the
   context average, large clones keep their own profile.

**Key parameters.** `context_key` (the nuisance context — Tutorial 2, section 5); `donor_key` (recorded per
clone, used by the tests); `representation` (`"kernel"`, what `threadfin.run` uses, or `"mean"`);
`min_cells=2` (the definition of what can be profiled, not a quality filter).

**Output.** The returned table has one row per clone: `n_cells`, `reliability` (expected squared correlation
between the estimated and the true profile), the donor and the number of contexts the clone was seen in.
The profiles themselves are in `adata.uns["threadfin"]["profiles"]["features"]`.

> **Watch out.** The ICC printed here is computed from the kernel features and smoothed cells; the coherence
> statistic you should report comes from step 2, which uses unsmoothed cells in the embedding itself.

## 2. Clonal coherence

```python
coh = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=200)
```

```
[threadfin] clonal coherence: ICC 0.284 vs 0.003 under within-'context' permutation (p = 0.00498, 200 permutations).
```

**What it computes.** The share of variance in cell state explained by clone identity (an intraclass
correlation, ICC), next to the same quantity after clone labels are shuffled among the cells of each
`strata_key`.

**Why shuffle within strata.** The shuffled clones keep every sample's cell-state composition and clone-size
distribution; only the link between a cell's clone and its state is destroyed. A test that shuffled across
samples would call any difference between samples "clonal", because cells of a clone are sampled together.

**Output.** `coh["icc"]`, `coh["null_mean"]`, `coh["null_q95"]`, `coh["excess"]`, `coh["p_value"]`. The P
value cannot be smaller than 1/(permutations + 1); use more permutations (`n_perm=1000`) for a finer value.

## 3. The clone map

```python
tf.tl.embed_clones(adata)
```

```
[threadfin] clone-map settings: n_neighbors 100, min_dist 0.95; in every layout at least 99.7% of clones form one piece ...
```

**What it computes.** A 2-D UMAP of the clone profiles, one point per clone, with settings chosen
automatically so that a continuum of clones is not broken into islands. Coordinates are added to the clone
table as `x` and `y`. See [Tutorial 3](03_clone_map.md) for how to read and check a map.

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

**What it computes.** Clones are connected to their 15 most similar clones (each connection weighted by the
two clones' reliabilities) and grouped by Leiden community detection. Two safeguards follow:

* *Split test.* Community detection always returns groups, even when clones vary continuously, so every
  split is tested against a single Gaussian group fitted to the same clones; splits that fail are merged.
  If no split survives, one programme is reported.
* *Stability.* Cells are resampled within clones (Poisson bootstrap) and the whole procedure repeated; each
  programme's stability is its mean Jaccard overlap with its best match in the resamples.

**How to read the table.** `stability` ≥ 0.75: stable; 0.6–0.75: present but blurred; < 0.6: do not
interpret. Here no programme reaches 0.75, and P1 (0.51) should not be interpreted — the clones vary more
continuously than three groups suggest. `mean_confidence` is the average share of resamples in which a
clone stayed in its programme. Programme labels are written to `adata.obs["clone_programme"]`.

> **Watch out.** A single programme means no split passed a deliberately conservative test; it is not proof
> of a continuum (Methods, "a note on counting programmes").

## 5. What distinguishes clones

```python
effect = tf.tl.profile_association(adata, "true_programme")
table = tf.tl.association_test(adata, "true_programme")
```

```
[threadfin] profile association 'true_programme': explains 55.3% of clone-profile variance vs 0.3% for permuted labels (p = 0.0005, 894 clones, within-'donor' permutation).
[threadfin] association 'true_programme': 894 clones, 12 programme effects at FDR < 0.05 (within-'donor' permutation).
```

**What they compute.** `profile_association` asks how much of the difference between clone profiles a label
explains (as in PERMANOVA for categories, or distance-based redundancy analysis for numbers). It works
whether or not distinct programmes exist. `association_test` then asks which programmes differ, with
per-programme odds ratios (or rank effects for numbers) and false-discovery-rate control. Both use clones
as the unit and shuffle labels among the clones of each donor.

**Collapsing a cell label to the clone.** `how="majority"` (default for categories), `how="mean"` (numbers)
or `how="fraction:<level>"`, e.g. `tf.tl.profile_association(adata, "isotype", how="fraction:IGHA")`.

## 6. Clonal memory

```python
memory = tf.tl.clonal_memory(adata, "timepoint")
```

```
[threadfin] clonal memory over 'timepoint': 677 clones, 677 snapshot pairs; memory index 0.63 [0.56, 0.71], p = 0.002; same programme 0.64 vs 0.38 for random clones.
```

**What it computes.** For every clone seen at both time points, its change of profile is compared with the
distance to a random other clone of the same donor, after removing the change expected from sampling few
cells. The memory index is 1 when clones keep everything that distinguishes them and 0 when a clone is no
more similar to its earlier self than to a random clone. Here it is 0.63, and 64% of clones were in the same
programme at both time points against 38% for random pairs of clones (the simulation kept a clone's
programme with probability 0.8). The same call works across tissues or sort gates
(`tf.tl.clonal_memory(adata, "tissue")`).

## 7. Further analyses

| Function | Question | What it needs |
|---|---|---|
| `tl.gene_heritability`, `tl.geneset_heritability` | Which genes are clonally inherited? | gene expression in `adata.X` or a layer |
| `tl.lineage_forest`, `tl.lineage_heritability` | Within a clone, do close relatives on the mutation lineage tree share state? | per-cell V-region sequences and germlines |
| `tl.export_lag` | How far behind its germinal-centre relatives is a clone's output, in somatic mutations? | cell states (GC vs output), mutation frequency, isotype |
| `tl.programme_markers` | Which genes mark each programme, with clones as replicates? | gene expression |
| `tl.programme_composition` | Which cell states make up each programme, with clones weighted equally? | a cell-state annotation |
| `tl.map_quality` | Is a clone map faithful and connected? | profiles and map coordinates |

The statistical model behind each step is described in [Methods](../METHODS.md). A complete analysis of a
published dataset, from download to biology, is [Tutorial 5](05_end_to_end_tutorial.ipynb).
