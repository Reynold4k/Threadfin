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

```text
cells + clone ids + embedding
        │
        ▼
1. clone_profiles ──► 2. clonal_coherence        (are cells of one clone more alike than others?)
        │
        ├──► 3. embed_clones                       (the clone map, a display)
        ├──► 4. find_programmes                    (groups of clones that pass a split test)
        ├──► 5. profile_association / association_test   (do annotations explain clone differences?)
        └──► 6. clonal_memory                      (do clones keep their state over time or tissue?)
```

Every later step reads the profiles stored by step 1 in `adata.uns["threadfin"]["profiles"]`, so step 1
comes first, and `association_test` also needs the programmes of step 4; the others can be run in any order.
If you change the cells, clone ids or embedding, rerun everything from step 1: coherence, programmes and
memory refuse to run on profiles fitted to different inputs, but the other functions do not check.

---

## 1. Clone profiles

```python
table = tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor",
                             representation="kernel")
print(table.head(3))
```

```
[threadfin] clone profiles: 962 clones (>= 2 cells, 5896 cells in all). Profiles are shrunk, so a clone reaches reliability 0.5 at 2 cells; nothing is filtered on that.
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
   average profile of all profiled clones, large clones keep their own profile.

**Key parameters.** `context_key` (the nuisance context — Tutorial 2, section 5); `donor_key` (recorded per
clone, used by the tests); `representation` (`"kernel"`, what `threadfin.run` uses, or `"mean"`);
`min_cells=2` (the definition of what can be profiled, not a quality filter).

**Output.** The returned table has one row per clone: `n_cells`, `reliability` (expected squared correlation
between the estimated and the true profile), the donor and the number of contexts the clone was seen in.
The profiles themselves are in `adata.uns["threadfin"]["profiles"]["features"]`.

> **Watch out.** The reliabilities come from the smoothed kernel features of the profiles, for which the
> expected squared correlation is slightly optimistic (it is exact for unsmoothed cells); the clonal ICC to
> report comes from step 2, which uses unsmoothed cells in the embedding itself.

## 2. Clonal coherence

```python
coh = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=200)
```

```
[threadfin] clonal coherence: ICC 0.284 vs 0.003 under within-'context' permutation (p = 0.00498, 200 permutations).
```

**What it computes.** The share of the variance of cell states (the expression embedding) explained by clone
identity (an intraclass correlation, ICC), next to the same quantity after clone labels are shuffled among the
cells of each `strata_key`.

**Why shuffle within strata.** The shuffled clones keep every sample's cell-state composition and clone-size
distribution; only the link between a cell's clone and its state is destroyed. A test that shuffled across
samples would call any difference between samples "clonal", because cells of a clone are sampled together.
Use the finest sampling unit as `strata_key` (the library, sort gate or tissue sample), also when the profiles
are centred on a coarser context such as the donor; in `tf.run` the same choice is `strata_key=`.

**Output.** `coh["icc"]`, `coh["null_mean"]`, `coh["null_q95"]`, `coh["excess"]`, `coh["p_value"]`. The P
value cannot be smaller than 1/(permutations + 1); use more permutations (`n_perm=1000`) for a finer value.

## 3. The clone map

```python
tf.tl.embed_clones(adata)
```

```
[threadfin] choosing clone-map settings (ladder) on 962 clones; target: at least 99.0% of clones in one piece.
[threadfin]   ladder: n_neighbors 56, min_dist 0.95 -> score 0.736 (smoothed 0.723)
[threadfin]   ladder: n_neighbors 100, min_dist 0.95 -> score 0.724 (smoothed 0.722)
[threadfin]   ladder: n_neighbors 180, min_dist 0.95 -> score 0.711 (smoothed 0.716)
[threadfin] clone-map settings: n_neighbors 100, min_dist 0.95; in every layout at least 99.7% of clones form one piece (local fidelity 0.536, global fidelity 0.914; 72 layouts, 39 s).
[threadfin] clone map: 962 of 962 profiled clones laid out (n_neighbors 100, min_dist 0.95, chosen automatically); 100.0% of clones in one connected piece.
```

**What it computes.** A 2-D UMAP of the clone profiles, one point per clone, with settings chosen
automatically so that a population of clones that is connected in profile space is not broken into islands.
Coordinates are added to the clone table as `x` and `y`. See [Tutorial 3](03_clone_map.md) for how to read and
check a map.

## 4. Clonal programmes

```python
programmes = tf.tl.find_programmes(adata)
print(programmes.to_string())
```

```
[threadfin] programmes: 3 programmes over 962 core clones (resolution 0.2 after merging 1 non-significant split(s)); 0 with bootstrap stability >= 0.75; 962 of 962 profiled clones assigned with posterior >= 0.7.
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
  If no split survives, one programme is reported. Graded variation also departs from one Gaussian group, so
  a programme is a partition of the clone profiles, not proof of a discrete type of clone.
* *Stability.* Cells are resampled within clones (Poisson bootstrap), with the clones, the smoothing and the
  variance components held fixed, and the graph, Leiden and the split test are repeated; each programme's
  stability is its mean Jaccard overlap with its best match in the resamples.

**How to read the table.** `stability` ≥ 0.75: stable; 0.6–0.75: present but blurred; < 0.6: do not
interpret. Here no programme reaches 0.75, and P1 (0.51) should not be interpreted: it is not reproduced when
the cells are resampled. `mean_confidence` is the average share of resamples in which a clone stayed in its
programme. Programme labels are written to `adata.obs["clone_programme"]`.

> **Watch out.** A single programme is not proof of a continuum, and several programmes are not proof of
> discrete types of clones (Methods, "a note on counting programmes").

## 5. What distinguishes clones

```python
effect = tf.tl.profile_association(adata, "true_programme")
table = tf.tl.association_test(adata, "true_programme")
```

```
[threadfin] profile association 'true_programme': explains 55.3% of clone-profile variance vs 0.4% for permuted labels (p = 0.0005, 894 clones, permutation within 'donor' x clone-size bins).
[threadfin] association 'true_programme': 894 clones, 12 programme effects at FDR < 0.05 (permutation within 'donor' x clone-size bins).
```

**What they compute.** `profile_association` asks how much of the difference between clone profiles a label
explains (as in PERMANOVA for categories, or distance-based redundancy analysis for numbers). It works
whether or not distinct programmes exist. `association_test` then asks which programmes differ, with
per-programme odds ratios pooled within donor × clone-size bins and their 95% intervals (or rank effects for numbers) and
false-discovery-rate control. Both use clones as the unit and shuffle labels among the clones of each donor
and of similar size (2, 3–4, 5–9 or ≥ 10 cells; `size_strata=False` for donors only), so that a label that
follows clone size alone, through the shrinkage of small clones, cannot appear to separate programmes. A label
fixed within each sample is flagged (`sample_level`): its comparison is between samples.

**Collapsing a cell label to the clone.** `how="majority"` (default for categories), `how="mean"` (numbers)
or `how="fraction:<level>"`, e.g. `tf.tl.profile_association(adata, "isotype", how="fraction:IGHA")`.

## 6. Clonal memory

```python
memory = tf.tl.clonal_memory(adata, "timepoint")
```

```
[threadfin] clonal memory over 'timepoint': 677 clones, 677 snapshot pairs, compared with other clones seen at the same two levels; memory index 0.69 [0.60, 0.79], p = 0.002; same programme 0.64 vs 0.38 for other clones.
```

**What it computes.** The memory index compares a clone with other clones of its donor seen at the same two
levels, with each snapshot's own sampling noise. `reference="all", noise="pooled"` gives the 4.5 index.
The memory index is 1 when clones keep everything that distinguishes them and 0 when a clone is no
more similar to its earlier self than to another clone seen at both levels. Here it is 0.69, and 64% of clones
were in the same programme at both time points against 38% for other clones seen at both time points (the
simulation kept a clone's programme with probability 0.8). The same call works across tissues or sort gates
(`tf.tl.clonal_memory(adata, "tissue", order=["PP", "MLN", "LP"])`).

Each clone's consecutive levels are paired, so their order matters. Without `order=`, Threadfin uses the order
of an ordered pandas Categorical column, then the numbers in the labels (here `t0` before `t1`), and otherwise
the alphabetical order, with a warning.

## 7. Further analyses

| Function | Question | What it needs |
|---|---|---|
| `tl.gene_heritability`, `tl.geneset_heritability` | Which genes are shared within clones (a clonal ICC per gene), and which gene sets more than genes of similar expression? (error from the spread between genes plus a clone bootstrap, so co-expressed genes are not independent evidence) | gene expression in `adata.X` or a layer |
| `tl.lineage_forest`, `tl.lineage_heritability` | Within a clone, do close relatives on the mutation lineage tree share state? | lineage trees built elsewhere: within-clone cell pairs with their tree distances, and root-to-tip depths |
| `tl.export_lag` | How much less mutated is a clone's output than its own germinal-centre cells? The median within-clone gap with a sign-test interval; with a time column also in days, with an interval | cell states (GC vs output), mutation frequency, isotype |
| `tl.programme_markers` | Which genes mark each programme, with clones as replicates? With `state_key=` and `state=`: which genes differ between programmes *within* one cell state? (Descriptive: the programmes come from the same expression.) | gene expression |
| `tl.landscape_genes` | Which genes change smoothly across the clones, when there are no distinct programmes? | gene expression |
| `tl.programme_composition` | Which cell states make up each programme? `weight="cells"` (default): shares of its cells; `weight="clones"`: the average clone's shares | a cell-state annotation |
| `tl.map_quality` | Is a clone map faithful and connected? | profiles and map coordinates |

The statistical model behind each step is described in [Methods](../METHODS.md). A complete analysis of a
published dataset, from download to biology, is [Tutorial 5](05_end_to_end_tutorial.ipynb).
