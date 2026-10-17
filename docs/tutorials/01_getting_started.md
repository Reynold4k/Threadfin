# Tutorial 1 — Getting started

This tutorial runs the whole Threadfin analysis on a simulated repertoire, so it needs no download and
finishes in about a minute and a half. Every step is the same on real data; [Tutorial 2](02_your_data.md)
shows how to prepare your own files.

## 1. A simulated repertoire

```python
import threadfin as tf

adata = tf.sim.simulate_repertoire(random_state=0)
print(adata)
```

```
AnnData object with n_obs × n_vars = 8467 × 20
    obs: 'clone_id', 'donor', 'context', 'true_programme', 'true_state', 'clone_size'
    uns: 'simulation'
    obsm: 'X_pca'
```

Each row is a cell. `clone_id` is the clone of the cell, `donor` the individual it came from and `context`
the sample (library) it was sequenced in. The simulation also records the truth: every clone belongs to one
of four programmes (`true_programme`), and every cell is in one of six states (`true_state`). As in real
repertoires, most of the 3,000 clones were captured once; 931 clones have two or more cells.

## 2. One call

```python
result = tf.run(
    adata,
    donor_key="donor",            # clones are compared within individuals
    sample_key="context",         # each clone is compared with the cells sampled alongside it
    basis="X_pca",                # an existing embedding; omit it to let Threadfin build one
    state_key="true_state",       # optional: describes the programmes by cell state
    test=["true_programme"],      # optional: labels to test against the clones
)
print(result.summary())
```

```
Threadfin summary
=================
Cells with a BCR: 8,467 of 8,467, in 3,000 clones (494 clones have >= 3 cells).

1. Does clone identity shape cell state? (clonal coherence)
   Clone identity explains 27.2% of transcriptional variance within 'context' strata; shuffled clones explain 0.2% (p = 0.00498, 200 permutations).
   Profiles are shrunk towards the mean, so a clone reaches reliability 0.5 at 3 cells. No clone is filtered out on reliability.

2. Which clones behave alike? (clonal programmes)
   4 programmes from 931 reliable clones (every split significant); 1 is stable (bootstrap Jaccard >= 0.75).
   P1: 285 clones, 1,577 cells, stability 0.75; S5 87%, S1 8%
   P2: 259 clones, 1,979 cells, stability 0.71; S2 53%, S4 19%
   P3: 229 clones, 1,655 cells, stability 0.66; S0 64%, S2 20%
   P4: 158 clones, 1,187 cells, stability 0.72; S3 56%, S5 32%

3. Do your labels explain how clones differ? (clone-level tests)
   true_programme explains clone states: 55.6% of clone-profile variance vs 0.3% for shuffled labels (p = 0.0005, 931 clones).
   ...
```

How to read it:

1. **Clonal coherence.** Clone identity explains 27% of the variation in cell state within samples,
   against 0.2% when clone labels are shuffled among the cells of each sample. The shuffled value is what
   the sampling design alone would produce.
2. **Clonal programmes.** Threadfin found the four simulated programmes. Only splits that pass a test
   against a single group are kept, and the bootstrap stability says how reproducible each programme is
   (≥ 0.75 stable; below 0.6, do not interpret).
3. **Clone-level tests.** A label is tested with clones, not cells, as the unit, and with labels shuffled
   among the clones of each donor.

## 3. The overview figure

```python
result.plot("threadfin_overview.png")
```

![Overview figure](../_static/tutorial_overview.png)

**a**, the clone map, one point per clone, with each programme highlighted in turn. **b**, clonal
coherence against the within-sample null. **c**, bootstrap stability of each programme. **d**, the cell
states of each programme, with every clone weighted equally.

The clone map was laid out with settings chosen automatically for this dataset (see
[Tutorial 3](03_clone_map.md)):

```python
mp = adata.uns["threadfin"]["profiles"]["map_params"]
print(mp["n_neighbors"], mp["min_dist"], mp["quality"]["coherence"])
```

```
56 0.95 0.998
```

## 4. What you get back

```python
result.clones.head()[["n_cells", "reliability", "clone_programme", "confidence"]]
```

```
           n_cells  reliability clone_programme  confidence
clone_id
D0|C00011        7     0.765086              P2    1.000000
D0|C00016        6     0.737834              P1    1.000000
D0|C00019        2     0.499726              P3    0.766667
D0|C00051        4     0.657441              P3    0.866667
D0|C00073        3     0.594096              P1    1.000000
```

| Output | Where | Meaning |
|---|---|---|
| per-clone table | `result.clones` | size, reliability, programme, bootstrap confidence, map coordinates |
| programme of every cell | `adata.obs["clone_programme"]` | the programme of the cell's clone |
| clonal coherence | `result.coherence` | variance explained by clone, permutation null and p-value |
| programmes | `result.programmes` | size and bootstrap stability of each programme |
| label tests | `result.profile_tests`, `result.tests` | variance explained by each label; per-programme effects |
| everything else | `adata.uns["threadfin"]` | profiles, map settings and the full record of each step |

Next: [Tutorial 2 — your own data](02_your_data.md).
