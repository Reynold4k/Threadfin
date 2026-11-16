# Clone reclustering controls (Threadfin 3)

`tf.clonotype_recluster` is the centroid-based clone clustering of Threadfin 3, kept for analyses made with it.
It places every clone at the centroid of its cells in an embedding (`basis`), builds a graph of clones,
clusters it with Leiden and draws a UMAP of the clones. New analyses should use `tf.run` and the functions of
`tf.tl` ([Tutorials 1–4](tutorials/)), which compare the sampling-adjusted state distributions of clones.

Three presets give starting values for the graph and the display. The default, `cohesive`, keeps the behaviour
of Threadfin 3.

| Preset | Intended exploration | Graph/UMAP neighbours | Leiden resolution | UMAP min_dist | spread |
| --- | --- | ---: | ---: | ---: | ---: |
| `cohesive` | compact, connected groups | 20 | 0.3 | 0.1 | 1.0 |
| `continuous` | more space within connected groups | 20 | 0.1 | 0.4 | 1.0 |
| `discrete` | local groups and finer partitions | 10 | 0.8 | 0.05 | 1.0 |

All presets use UMAP learning_rate=1.0. They are practical starting points, not fitted or validated biological
categories. Dataset sampling, clone definition and the cell representation affect the result. Choose a setting
before interpreting marker patterns, and assess stability across nearby settings rather than selecting the most
convincing branch shape.

The examples run on the simulated repertoire of [Tutorial 1](tutorials/01_getting_started.md), which has clone
ids in `obs["clone_id"]` and a cell embedding in `obsm["X_pca"]`:

```python
import threadfin as tf

adata = tf.sim.simulate_repertoire(random_state=0)
tf.clonotype_recluster(
    adata, basis="X_pca", min_clone_size=3,
    preset="continuous", random_state=123,
)
tf.pl.clone_map(adata)
print(adata.uns["threadfin"]["clonotype_recluster"])

# Explicit arguments override the preset. UMAP neighbours can be independent
# of the graph used for Leiden; display changes do not redefine clusters.
tf.clonotype_recluster(
    adata, basis="X_pca", preset="continuous", n_neighbors=25, umap_n_neighbors=20,
    resolution=0.2, min_dist=0.35, spread=1.2, learning_rate=1.0,
    random_state=123,
)
```

Always name the embedding: `basis` defaults to `"X_umap"`, the two-dimensional cell UMAP, as in Threadfin 3,
whereas `"X_pca"` avoids averaging a nonlinear display.

`tf.reclustering_presets()` returns editable copies of the preset values.
Requested/effective neighbour counts, seed, basis, preset, UMAP controls,
clone-size filter and embedding mode are saved with the result. Neighbour
counts automatically cap at `number_of_retained_clones - 1`. Missing, blank,
and common string placeholders for clone IDs are excluded. `min_clone_size=3`
means at least three cells; use `4` for strictly greater than three.

`embedding_mode="precomputed"` embeds clone distances directly. The optional
`"distance_profiles"` mode uses each row of the clone distance matrix as
Euclidean features. Pair it with `cluster_on="embedding"` to cluster a Scanpy
neighbour graph of the clone UMAP, as the historical notebook from which
Threadfin 3 grew did. That recipe averages the cell UMAP, so it needs one:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_pca", random_state=0)      # a cell UMAP, if adata has none
sc.tl.umap(adata, random_state=0)
tf.clonotype_recluster(
    adata, basis="X_umap", preset="continuous",
    embedding_mode="distance_profiles", cluster_on="embedding",
    n_neighbors=15, umap_n_neighbors=20, resolution=0.3, random_state=123,
    cluster_random_state=0,
)
```

This reproduces the notebook's geometry and graph (UMAP seed 123, graph and
Leiden seed 0 as in Scanpy's defaults, Leiden resolution 0.3), not its clone
definitions or filtering.

The default `cluster_on="distances"` graph uses the original clone distances. Changing
`min_dist`, `spread`, `learning_rate`, `umap_n_neighbors` or `embedding_mode`
only changes the display in that mode. Explicit `cluster_on="embedding"`
requires `embed_clones=True`, and UMAP parameters can affect its clustering.
Scanpy neighbour counts include self; the default distance-graph count is
the number of other neighbours. The chosen mode and graph method are saved.
Measured GC compartments and expression signatures can motivate a
GC selection/output hypothesis, but cannot measure direction or future fate.
That interpretation belongs to model-antigen GC studies with supporting
experimental labels; it does not extend to non-GC samples.

The effects of UMAP neighbour count and min_dist follow the
[official UMAP parameter documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html).
The preset values themselves are Threadfin starting choices, not an externally
validated optimal configuration.

The presets were reviewed on germinal-centre data (GSE246382) with
[`run_gc_reclustering.py`](../manuscript/analyses/technical_validation/filter_comparison/run_gc_reclustering.py):

```bash
THREADFIN_DATA=/path/to/public-data python manuscript/analyses/technical_validation/filter_comparison/run_gc_reclustering.py
```

The script verifies raw-cell/metadata alignment against committed family
calls and uses gates/markers only after clustering. The default notebook
pipeline averages saved receptor-excluded cell coordinates. `--pipeline pca`
retains the separate PCA/distance-graph analysis.
These exploratory partitions are not validated discrete programmes or future fates.
