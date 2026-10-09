# Clone reclustering controls

`tf.clonotype_recluster` provides three starting configurations for the
centroid-based workflow. The existing default behaviour is preserved.

| Preset | Intended exploration | Graph/UMAP neighbours | Leiden resolution | UMAP min_dist | spread |
| --- | --- | ---: | ---: | ---: | ---: |
| `cohesive` | Compact, connected groups (聚集连贯) | 20 | 0.3 | 0.1 | 1.0 |
| `continuous` | More space within connected groups (连续连贯) | 20 | 0.1 | 0.4 | 1.0 |
| `discrete` | Local groups and finer partitions (离散) | 10 | 0.8 | 0.05 | 1.0 |

All presets use UMAP learning_rate=1.0. They are practical starting points,
not fitted or validated biological categories. Dataset sampling, clone
definition and the cell representation affect the result. Choose a setting
before interpreting marker patterns; assess stability across nearby settings
rather than selecting the most convincing branch shape.

```python
import threadfin as tf

tf.clonotype_recluster(
    adata, basis="X_pca", min_clone_size=3,
    preset="continuous", random_state=123,
)
tf.pl.clone_map(adata)
print(adata.uns["threadfin"]["clonotype_recluster"])

# Explicit arguments override the preset. UMAP neighbours can be independent
# of the graph used for Leiden; display changes do not redefine clusters.
tf.clonotype_recluster(
    adata, preset="continuous", n_neighbors=25, umap_n_neighbors=20,
    resolution=0.2, min_dist=0.35, spread=1.2, learning_rate=1.0,
    random_state=123,
)
```

`tf.reclustering_presets()` returns editable copies of the preset values.
Requested/effective neighbour counts, seed, basis, preset, UMAP controls,
clone-size filter and embedding mode are saved with the result. Neighbour
counts automatically cap at `number_of_retained_clones - 1`. Missing, blank,
and common string placeholders for clone IDs are excluded. `min_clone_size=3`
means at least three cells; use `4` for strictly greater than three.

`embedding_mode="precomputed"` embeds clone distances directly. The optional
`"distance_profiles"` mode uses each row of the clone distance matrix as
Euclidean features. Pair it with `cluster_on="embedding"` to also use the
historical notebook's Scanpy neighbour graph on the clone UMAP:

```python
tf.clonotype_recluster(
    adata, basis="X_umap", preset="continuous",
    embedding_mode="distance_profiles", cluster_on="embedding",
    n_neighbors=15, umap_n_neighbors=20, resolution=0.3, random_state=123,
)
```

This matches the executed notebook geometry/graph recipe, including Leiden
resolution 0.3 (the report text says 0.1). It does not recreate its clone
definitions or filtering. The archived GSE246382 reference comes from the original report and
is not evidence that a new package run reproduces its coordinates. Current
Figure 2B uses a fresh, audited run on GSE246382 with frozen
same-mouse sequence-defined families, retaining 49 families with at least
three cells. It averages the saved receptor-excluded cell UMAP, then computes
new clone coordinates and three Leiden clusters. Colours are `clone_cluster`;
point area follows captured family size. Historical settings and seed 123
were fixed before examining gates/markers; presets and both seeds are saved
for comparison. The
notebook's executed filtering/resolution also differ from its report methods.
Figure provenance is recorded in
[`legacy_gc_provenance.json`](../paper/figure_plan/assets/legacy_gc_provenance.json).

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

The v4 `tf.run` / `tf.tl.clone_profiles` workflow compares sampling-adjusted
state distributions and remains separate from centroid reclustering. These
presets do not turn an exploratory Leiden partition into a validated v4
programme or learn parameters to force a desired biological result.

The effects of UMAP neighbour count and min_dist follow the
[official UMAP parameter documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html).
The preset values themselves are Threadfin starting choices, not an externally
validated optimal configuration.

For the real-data review, run:

```bash
THREADFIN_DATA=/path/to/public-data python case_studies/run_gc_reclustering.py
python paper/figure_plan/gc_reclustering_panels.py
```

The script verifies raw-cell/metadata alignment against committed family
calls and uses gates/markers only after clustering. The default notebook
pipeline averages saved receptor-excluded cell coordinates. `--pipeline pca`
retains the separate PCA/distance-graph analysis. The
[clone embedding preview](../paper/figure_plan/review/GSE246382_clone_embedding.png)
shows the primary three-cluster map. [The preset comparison](../paper/figure_plan/review/GSE246382_reclustering_presets.png)
uses identical families across panels and yields two or four partitions.
These exploratory partitions are not validated discrete programmes or future fates.
