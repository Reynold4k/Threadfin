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
Euclidean features, matching that particular step of the historical notebook:

```python
tf.clonotype_recluster(
    adata, basis="X_umap", preset="continuous",
    embedding_mode="distance_profiles", random_state=123,
)
```

This does not recreate the historical clone definitions, filtering or Scanpy
neighbour graph. The archived GSE246382 reference comes from the original report and
is not evidence that a new package run reproduces its coordinates. Current
Figure 2B instead uses a fresh, audited run on raw GSE246382 RNA with frozen
same-mouse sequence-defined families, retaining 49 families with at least
three cells. Its continuous preset and seed 123 were fixed before examining
gates/markers; the three presets and two seeds are saved for comparison. The
notebook's executed filtering/resolution also differ from its report methods.
Figure provenance is recorded in
[`legacy_gc_provenance.json`](../paper/figure_plan/assets/legacy_gc_provenance.json).

The default Leiden graph always uses the original clone distances. Changing
`min_dist`, `spread`, `learning_rate`, `umap_n_neighbors` or `embedding_mode`
only changes the display. Measured GC compartments and expression signatures can motivate a
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
calls, excludes receptor genes from PCA, and uses gates/markers only after
clustering. [The preset comparison](../paper/figure_plan/review/GSE246382_reclustering_presets.png)
uses identical families across panels. The primary continuous graph has one
Leiden partition; the displayed state biases are not validated discrete
programmes or future fates.
