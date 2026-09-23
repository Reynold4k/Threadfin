# Threadfin v2 Design — BCR-sequence-aware clonal-state analysis

Status: design contract for the v2 upgrade. Audience: developers & reviewers.
All new public APIs specified here are the implementation contract; signatures
must match this document.

## Motivation

v1 clusters clones purely by transcriptional-state centroids (GEX), with an
optional crude CDR3 distance blend (equal-length Hamming). Literature
(Benisse, TESSA, CoNGA, mvTCR) shows joint repertoire+transcriptome latent
structure is more informative than either modality alone, and Erlach et al.
2024 shows naive feature concatenation does *not* work — integration must
happen in a coupled latent space. v2 adds a principled, dependency-light,
Benisse-inspired integration layer plus clonal-research readouts, while
keeping every v1 API working unchanged.

## Design principles

1. **Deterministic, no deep-learning dependency.** Instead of Benisse's
   pretrained contrastive CNN (AchillesEncoder, torch), CDR3 sequences are
   embedded with Atchley factors (the same physicochemical basis Benisse/TESSA
   use as input features) via fixed summary statistics. Reproducible, fast,
   no GPU, no model weights to ship.
2. **Sparsity by biology.** BCR similarity is only computed within V/J-family
   blocks (Benisse's SI constraint). This is both the biologically correct
   support set and the scalability fix (no all-pairs CDR3 alignment).
3. **Coupled Laplacian, not ADMM.** Benisse's latent geometry is
   Sigma = Q^-1 with Q = I + 4γ(λ2 L_GEX + L_BCR) — i.e. the regularized
   inverse of a coupled graph Laplacian; latent distances are (scaled)
   commute-time distances of that coupled graph. We compute the same geometry
   directly with normalized-Laplacian eigenmaps of the coupled sparse graph:
   O(n·k·m) instead of O(n^3) ADMM with dense n×n eigendecompositions.
4. **Honesty about what each piece proves.** Diagnostics quantify how much
   each modality drives the joint structure (Benisse's testCor analogues).
   Negative results are reported, not smoothed over.

## Module map

```
threadfin/
  sequence.py   (extend)  CDR3 similarity: BLOSUM62 alignment, Atchley embed,
                          V/J distance, SHM helpers
  bcrgraph.py   (new)     sparse clone×clone BCR similarity graph;
                          sequence-similarity clone definition
  integrate.py  (new)     coupled-graph joint embedding + diagnostics
  clones.py     (new)     clone-level annotations: isotype, SHM, fate tracking
  core.py       (extend)  clonotype_recluster accepts precomputed distances /
                          joint basis; weighted centroids; graph-based
                          pseudotime
  metrics.py    (extend)  modality contribution; community isotype/SHM stats
  plotting.py   (extend)  clone-map coloring by isotype/SHM; fate alluvial
```

## API contract

### sequence.py (additions)

```python
BLOSUM62: dict[tuple[str, str], int]          # 24x24 incl. X, *, gap-less
ATCHLEY: dict[str, np.ndarray]                # 20 aa -> 5 factors

def cdr3_similarity(a: str, b: str, *, mode: str = "auto") -> float:
    """Similarity in [0, 1]. Equal length -> 1 - Hamming/len (fast path);
    otherwise BLOSUM62 global alignment score normalized by self-scores.
    'auto': identical -> 1.0; len diff > 4 -> 0.0; equal len -> Hamming;
    else alignment."""

def cdr3_similarity_pairs(seqs_a, seqs_b, *, mode="auto") -> np.ndarray:
    """Vectorized pairwise for pre-filtered candidate pairs (1-D arrays of
    equal length). Used by bcrgraph; NOT an all-pairs API."""

def atchley_embedding(seqs: list[str]) -> np.ndarray:
    """(n, 21) deterministic embedding: per-sequence mean/sd/min/max of the
    5 Atchley factors over residues + normalized length."""

def vj_distance(v1, j1, v2, j2, *, family_level=False) -> float:
    """0.0 same V&J gene; 0.5 same families; 1.0 otherwise (family_level
    collapses the 0.5 tier)."""
```

`cdr3_distance_matrix` and `blend_distances` keep their signatures (v1
compat); `cdr3_distance_matrix` gains `mode="hamming"` default and accepts
`mode="blosum"`.

### bcrgraph.py (new)

```python
def bcr_similarity_graph(clone_info: pd.DataFrame, *,
                         cdr3_sim_threshold: float = 0.85,
                         same_vj: bool = True,
                         include_light: bool = False,
                         max_pairs_per_clone: int = 200
                         ) -> scipy.sparse.csr_matrix:
    """Sparse (n_clones, n_clones) adjacency. Candidate pairs only within
    (V-gene, J-gene) blocks; edge weight = cdr3_similarity. clone_info must
    have v_call, j_call, cdr3 (and optionally cdr3_light, v_call_light),
    indexed by clone id."""

def define_clones(adata_or_bcr, *, cdr3_sim_threshold=0.85, same_vj=True,
                  method="leiden", resolution=0.5, key="clone_id_seq"):
    """Sequence-similarity clone (re)definition: cluster the BCR similarity
    graph (connected components or Leiden). Returns per-cell Series."""
```

### integrate.py (new)

```python
def joint_embedding(adata, *, clone_key="clone_id", basis="X_pca",
                    min_clone_size=3, n_neighbors=20, lam: float = 0.5,
                    n_components: int = 20, bcr_graph=None,
                    random_state=0) -> pd.DataFrame:
    """Coupled-Laplacian eigenmap embedding of clones.
    W = (1-lam)*W_gex + lam*W_bcr (row-normalized each), symmetric
    normalized Laplacian, bottom n_components+1 eigenvectors (drop trivial).
    Returns DataFrame indexed by clone id: joint_0..joint_{m-1}, n_cells.
    Stores in adata.uns['threadfin']['joint']."""

def integration_diagnostics(adata, joint=None, *, basis="X_pca") -> dict:
    """Benisse testCor analogues: on graph edges, Spearman(latent dist,
    GEX dist) and Spearman(latent dist, BCR dist); plus per-community
    modality contribution (variance decomposition of community separation
    into GEX-component and BCR-component)."""
```

`clonotype_recluster(..., basis=...)` accepts `basis="joint"` (uses
`uns['threadfin']['joint']`, computing it if absent) and
`distances=<precomputed ndarray>`.

### clones.py (new)

```python
def clone_isotype_summary(adata, *, cluster_key="clone_cluster") -> DataFrame
def clone_shm_summary(adata, *, mut_col=None) -> DataFrame  # needs AIRR muts
def clone_fate_table(adata, *, clone_key="clone_id", time_key="timepoint",
                     state_key="state") -> DataFrame  # clone × time × state
def community_transition(adata, *, time_key, cluster_key="clone_cluster")
    # clones observed at >=2 timepoints: community membership shift matrix
```

## Non-goals (documented in README)

- No full clonal lineage trees (use dandelion/Immcantation; Threadfin
  consumes their clone ids).
- No deep-learning sequence encoder in core deps (torch stays out;
  Atchley embedding is the deterministic stand-in, and `joint_embedding`
  accepts any user-supplied `bcr_graph` / embeddings).
- No antigen-specificity prediction.

## Validation commitments (what "works" means)

1. All 12 v1 tests keep passing; new modules get unit tests (target the
   same 80%+ coverage).
2. On all 5 public datasets: joint-embedding communities must (a) remain
   significant against Null 1/Null 3, (b) pass held-out split-clone
   co-clustering above chance, (c) reproduce the headline enrichment under
   both PCA and UMAP bases.
3. Biology readouts to demonstrate (report negative results honestly):
   - LN vaccine (GSE195673): GC communities with class-switch (IGHG/IGHA)
     gradients and SHM increase along clonal pseudotime.
   - Flu vaccine: d7 plasmablast-expanded clones concentrated in ASC
     communities; compare young vs older donors.
   - Tonsil: communities recover author GC/memory/plasma subsets.
   - EBV organoid: GFP+ timepoints form convergent activated communities.
   - Stephenson COVID: expanded clones in plasmablast compartment.
4. Benisse-style diagnostics reported per dataset; if the BCR modality adds
   nothing on a dataset, say so.
