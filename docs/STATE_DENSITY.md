# Optional reference-based state distributions

This API adds a sampling-aware description of clone composition. Existing
mean/kernel profiles and the default run pipeline are unchanged. It is an
opt-in analysis, not a replacement selected because it won every benchmark.

## Fit once, project new cells

All RNA preprocessing must be fitted on the reference if the query is meant
to be held out. The helper below selects receptor-excluded Seurat HVGs, fits
scaling and PCA on the reference, and freezes them. It does not integrate a
new donor into a jointly refitted embedding.

~~~python
import threadfin as tf

embedding = tf.pp.FrozenExpressionModel.fit(
    reference_adata, n_top_genes=2000, n_comps=30,
)
reference_x = embedding.transform(reference_adata)
query_x = embedding.transform(query_adata)

model = tf.StateDensityModel.fit(
    reference_x,
    reference_adata.obs["clone_id"],
    n_states=16,
    prior_strength=2.0,            # fixed; or "auto", calibrated on reference cells
)
result = model.transform(query_x, query_adata.obs["clone_id"])

result.proportions                 # shrunken region proportions
result.raw_proportions             # observed region proportions
result.clone_table                 # n_cells, data_weight, background provenance
result.sampling_lower              # conservative simultaneous sampling intervals
result.sampling_upper
result.lower                       # model-conditional marginal credible intervals
result.upper
result.contrast("clone_A", "clone_B")
~~~

Pass only cells with known, donor-private biological clone identifiers to
model fitting. The AnnData wrapper tf.tl.clone_densities omits cells with
missing clone IDs; it requires an explicitly fitted reference_model. Its
results are stored separately under uns['threadfin']['densities'].

Query gene order may differ, but FrozenExpressionModel requires the same
gene universe for consistent total-count normalisation. Counts must be
non-negative, finite, and have positive library sizes. Save/load either
model using its save/load methods; NPZ files contain numeric arrays and
metadata, with pickle disabled.

## What is estimated

Clone-balanced KMeans establishes fixed Voronoi regions in the reference
embedding. A region is a computational partition, not an independently
established cell type. For clone c, its integer region counts are
n_c1, ..., n_cK. Given background proportions q and prior strength alpha:

- raw proportion: n_ck / n_c;
- posterior parameters: n_ck + alpha q_ck;
- posterior mean: (n_ck + alpha q_ck) / (n_c + alpha);
- data_weight: n_c / (n_c + alpha).

The last quantity reports how much weight comes from observed counts. It is
not a calibrated reliability probability and is not the legacy ICC-based
profile reliability.

Each biological reference clone has equal weight in the background. When
contexts are supplied, the background uses other reference clones in that
context. A query clone's entire contribution is excluded. If no independent
clone remains in the context, the model uses the global reference excluding
that clone and reports global_background_fraction. It never silently
estimates a new background from query expression. Multiple query contexts
are combined according to their captured cell counts.

Leaving a clone out of the background does not remove its historical
influence on the already fitted region centres. Use disjoint reference
clones for a fully independent evaluation. Automatic alpha selection uses
split-cell predictive log scores from reference clones only, a fixed
candidate grid, and equal clone weighting. Outcome labels are not used.

## Two different kinds of interval

| Output | Meaning | Limit |
| --- | --- | --- |
| lower / upper | Marginal Dirichlet posterior credible intervals | Conditional on the fixed partition, background and alpha; can severely under-cover when the prior is wrong |
| sampling_lower / sampling_upper | Exact binomial intervals with Bonferroni correction across the K regions of each clone | Conservative coverage under independent multinomial sampling; not simultaneous across all clones |
| contrast | Posterior region differences and direction probabilities | Marginal, conditional, no multiplicity correction; direction probability is not a p-value |

Neither interval accounts for dependence between captured cells, shared
culture/donor effects, sorting bias, uncertain clone definitions, or the
uncertainty of the fitted reference geometry. Coverage in simulated
independent sampling is not coverage under every experimental design.

## Locate changes and describe regions

~~~python
# Numeric log-expression, module scores, or one-hot cell annotations, rows aligned to reference_x.
annotation = model.annotate_regions(
    reference_x, reference_gene_or_module_scores,
    reference_adata.obs["clone_id"],
)
annotation["values"]       # first average within clone/region, then equally across clones
annotation["coverage"]     # number of cells and clones supporting each region

difference = result.contrast("clone_A", "clone_B")
~~~

A region's marker/module description helps interpret a shift. It reuses
expression and is descriptive, rather than an independent differential
expression test. A changed region weight does not establish a transition
rate, antigen affinity, mutation rate or future fate.

model.log_density(points, one_clone_proportions) reconstructs a normalised
Gaussian mixture centred at the fixed landmarks. The default scales
0.5, 1 and 2 share equal weight; each component integrates to one. This is a
smooth display of estimated region mass, not a fitted cell-level KDE or
Clonotrace's graph-density/transport model. The intervals concern region
proportions, not pointwise coverage of this continuous display.

## Scaling and validation

transform_batches accepts an iterator of (embedding, clone_ids, contexts)
batches; one clone can span several batches. Auxiliary accumulation uses
O(batch_size * K + clone_context_count * K) memory, plus the retained
reference. No cell-by-cell distance matrix is retained. Reference KMeans
fitting still uses the supplied reference matrix; it is not streaming fit.

The frozen-reference approach keeps graph smoothing, RFF, shrinkage and
clone PCA separate: the new density model uses neither graph diffusion nor
RFF/PCA compression of clone features. It is not a complete ablation study
of every legacy component.

Reproducible validations are in case_studies/validate_density_model.py,
validate_density_reporter.py and benchmark_representation_runtime.py;
results are in case_studies/results/algorithm_revision/.

- Simulations include rare states, strong heterogeneity and pure clones.
  Fixed-prior shrinkage helps some regimes and hurts others. Posterior
  intervals can be misleading near pure-state boundaries; the separate
  sampling intervals retain conservative coverage under the tested design.
- LARRY split-cell estimation separates reference and evaluation biological
  barcodes, but uses shared precomputed RNA coordinates. It tests estimation
  in a fixed space, not a wholly inductive RNA pipeline.
- NP and RBD reporter validation holds out entire mice, including all
  preprocessing. Simple RNA means remain strong and often better.
- Isolated-process timings include fitting. The density implementation used
  less peak memory than the legacy kernel on the 50,000-cell example, but
  took longer. No general speed or accuracy superiority is claimed.
