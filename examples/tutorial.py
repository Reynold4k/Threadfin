"""Threadfin tutorial: from a paired dataset to clone-level answers.

Run it as a script, or paste the sections into a notebook:

    python examples/tutorial.py

It uses simulated data with a known answer, so it needs no download and
finishes in about a minute. Every step is the same on real data; the only
difference is that you pass your own AnnData and BCR table to ``tf.run``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import threadfin as tf  # noqa: E402

# ---------------------------------------------------------------- 1. the data
# A simulated repertoire: four groups of clones, sampled in batches that differ
# technically, with clone sizes as uneven as in a real experiment.
adata = tf.sim.simulate_repertoire(random_state=0)
print(adata)
print("\ncells per clone (most clones are singletons):")
print(adata.obs["clone_id"].value_counts().describe()[["count", "mean", "max"]])

# ---------------------------------------------------------------- 2. one call
# donor_key: who each cell came from - clones are never formed across donors.
# sample_key: which library/sort/time point - each clone is compared with the
#             cells it was sampled beside, and the null shuffles within it.
# basis:      an existing embedding; omit it on real data and Threadfin builds
#             one that excludes immunoglobulin genes.
result = tf.run(adata, donor_key="donor", sample_key="context", basis="X_pca",
                state_key="true_state", test=["true_programme"])
print(result.summary())

# ---------------------------------------------------------------- 3. the parts
# Everything above is also available step by step, which is what you want when
# you need to change the sampling context or the number of permutations.
tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor",
                     representation="kernel")
coherence = tf.tl.clonal_coherence(adata, strata_key="context", n_perm=500)
print(f"\nclone identity explains {100 * coherence['icc']:.1f}% of state, "
      f"{100 * coherence['null_mean']:.1f}% when shuffled (p = {coherence['p_value']:.3g})")

# Groups of clones are reported only when the split between them is real.
programmes = tf.tl.find_programmes(adata)
print(programmes)

# A label is first tested against the clone profiles - this works whether or not
# distinct groups exist - and then, if there are groups, between them.
effect = tf.tl.profile_association(adata, "true_programme")
print(f"\nthe label explains {100 * effect['r2']:.1f}% of how clones differ "
      f"(p = {effect['p_value']:.3g}, {effect['n_clones']} clones)")

# ---------------------------------------------------------------- 4. the table
# One row per clone: its size, how reliable its profile is given that size, its
# group, and any label you collapsed onto it.
print("\nper-clone table:")
print(result.clones.head()[["n_cells", "reliability", "clone_programme", "confidence"]])

# ---------------------------------------------------------------- 5. figures
result.plot("tutorial_overview.png")
print("\nwrote tutorial_overview.png")

# ---------------------------------------------------------------- what next
# On your own data:
#   result = tf.run(adata, bcr="filtered_contig_annotations.csv",
#                   donor_key="donor", sample_key="sample",
#                   test=["isotype", "antigen_binding"], time_key="timepoint")
# See report/ for worked examples on eight published datasets, and
# docs/METHODS.md for what each step does.
