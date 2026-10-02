"""Threadfin: clone-level analysis of B-cell states from paired scRNA-seq + BCR-seq.

Quick start
-----------
>>> import threadfin as tf
>>> result = tf.run(adata, bcr="filtered_contig_annotations.csv",
...                 donor_key="donor", sample_key="sample")
>>> print(result.summary())

Namespaces
----------
``tf.run`` / ``tf.read_bcr``   one-call analysis and BCR reader
``tf.tl``                      the individual analysis steps
``tf.pp``                      preprocessing (embedding without IG genes)
``tf.pl``                      figures
``tf.sim``                     simulated data with known ground truth

The v3 functions (``clonotype_recluster``, ``joint_embedding``, ...) remain
importable for backward compatibility.
"""

from . import plotting, pp, simulate, tl
from .api import ThreadfinResult, read_bcr, run
from .clones import define_clones
from .io import attach_bcr, build_clone_key, read_10x_vdj, read_airr

pl = plotting
sim = simulate

# v3 API, kept for backward compatibility
from . import clones, metrics, migration, programs, sequence, specificity  # noqa: E402
from .bcrgraph import bcr_similarity_graph  # noqa: E402
from .core import bcr_reclustering, clonal_pseudotime, clone_centroids, clonotype_recluster  # noqa: E402
from .integrate import integration_diagnostics, joint_embedding  # noqa: E402
from .migration import clone_distribution, expansion_index, migration_index, transition_index  # noqa: E402
from .programs import community_markers, community_score  # noqa: E402
from .specificity import annotate_specificity, specificity_enrichment  # noqa: E402

__version__ = "4.0.0"
__all__ = [
    "ThreadfinResult",
    "attach_bcr",
    "build_clone_key",
    "define_clones",
    "pl",
    "pp",
    "read_10x_vdj",
    "read_airr",
    "read_bcr",
    "run",
    "sim",
    "tl",
    # v3 (legacy)
    "annotate_specificity",
    "bcr_reclustering",
    "bcr_similarity_graph",
    "clonal_pseudotime",
    "clone_centroids",
    "clone_distribution",
    "clones",
    "clonotype_recluster",
    "community_markers",
    "community_score",
    "expansion_index",
    "integration_diagnostics",
    "joint_embedding",
    "metrics",
    "migration",
    "migration_index",
    "plotting",
    "programs",
    "sequence",
    "specificity",
    "specificity_enrichment",
    "transition_index",
]
