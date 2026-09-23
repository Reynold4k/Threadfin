"""Threadfin: transcriptional-state-aware reclustering of B-cell clonotypes.

Threadfin integrates paired scRNA-seq and scBCR-seq data by placing every
clonotype at the centroid of its member cells in a transcriptional embedding
and clustering clonotypes in that space — grouping clones that share a
transcriptional state rather than a sequence.

Quick start
-----------
>>> import threadfin as tf
>>> bcr = tf.read_10x_vdj("filtered_contig_annotations.csv")
>>> bcr = tf.build_clone_key(bcr)          # v_call_d_call_j_call
>>> adata = tf.attach_bcr(adata, bcr)      # adds clone_id to adata.obs
>>> adata = tf.clonotype_recluster(adata)  # adds obs['clone_cluster']
>>> tf.plotting.clone_map(adata, save="clone_map.png")
"""

from .core import (
    bcr_reclustering,
    clonal_pseudotime,
    clone_centroids,
    clonotype_recluster,
)
from .integrate import integration_diagnostics, joint_embedding
from .bcrgraph import bcr_similarity_graph, define_clones
from . import clones
from .io import attach_bcr, build_clone_key, read_10x_vdj, read_airr
from . import metrics, plotting, sequence

__version__ = "2.0.0"
__all__ = [
    "attach_bcr",
    "bcr_reclustering",
    "bcr_similarity_graph",
    "build_clone_key",
    "clonal_pseudotime",
    "clone_centroids",
    "clones",
    "clonotype_recluster",
    "define_clones",
    "integration_diagnostics",
    "joint_embedding",
    "metrics",
    "plotting",
    "read_10x_vdj",
    "read_airr",
    "sequence",
]
