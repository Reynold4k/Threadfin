"""Step-by-step analysis functions.

:func:`threadfin.run` chains these in order; call them yourself when you
need finer control:

1. :func:`define_clones_dandelion` - BCR contigs -> clones (within donors); Dandelion's
   paired heavy + light definition is Threadfin's clone definition
   (:func:`define_clones`, the old heavy-chain-only caller, is deprecated)
2. :func:`clone_profiles` - one shrunken state profile per clone
3. :func:`clonal_coherence` - are cells of one clone more alike than cells of
   different clones?
4. :func:`find_programmes` - group clones with similar profiles (significant
   splits only), with bootstrap stability; :func:`embed_clones` - the clone map,
   whose UMAP settings :func:`tune_clone_map` chooses automatically (a ladder
   search over layouts), and
   :func:`map_quality` - how coherent and faithful a clone map is
5. :func:`association_test` / :func:`programme_markers` - clone-level tests
   of programmes vs labels and genes; :func:`profile_association` - does a
   label explain how clones differ (works without distinct programmes)
6. :func:`clonal_memory` - do clones keep their programme over time/tissue?
   :func:`export_lag` - how far behind its own germinal centre is a clone's output?
7. :func:`gene_heritability` - which genes are shared within clones (clonal ICC per gene)?
8. :func:`lineage_forest` + :func:`lineage_heritability` - inside a clone, do cells that are close
   relatives on the receptor lineage tree share transcriptional state?
"""

from .clones import define_clones, find_threshold, mutation_frequency
from .dandelion import define_clones_dandelion, read_contigs
from .density import clone_densities
from .dynamics import clonal_memory
from .export import export_lag
from .heritability import gene_heritability, geneset_heritability
from .maptune import layout_clone_map, map_quality, tune_clone_map
from .phylo import lineage_forest, lineage_heritability
from .profiles import clone_profiles
from .programmes import embed_clones, find_programmes, landscape_genes, programme_composition, programme_markers
from .stats import association_test, clonal_coherence, clone_labels, profile_association

__all__ = [
    "association_test",
    "clonal_coherence",
    "clonal_memory",
    "export_lag",
    "clone_labels",
    "clone_profiles",
    "clone_densities",
    "define_clones",
    "define_clones_dandelion",
    "embed_clones",
    "find_programmes",
    "find_threshold",
    "gene_heritability",
    "geneset_heritability",
    "lineage_forest",
    "layout_clone_map",
    "lineage_heritability",
    "map_quality",
    "mutation_frequency",
    "profile_association",
    "programme_composition",
    "programme_markers",
    "landscape_genes",
    "tune_clone_map",
]
