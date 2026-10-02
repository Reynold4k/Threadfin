"""Step-by-step analysis functions.

:func:`threadfin.run` chains these in order; call them yourself when you
need finer control:

1. :func:`define_clones` - BCR sequences -> clones (within donors)
2. :func:`clone_profiles` - one shrunken state profile per clone
3. :func:`clonal_coherence` - does clone identity explain cell state?
4. :func:`find_programmes` - group clones with similar profiles, with stability
5. :func:`association_test` / :func:`programme_markers` - clone-level tests
   of programmes vs labels and genes
6. :func:`clonal_memory` - do clones keep their programme over time/tissue?
7. :func:`gene_heritability` - which genes are clonally inherited?
"""

from .clones import define_clones, find_threshold, mutation_frequency
from .dynamics import clonal_memory
from .heritability import gene_heritability, geneset_heritability
from .profiles import clone_profiles
from .programmes import find_programmes, programme_composition, programme_markers
from .stats import association_test, clonal_coherence, clone_labels

__all__ = [
    "association_test",
    "clonal_coherence",
    "clonal_memory",
    "clone_labels",
    "clone_profiles",
    "define_clones",
    "find_programmes",
    "find_threshold",
    "gene_heritability",
    "geneset_heritability",
    "mutation_frequency",
    "programme_composition",
    "programme_markers",
]
