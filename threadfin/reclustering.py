"""Reproducible starting configurations for centroid-based clone reclustering.

These presets describe graph/UMAP settings, not biological classes. They do
not train on markers, select a preferred shape, or infer differentiation.
The v4 sampling-adjusted profile workflow remains a separate analysis.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ._utils import require_positive_int

_PRESETS = {
    "cohesive": {
        "description": "Compact, connected neighbourhoods; preserves legacy defaults.",
        "n_neighbors": 20, "resolution": 0.3, "min_dist": 0.1,
        "spread": 1.0, "learning_rate": 1.0,
    },
    "continuous": {
        "description": "More space within connected neighbourhoods; report-style starting point.",
        "n_neighbors": 20, "resolution": 0.1, "min_dist": 0.4,
        "spread": 1.0, "learning_rate": 1.0,
    },
    "discrete": {
        "description": "More local neighbourhoods and finer exploratory partitions.",
        "n_neighbors": 10, "resolution": 0.8, "min_dist": 0.05,
        "spread": 1.0, "learning_rate": 1.0,
    },
}


def reclustering_presets() -> dict:
    """Return independent copies of the three centroid-reclustering presets.

    Pass a preset name to :func:`threadfin.clonotype_recluster`. Explicit
    keyword values override that preset. A preset cannot establish whether
    real data contain continuous or discrete biological populations.
    """
    return deepcopy(_PRESETS)


def _resolve_parameters(preset, **overrides) -> dict:
    if not isinstance(preset, str) or preset not in _PRESETS:
        raise ValueError("preset must be 'cohesive', 'continuous', or 'discrete'.")
    settings = {k: v for k, v in _PRESETS[preset].items() if k != "description"}
    settings.update({k: v for k, v in overrides.items() if v is not None})
    require_positive_int(settings["n_neighbors"], "n_neighbors")
    if settings["n_neighbors"] < 2:
        raise ValueError("n_neighbors must be at least 2 for clone UMAP.")
    for name in ("resolution", "min_dist", "spread", "learning_rate"):
        value = settings[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite number.")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite number.") from None
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        settings[name] = value
    if settings["resolution"] <= 0 or settings["spread"] <= 0 or settings["learning_rate"] <= 0:
        raise ValueError("resolution, spread, and learning_rate must be greater than zero.")
    if not 0 <= settings["min_dist"] <= settings["spread"]:
        raise ValueError("min_dist must be between zero and spread.")
    return settings
