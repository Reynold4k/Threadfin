"""One place that answers "where are this dataset's clone-map coordinates?".

Two things used to be confused here, and keeping them apart is the whole point of this module.

**The clone definition** — which clones exist — changed on 2026-10-11 when clone calling moved to
Dandelion's paired heavy + light definition. Every clone map was rebuilt from the new clones.

**The layout parameters** — where the points are drawn — did *not* change, and are not invalidated by
the clone definition. Three datasets had their UMAP settings chosen by a documented sweep, and those
settings are now carried by the pipeline itself (``map_params`` in ``run_case_study.CONFIGS``, passed
to ``find_programmes`` as ``map_n_neighbors`` / ``map_min_dist``), so ``clone_table.csv`` already holds
the tuned layout:

================  ==============================  ====================================================
dataset           layout                          chosen by
================  ==============================  ====================================================
malaria_late      n_neighbors 50, min_dist 0.5    85 settings swept (malaria_late_embedding_audit);
                                                  most continuous of the most faithful, Spearman
                                                  distance correlation 0.856 against 0.764 default
ln_vaccine        n_neighbors 50, min_dist 0.3    48 settings swept (benchmark/results/b3_map_params);
                                                  connected, equal Moran's I for mutation and for day
mouse_rbd         n_neighbors 15, min_dist 0.1    the sweep was a robustness audit and confirmed the
                                                  default (mouse_rbd_embedding_audit/verify.json)
everything else   n_neighbors 15, min_dist 0.1    package default
================  ==============================  ====================================================

GSE246382 has no entry because it has no clone map: 884 cells over 11 mice is about 80 cells per
animal, so the same clone is almost never captured twice. Its clonal ICC is 0.05 (p = 0.23) under
either clone definition, and a map of its multi-cell clones carries no measured-compartment structure
(15-NN gate agreement 0.556 against a null of 0.567, p = 0.80). The historical 377-point panel grouped
cells by V-D-J gene usage with singletons included, and the audit's own metadata records that 83 of
those 377 groups hold more than one junction — so its points were not clones.

Resolve coordinates through :func:`coords` and quote :func:`provenance` in the legend, rather than
reading ``clone_table.csv`` directly: the provenance line is what tells a reader whether a panel is on
a tuned layout or on the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/data/scratch/projects/punim1236/AAA_Chen_is_here/Threadfin发表")
CASE = ROOT / "Threadfin" / "case_studies" / "results"

DEFAULT_MAP = {"map_n_neighbors": 50, "map_min_dist": 0.8}


def _map_params(dataset: str) -> dict:
    """The layout settings `embed_clones` actually used, as the run recorded them.

    Read `summary["clone_map"]`, which `embed_clones` writes. An older key, `summary["programmes"]
    ["map"]`, is left behind by `find_programmes` and does **not** describe the map any more -- it
    reported n_neighbors 15 / min_dist 0.1 for every dataset while the maps had been drawn at 50/0.8,
    200/0.1 and 15/0.8, so a legend built from it was wrong on all three.
    """
    try:
        d = json.loads((CASE / dataset / "summary.json").read_text()).get("clone_map") or {}
    except FileNotFoundError:
        return dict(DEFAULT_MAP)
    if "n_neighbors" not in d:
        return dict(DEFAULT_MAP)
    return {"map_n_neighbors": d["n_neighbors"], "map_min_dist": d["min_dist"]}


def coords(dataset: str) -> pd.DataFrame:
    """Clone-map coordinates for a dataset, indexed by clone id.

    ``attrs['provenance']`` names the layout the run used, tuned or default.
    """
    t = pd.read_csv(CASE / dataset / "clone_table.csv")
    if "x" not in t.columns:
        raise FileNotFoundError(
            f"{dataset} has no clone-map coordinates. find_programmes only embeds clones that reach "
            "reliability 0.5, so a dataset with too few repeatedly captured clones has no map.")
    xy = t.dropna(subset=["x", "y"]).copy()
    xy.index = xy["clone_id"].astype(str)
    p = _map_params(dataset)
    tuned = (p["map_n_neighbors"], p["map_min_dist"]) != (DEFAULT_MAP["map_n_neighbors"],
                                                          DEFAULT_MAP["map_min_dist"])
    xy.attrs["provenance"] = (
        f"{'swept' if tuned else 'package default'}: n_neighbors {p['map_n_neighbors']}, "
        f"min_dist {p['map_min_dist']}")
    return xy[["x", "y"] + [c for c in xy.columns if c not in ("x", "y", "clone_id")]]


def provenance(dataset: str) -> str:
    """One line naming the layout a dataset's map was drawn with, for the figure legend."""
    return coords(dataset).attrs["provenance"]
