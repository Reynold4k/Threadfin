#!/usr/bin/env python
"""Draw docs/_static/tutorial_clone_map.png, the figure of Tutorial 3, section 3, with the tutorial's own code.

The automatic clone map of section 1 and the map set by hand in section 3, of the same 931 simulated clones,
coloured by the simulated programme.

Usage: python docs/tutorials/make_tutorial_clone_map.py [output.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import threadfin as tf  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "_static" / "tutorial_clone_map.png"

adata = tf.sim.simulate_repertoire(random_state=0)                      # section 1
tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", representation="kernel",
                     verbose=False)
tf.tl.embed_clones(adata, verbose=False)
prof = adata.uns["threadfin"]["profiles"]
auto = (int(prof["map_params"]["n_neighbors"]), float(prof["map_params"]["min_dist"]))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
for ax, setting, name in ((axes[0], auto, "automatic"), (axes[1], (10, 0.05), "set by hand")):
    tf.tl.embed_clones(adata, n_neighbors=setting[0], min_dist=setting[1], verbose=False)   # section 3
    prof["clone_table"]["simulated programme"] = tf.tl.clone_labels(adata, "true_programme")
    tf.pl.clone_map(adata, color="simulated programme", facet=False, ax=ax,
                    title=f"{name}\nn_neighbors {setting[0]}, min_dist {setting[1]:g}")
    connected = float(prof["map_params"]["quality"]["coherence"])
    ax.text(0.02, 0.02, f"{100 * connected:.1f}% of clones in one piece", transform=ax.transAxes, fontsize=7)
axes[0].get_legend().remove()
axes[1].get_legend().set_title("simulated\nprogramme")
axes[1].get_legend().set_frame_on(False)
fig.tight_layout()
fig.savefig(out, dpi=200)
print(f"wrote {out}: automatic n_neighbors {auto[0]}, min_dist {auto[1]:g}")
