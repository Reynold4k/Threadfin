"""Smoke tests: the figures that take result objects draw from real results of the analysis functions.

``pl.association``, ``pl.memory``, ``pl.label_effects`` and ``pl.heritability`` take the tables and dicts
returned by ``tl.association_test``, ``tl.clonal_memory``, ``tl.profile_association`` and
``tl.gene_heritability``. Each test draws one figure, checks what it shows against the result and saves it.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import threadfin as tf


@pytest.fixture(scope="module")
def results():
    """Results of the step-by-step analysis of one simulated two-time-point repertoire."""
    adata = tf.sim.simulate_repertoire(n_clones=1500, clone_size_exponent=1.7, n_timepoints=2, random_state=5)
    tf.tl.clone_profiles(adata, basis="X_pca", context_key="context", donor_key="donor", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf.tl.find_programmes(adata, resolution=1.0, n_boot=0, test_splits=False, verbose=False)
        adata.obs["share_S0"] = (adata.obs["true_state"] == "S0").astype(float)
        out = {
            "categorical": tf.tl.association_test(adata, "programme_at_start", n_perm=50, verbose=False),
            "numeric": tf.tl.association_test(adata, "share_S0", n_perm=50, verbose=False),
            "labels": {key: tf.tl.profile_association(adata, key, n_perm=50, verbose=False)
                       for key in ("programme_at_start", "share_S0")},
            "memory": tf.tl.clonal_memory(adata, "timepoint", n_null=50, n_boot=50, verbose=False),
        }
    return out


def _saved(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def test_association_forest_plot_draws_odds_ratios_and_rank_effects(results, tmp_path):
    """Guards pl.association: draws real association_test tables, for a categorical and a numeric label."""
    tab = results["categorical"]
    level = str(tab["level"].iloc[0])
    fig = tf.pl.association(tab, level=level)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and level in ax.get_xlabel()
    assert {t.get_text() for t in ax.get_yticklabels()} == set(tab["programme"].astype(str))
    assert _saved(fig, tmp_path / "association.png")
    fig = tf.pl.association(results["numeric"])
    assert "rank effect" in fig.axes[0].get_xlabel()
    assert len(fig.axes[0].get_yticklabels()) == len(results["numeric"])
    assert _saved(fig, tmp_path / "association_numeric.png")


def test_memory_plot_draws_the_index_of_the_result(results, tmp_path):
    """Guards pl.memory: draws a real clonal_memory result, with its memory index in the title."""
    mem = results["memory"]
    fig = tf.pl.memory(mem)
    ax = fig.axes[0]
    assert len(ax.patches) == 2
    assert f"{mem['memory_index']:.2f}" in ax.get_title()
    np.testing.assert_allclose(sorted(p.get_height() for p in ax.patches),
                               sorted([mem["mean_change"], mem["mean_change_random"]]))
    assert _saved(fig, tmp_path / "memory.png")


def test_label_effects_plot_draws_observed_and_shuffled_shares(results, tmp_path):
    """Guards pl.label_effects: draws real profile_association results, observed and shuffled shares."""
    labels = results["labels"]
    fig = tf.pl.label_effects(labels)
    ax = fig.axes[0]
    assert len(ax.patches) == 2 * len(labels)
    assert {t.get_text() for t in ax.get_yticklabels()} == set(labels)
    widths = sorted(p.get_width() for p in ax.patches)
    expected = sorted([100 * r["r2"] for r in labels.values()] + [100 * r["null_mean"] for r in labels.values()])
    np.testing.assert_allclose(widths, expected)
    assert _saved(fig, tmp_path / "label_effects.png")
    with pytest.raises(ValueError, match="No label results"):
        tf.pl.label_effects({})


def test_heritability_plot_ranks_genes_and_highlights_a_set(tmp_path):
    """Guards pl.heritability: draws a real gene_heritability table with a highlighted gene set."""
    rng = np.random.default_rng(1)
    clone = np.repeat(np.arange(120), 5)
    clonal = rng.normal(0, 1, (120, 3))[clone] + rng.normal(0, 0.5, (clone.size, 3))
    x = np.column_stack([clonal, rng.normal(0, 1, (clone.size, 9))]) + 5
    genes = ["clonal1", "clonal2", "clonal3"] + [f"noise{i}" for i in range(9)]
    adata = AnnData(x, obs=pd.DataFrame({"clone_id": [f"c{c}" for c in clone], "sample": "s1"},
                                        index=[f"cell{i}" for i in range(clone.size)]),
                    var=pd.DataFrame(index=genes))
    table = tf.tl.gene_heritability(adata, clone_key="clone_id", context_key="sample", layer=None, n_perm=20,
                                    verbose=False)
    fig = tf.pl.heritability(table, highlight={"clonal genes": ["clonal1", "clonal2", "clonal3"]}, n_label=3)
    ax = fig.axes[0]
    assert ax.collections[0].get_offsets().shape[0] == len(table)
    assert ax.collections[1].get_offsets().shape[0] == 3
    assert ax.get_legend() is not None and ax.get_xscale() == "log"
    assert {t.get_text() for t in ax.texts} == set(table.sort_values("icc", ascending=False).index[:3])
    assert _saved(fig, tmp_path / "heritability.png")
