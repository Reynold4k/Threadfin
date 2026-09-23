"""Render per-dataset discovery figures from the committed CSV tables.

Lightweight (no AnnData reloading): reads results/<dataset>/*.csv and writes
results/<dataset>/figures/discovery_<dataset>.png for the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def _sig_heatmap(ax, ds: str, title="Gene programmes per community"):
    sc_df = pd.read_csv(RES / ds / "community_scores.csv")
    pivot = sc_df.pivot_table(index="signature", columns="community", values="score")
    z = pivot.sub(pivot.mean(axis=1), axis=0)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(z.values)), vmax=np.nanmax(np.abs(z.values)))
    ax.set_xticks(range(len(z.columns)), [str(c) for c in z.columns])
    ax.set_yticks(range(len(z.index)), z.index)
    ax.set_xlabel("clone community")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)


def _enr_bars(ax, enr: pd.DataFrame, label_col: str, label: str, ylab: str):
    e = enr[enr[label_col] == label].copy()
    e["community"] = e[e.columns[0]].astype(str)
    e = e.sort_values("odds_ratio", ascending=False)
    colors = ["#b2182b" if f < 0.05 else "#999999" for f in e["fdr"]]
    ax.bar(e["community"], np.log10(e["odds_ratio"].clip(lower=1e-3)), color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(ylab)
    ax.set_xlabel("clone community")


def figure_flu():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    expa = pd.read_csv(RES / "flu_gse175522/expansion_index_age_tp.csv", index_col=0)
    ax = axes[0]
    names = list(expa.index)
    vals = expa.iloc[:, 0].values
    order = np.array(sorted(range(len(names)),
                            key=lambda i: ("young" in names[i], "d7" in names[i])))
    colors = np.array(["#2166ac" if "d0" in n else "#b2182b" for n in names])
    ax.bar(range(len(names)), np.asarray(vals)[order], color=colors[order])
    ax.set_xticks(range(len(names)), [names[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("clonal expansion index")
    ax.set_title("Expansion rises after vaccination\n(blue d0, red d7)")
    ax = axes[1]
    mig = pd.read_csv(RES / "flu_gse175522/migration_timepoint.csv", index_col=0)
    im = ax.imshow(mig.values, cmap="viridis")
    ax.set_xticks(range(len(mig.columns)), mig.columns)
    ax.set_yticks(range(len(mig.index)), mig.index)
    for i in range(mig.shape[0]):
        for j in range(mig.shape[1]):
            ax.text(j, i, f"{mig.values[i, j]:.1f}", ha="center", va="center",
                    color="w" if mig.values[i, j] < mig.values.max() / 2 else "k")
    ax.set_title("clone migration d0 ↔ d7")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _sig_heatmap(axes[2], "flu_gse175522")
    fig.tight_layout()
    _save(fig, "flu_gse175522", "discovery_flu.png")


def figure_ebv():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    enr = pd.read_csv(RES / "ebv_organoid_gse317492/gfp_enrichment.csv")
    _enr_bars(axes[0], enr, "condition", "GFP+",
              "log10 OR (GFP+ experimentally infected)")
    axes[0].set_title("Communities vs experimental infection\n(red: FDR<0.05)")
    _sig_heatmap(axes[1], "ebv_organoid_gse317492")
    fig.tight_layout()
    _save(fig, "ebv_organoid_gse317492", "discovery_ebv.png")


def figure_tonsil():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sc_df = pd.read_csv(RES / "tonsil_king2021/community_scores.csv")
    pb = sc_df[sc_df["signature"] == "plasmablast"].sort_values("community")
    ax = axes[0]
    ax.bar(pb["community"].astype(str), pb["score"], color="#b2182b")
    ax.set_ylabel("plasmablast programme score")
    ax.set_xlabel("clone community")
    ax.set_title("Plasmablast programme lights up\nexactly one lineage community")
    _sig_heatmap(axes[1], "tonsil_king2021")
    fig.tight_layout()
    _save(fig, "tonsil_king2021", "discovery_tonsil.png")


def figure_stephenson():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sev = pd.read_csv(RES / "stephenson2021/severity_enrichment.csv")
    pivot = sev.pivot_table(index="severity", columns="clone_cluster",
                            values="odds_ratio")
    order = ["Asymptomatic", "Mild", "Moderate", "Severe", "Critical"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    ax = axes[0]
    im = ax.imshow(np.log10(pivot.values), cmap="RdBu_r", aspect="auto",
                   vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(pivot.columns)), [f"community {c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            fdr = sev[(sev["severity"] == pivot.index[i]) &
                      (sev["clone_cluster"].astype(str) == str(pivot.columns[j]))]["fdr"]
            star = "*" if len(fdr) and fdr.iloc[0] < 0.05 else ""
            ax.text(j, i, f"{pivot.values[i, j]:.1f}{star}", ha="center", va="center")
    ax.set_title("community × COVID severity\nlog10 OR (* FDR<0.05)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _sig_heatmap(axes[1], "stephenson2021")
    fig.tight_layout()
    _save(fig, "stephenson2021", "discovery_stephenson.png")


def _save(fig, ds: str, name: str):
    out = RES / ds / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}", flush=True)


if __name__ == "__main__":
    figure_flu()
    figure_ebv()
    figure_tonsil()
    figure_stephenson()
