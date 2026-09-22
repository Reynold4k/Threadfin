import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _random_cdr3(rng, length=14):
    return "".join(rng.choice(list(_AA), size=length))


@pytest.fixture()
def synthetic():
    """Synthetic paired GEX+BCR AnnData.

    120 clones in 3 transcriptional home states (85 % of a clone's cells live
    in its home state), 10 singleton clones, 50 cells without BCR.
    """
    rng = np.random.default_rng(42)
    n_states, d = 3, 8
    centers = rng.normal(0, 6.0, size=(n_states, d))

    n_clones = 120
    clone_home = rng.integers(0, n_states, size=n_clones)

    obs_rows, emb_rows, bcr_rows = [], [], []
    v_pool = [f"IGHV{i}-{j}*01" for i in (1, 3, 4, 5) for j in range(1, 12)]
    d_pool = [f"IGHD{i}-{j}*01" for i in (1, 2, 3) for j in range(1, 8)]
    j_pool = [f"IGHJ{i}*01" for i in range(1, 7)]

    cell_i = 0
    for c in range(n_clones):
        home = clone_home[c]
        n_cells = int(rng.integers(5, 25))
        v = v_pool[(c * 7 + home) % len(v_pool)]
        d_ = d_pool[c % len(d_pool)]
        j = j_pool[(c * 3 + home) % len(j_pool)]
        cdr3 = _random_cdr3(rng)
        for _ in range(n_cells):
            state = home if rng.random() < 0.85 else int(rng.integers(0, n_states))
            barcode = f"cell{cell_i:05d}"
            emb_rows.append(centers[state] + rng.normal(0, 1.0, size=d))
            obs_rows.append({"state": f"state{state}"})
            bcr_rows.append(
                {"barcode": barcode, "clone_id": f"CLONE{c:03d}", "v_call": v,
                 "d_call": d_, "j_call": j, "cdr3": cdr3}
            )
            cell_i += 1

    # singleton clones (filtered out by min_clone_size)
    for s in range(10):
        barcode = f"cell{cell_i:05d}"
        state = int(rng.integers(0, n_states))
        emb_rows.append(centers[state] + rng.normal(0, 1.0, size=d))
        obs_rows.append({"state": f"state{state}"})
        bcr_rows.append({"barcode": barcode, "clone_id": f"SINGLE{s}",
                         "v_call": v_pool[s], "d_call": d_pool[s % len(d_pool)],
                         "j_call": j_pool[s % len(j_pool)],
                         "cdr3": _random_cdr3(rng)})
        cell_i += 1

    # cells without BCR
    for _ in range(50):
        barcode = f"cell{cell_i:05d}"
        state = int(rng.integers(0, n_states))
        emb_rows.append(centers[state] + rng.normal(0, 1.0, size=d))
        obs_rows.append({"state": f"state{state}"})
        cell_i += 1

    n_cells = cell_i
    n_genes = 100
    x = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame(obs_rows, index=[f"cell{i:05d}" for i in range(n_cells)])
    obs.index.name = "barcode"
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = AnnData(X=x, obs=obs, var=var)

    emb = np.array(emb_rows)
    adata.obsm["X_umap"] = emb[:, :2].copy()
    adata.obsm["X_pca"] = emb

    bcr = pd.DataFrame(bcr_rows).set_index("barcode")
    truth = {f"CLONE{c:03d}": f"state{clone_home[c]}" for c in range(n_clones)}
    return adata, bcr, truth
