#!/usr/bin/env python3
"""Score native representations against measured, label-held-out GC gates.

Uses the fixed donor-private families in the published case-study tables.
Reporter fields enter only target construction and the downstream ridge readout.
Representations are unsupervised/transductive; this is not an inductive test.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'case_studies/results/native_benchmark'
TARGETS = {'mouse_np': [('division_gate', 'mCherry-low')],
           'mouse_rbd': [('division_gate', 'mCherry-low'), ('rbd_bait', 'RBD+'), ('zone_gate', 'DZ')]}
ALPHAS = [0.1, 1, 10, 100, 1000]


def common_input(dataset):
    folder = OUT / dataset
    pcs = pd.read_csv(folder / 'expression_pca.csv', index_col=0).T
    obs = pd.read_csv(ROOT / f'case_studies/results/{dataset}/cells.csv.gz', index_col=0).loc[pcs.index]
    assert obs.clone_id.notna().all(), 'Native common cells must have fixed family membership.'
    assert obs.groupby('clone_id', observed=True).donor.nunique().eq(1).all()
    return pcs, obs


def family_targets(obs, column, positive):
    """Unmeasured gates stay missing, rather than becoming negative labels."""
    measured = obs[column].notna()
    tab = obs.loc[measured, ['clone_id', 'donor']].copy()
    tab['positive'] = obs.loc[measured, column].eq(positive).astype(float)
    result = tab.groupby('clone_id', observed=True).agg(
        target=('positive', 'mean'), n_measured=('positive', 'size'), donor=('donor', 'first'))
    return result


def pool_cells(features, obs):
    """Exclude partially represented families; weight native nodes by cells."""
    assert features.index.is_unique, 'Native output contains duplicated cell IDs.'
    assert obs.index.is_unique
    assert not len(features.index.difference(obs.index)), 'Native output has unmapped cell IDs.'
    n_all = obs.groupby('clone_id', observed=True).size()
    represented = features.reindex(obs.index).dropna()
    tab = represented.assign(_family=obs.clone_id.reindex(represented.index))
    counts = tab.groupby('_family', observed=True).size()
    complete = counts.index[counts.eq(n_all.reindex(counts.index))]
    return tab.groupby('_family', observed=True).mean().reindex(complete).dropna()


def squared_distance_features(distance, n_components=30):
    """Classical MDS of Benisse's native squared latent-distance output.

    This is an output adapter. Benisse itself, its learned graph and Q are
    unchanged. Only the largest positive 30 eigencomponents are retained.
    """
    distance = np.asarray(distance, dtype=float)
    assert distance.ndim == 2 and distance.shape[0] == distance.shape[1]
    np.testing.assert_allclose(distance, distance.T, atol=1e-7)
    assert np.isfinite(distance).all()
    row_mean = distance.mean(axis=1)
    gram = -0.5 * (distance - row_mean[:, None] - row_mean[None, :] + row_mean.mean())
    k = min(n_components, len(distance) - 1)
    values, vectors = eigsh(gram, k=k, which='LA', v0=np.random.default_rng(0).normal(size=len(gram)))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = values > 1e-10
    features = vectors[:, keep] * np.sqrt(values[keep])
    return features, {'adapter': 'classical MDS of native squared latent distances',
                      'n_components': int(keep.sum()), 'positive_eigenvalues': values[keep].tolist(),
                      'fraction_positive_trace_retained': float(values[keep].sum() / max(np.trace(gram), 1e-12))}


def family_distance_kernel(distance, cell_nodes, obs):
    """Preserve the complete native geometry after cell-weighted family pooling."""
    assert cell_nodes.index.is_unique and obs.index.is_unique
    assert cell_nodes.index.equals(obs.index)
    sizes = obs.groupby('clone_id', observed=True).size()
    ids = sizes.index[sizes.ge(2)]
    family_codes = ids.get_indexer(obs.clone_id)
    keep = family_codes >= 0
    weights = csr_matrix((1 / sizes.reindex(obs.loc[keep, 'clone_id']).to_numpy(),
                          (family_codes[keep], cell_nodes.to_numpy()[keep])),
                         shape=(len(ids), len(distance)))
    # Double centring removes the weighted node norms. The result is exactly
    # the dot-product kernel of cell-weighted family means, with no truncation.
    pooled = np.asarray((weights @ distance) @ weights.T)
    means = pooled.mean(1)
    kernel = -.5 * (pooled - means[:,None] - means[None,:] + means.mean())
    return pd.DataFrame(kernel, index=ids, columns=ids)


def build_representations(dataset, stage='profiles'):
    import threadfin as tf
    folder = OUT / dataset
    destination = OUT / 'representations' / dataset
    destination.mkdir(parents=True, exist_ok=True)
    pcs, obs = common_input(dataset)
    # Do not put measured gate columns into either AnnData model input.
    bare = obs[['clone_id', 'donor']].copy()
    a = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=bare)
    a.obsm['X_common'] = pcs.to_numpy(float)
    runs = {}
    if stage in ('profiles', 'all'):
        start = time.monotonic()
        centroid = pcs.assign(_family=bare.clone_id).groupby('_family', observed=True).mean()
        sizes = bare.groupby('clone_id', observed=True).size()
        centroid = centroid.loc[sizes[sizes.ge(2)].index]
        centroid.to_csv(destination / 'RNA_centroid.csv')
        runs['RNA_centroid'] = {'seconds': time.monotonic() - start, 'n_families': len(centroid)}
        start = time.monotonic()
        centred = pcs - pcs.groupby(bare.donor, observed=True).transform('mean')
        centred = centred.assign(_family=bare.clone_id).groupby('_family', observed=True).mean().loc[centroid.index]
        centred.to_csv(destination / 'RNA_context_centroid.csv')
        runs['RNA_context_centroid'] = {'seconds':time.monotonic()-start,'n_families':len(centred),'context_key':'donor'}
        for kind in ('mean', 'kernel'):
            start = time.monotonic()
            tf.tl.clone_profiles(a, clone_key='clone_id', basis='X_common', donor_key='donor',
                                 context_key='donor', representation=kind, min_cells=2,
                                 n_features=256, n_components=30, random_state=0, verbose=False)
            z = a.uns['threadfin']['profiles']['features']
            z.to_csv(destination / f'Threadfin_{kind}.csv')
            runs[f'Threadfin_{kind}'] = {'seconds': time.monotonic() - start,
                                        'n_families': len(z), 'context_key': 'donor',
                                        'smooth': 15 if kind == 'kernel' else 0}
    if stage in ('clone2vec', 'all'):
        import clone2vec as c2v
        start = time.monotonic()
        clones = c2v.pp.clones_adata(a, obs_name='clone_id', min_size=2, fill_obs=None)
        c2v.tl.clonal_nn(a, clones, obs_name='clone_id', k=15, use_rep='X_common', random_state=0)
        c2v.tl.clone2vec(clones, z_dim=30, max_iter=500, device='cpu',
                         progress_bar=False, random_state=0)
        z = pd.DataFrame(clones.obsm['clone2vec'], index=clones.obs_names)
        z.to_csv(destination / 'clone2vec.csv')
        runs['clone2vec'] = {'seconds': time.monotonic() - start, 'n_families': len(z),
                             'version': importlib.metadata.version('clone2vec'),
                             'native_parameters': {'k': 15, 'z_dim': 30, 'max_iter': 500,
                                                   'early_stopping_patience': 5, 'init': 'svd'}}
    if stage in ('native', 'all'):
        status = json.loads((folder / 'run.json').read_text())
        if status.get('benisse_r_status') == 'completed':
            start = time.monotonic()
            work = folder / 'benisse_r'
            annotation = pd.read_csv(work / 'clone_annotation.csv', index_col=0)
            node_ids = annotation.v_gene + '_' + annotation.cdr3 + '_' + annotation.j_gene
            distance = np.loadtxt(work / 'latent_dist.txt')
            assert distance.shape == (len(annotation), len(annotation))
            features, info = squared_distance_features(distance)
            nodes = pd.DataFrame(features, index=node_ids)
            cell_keys = pd.read_csv(work / 'clonality_label.txt', header=None)[0]
            aliases = (work / 'cleaned_exp.txt').open().readline().split()
            assert len(aliases) == len(cell_keys)
            aliases_to_ids = pd.read_csv(folder / 'benisse_cell_aliases.csv').set_index('alias').cell_id
            cells = nodes.loc[cell_keys].copy()
            cells.index = aliases_to_ids.loc[aliases].values
            z = pool_cells(cells, obs)
            z.to_csv(destination / 'Benisse.csv')
            node_codes = pd.Series(np.arange(len(node_ids)), index=node_ids)
            codes = pd.Series(node_codes.loc[cell_keys].to_numpy(), index=cells.index).reindex(obs.index)
            assert codes.notna().all()
            native_kernel = family_distance_kernel(distance, codes, obs)
            native_kernel.to_csv(destination / 'Benisse_kernel.csv')
            runs['Benisse'] = {**info, 'adapter_seconds': time.monotonic() - start,
                               'n_native_nodes': len(annotation), 'n_cells': len(cells), 'n_families': len(z),
                               'performance_adapter': 'complete cell-weighted family kernel; no spectral truncation',
                               'spectral_features_used_for_performance': False}
        if status.get('bigcn_status') == 'completed':
            import torch
            work = folder / 'BiGCN_official/dataset'
            # Upstream saves a single tensor, not a model to be executed here.
            matrix = torch.load(work / 'output/newFile/embedding/embedding_cos_100_balance.pt',
                                map_location='cpu', weights_only=True).detach().numpy()
            nodes = pd.read_csv(work / 'newFile/BCR_embedding.csv', header=None).iloc[:, 1]
            assert len(nodes) == len(matrix)
            node_features = pd.DataFrame(matrix, index=nodes)
            mapping = pd.read_csv(work / 'newFile/exp_pca.csv', header=None, usecols=[0, 1])
            cells = node_features.loc[mapping[1]].copy()
            cells.index = mapping[0].values
            z = pool_cells(cells, obs)
            z.to_csv(destination / 'BiGCN.csv')
            runs['BiGCN'] = {'n_native_nodes': len(nodes), 'n_cells': len(cells), 'n_families': len(z),
                             'adapter': 'cell-count-weighted native node mean within fixed families'}
    old_path = destination / 'representation_audit.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    old.update(runs)
    old_path.write_text(json.dumps(old, indent=2) + '\n')
    print(json.dumps({'dataset': dataset, 'stage': stage, 'representations': list(runs)}), flush=True)


def fit_predict(kernel, y, train, test, alpha):
    """Train-centred, train-trace-normalised linear-kernel ridge with intercept."""
    k_train = kernel[np.ix_(train, train)]
    k_test = kernel[np.ix_(test, train)]
    means = k_train.mean(0)
    grand = means.mean()
    k_train = k_train - means[:,None] - means[None,:] + grand
    k_test = k_test - k_test.mean(1)[:,None] - means[None,:] + grand
    scale = max(float(np.trace(k_train) / len(k_train)), 1e-12)
    mean_y = y[train].mean()
    model = KernelRidge(alpha=alpha, kernel='precomputed').fit(k_train / scale, y[train] - mean_y)
    return model.predict(k_test / scale) + mean_y


def readout(kernel, metadata):
    """Whole-mouse folds; centring, scaling and alpha choice use training only."""
    rows, predictions = [], []
    for donor in sorted(metadata.donor.unique()):
        test = metadata.donor.eq(donor).to_numpy()
        train = ~test
        if train.sum() < 20 or test.sum() < 3:
            continue
        x, y = kernel.to_numpy(), metadata.target.to_numpy()
        inner_mice = sorted(metadata.loc[train, 'donor'].unique())
        losses = {alpha: [] for alpha in ALPHAS}
        for inner in inner_mice:
            inner_test = train & metadata.donor.eq(inner).to_numpy()
            inner_train = train & ~inner_test
            if inner_train.sum() < 15 or inner_test.sum() < 3:
                continue
            for alpha in ALPHAS:
                pred = fit_predict(x, y, inner_train, inner_test, alpha)
                losses[alpha].append(mean_absolute_error(y[inner_test], pred))
        alpha = min(ALPHAS, key=lambda k: np.mean(losses[k]) if losses[k] else np.inf)
        if not losses[alpha]:
            alpha = 10
        pred = fit_predict(x, y, train, test, alpha)
        rows.append({'held_out_donor': donor, 'n_train': int(train.sum()), 'n_test': int(test.sum()),
                     'r2': float(r2_score(y[test], pred)) if np.var(y[test]) > 1e-12 else np.nan,
                     'mae': float(mean_absolute_error(y[test], pred)), 'readout_alpha': alpha,
                     'n_inner_mice': len(losses[alpha]),
                     'test_target_mean':float(y[test].mean()), 'test_target_variance':float(y[test].var()),
                     'training_target_mean':float(y[train].mean()),
                     'predictions_outside_unit_interval':int(((pred<0)|(pred>1)).sum())})
        for family, observed, estimate in zip(metadata.index[test], y[test], pred):
            predictions.append({'held_out_donor': donor, 'clone_id': family,
                                'observed': observed, 'predicted': estimate})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def score(dataset):
    pcs, obs = common_input(dataset)
    sizes = obs.groupby('clone_id', observed=True).size()
    expanded = sizes.index[sizes.ge(2)]
    folder = OUT / 'representations' / dataset
    methods = ['RNA_centroid', 'RNA_context_centroid', 'Threadfin_mean', 'Threadfin_kernel', 'Benisse', 'BiGCN', 'clone2vec']
    features = {m: pd.read_csv(folder / f'{m}.csv', index_col=0) for m in methods if (folder / f'{m}.csv').exists()}
    if len(features) != len(methods):
        raise ValueError(f'{dataset}: missing native representations {set(methods) - set(features)}')
    assert features
    shared = pd.Index(sorted(set(expanded).intersection(*(set(z.index) for z in features.values()))))
    kernels = {m: pd.DataFrame(z.loc[shared].to_numpy() @ z.loc[shared].to_numpy().T,
                              index=shared, columns=shared) for m,z in features.items() if m != 'Benisse'}
    kernels['Benisse'] = pd.read_csv(folder / 'Benisse_kernel.csv', index_col=0).loc[shared,shared]
    # Zero-feature representation yields the training-target-mean comparator.
    kernels['Training_mean'] = pd.DataFrame(np.zeros((len(shared),len(shared))), index=shared, columns=shared)
    coverage = [{'dataset': dataset, 'method': m, 'n_expanded_families': len(expanded),
                 'n_complete_families': len(z.index.intersection(expanded)),
                 'coverage': len(z.index.intersection(expanded)) / len(expanded),
                 'n_common_families': len(shared), 'native_dimensions': z.shape[1] if m != 'Benisse' else 'full kernel'} for m, z in features.items()]
    coverage.append({'dataset':dataset,'method':'Training_mean','n_expanded_families':len(expanded),
                     'n_complete_families':len(expanded),'coverage':1.,'n_common_families':len(shared),
                     'native_dimensions':0})
    scores, predictions, target_tables = [], [], []
    for column, positive in TARGETS[dataset]:
        target = family_targets(obs, column, positive).reindex(shared).dropna(subset=['target'])
        target_name = column + ':' + positive
        target_tables.append(target.assign(target_name=target_name))
        for min_cells in (2, 5):
            selected = target.index[sizes.reindex(target.index).ge(min_cells) & target.n_measured.ge(min_cells)]
            for method, kernel in kernels.items():
                result, pred = readout(kernel.loc[selected,selected], target.loc[selected])
                if result.empty:
                    continue
                result = result.assign(method=method, dataset=dataset, target=target_name,
                                       min_cells=min_cells, coverage=next(v['coverage'] for v in coverage if v['method'] == method),
                                       n_common_families=len(shared), n_target_families=len(target),
                                       n_selected_families=len(selected), n_selected_donors=target.loc[selected].donor.nunique(),
                                       status='completed', representation_scope='label-blind transductive; common no-IG PCA100; fixed donor-private families; native full-geometry linear-kernel readout')
                pred = pred.assign(method=method, dataset=dataset, target=target_name, min_cells=min_cells)
                scores.append(result); predictions.append(pred)
    dest = OUT / 'scores' / dataset
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(scores, ignore_index=True).to_csv(dest / 'heldout_scores.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(dest / 'heldout_predictions.csv', index=False)
    pd.concat(target_tables).to_csv(dest / 'targets.csv')
    pd.DataFrame(coverage).to_csv(dest / 'coverage.csv', index=False)
    library_rows=[]
    libraries=obs.index.to_series().str.split('|',n=1).str[0]
    for column,positive in TARGETS[dataset]:
        for (library,gate),count in pd.DataFrame({'library':libraries,'gate':obs[column].fillna('Unmeasured')}).value_counts().items():
            library_rows.append({'dataset':dataset,'gate_field':column,'source_library_prefix':library,
                                 'gate':gate,'n_common_cells':count})
    pd.DataFrame(library_rows).to_csv(dest/'source_library_gate_counts.csv',index=False)
    manifest={'dataset':dataset,'n_common_cells':len(obs),'n_common_families':len(shared),
              'cell_ids_sha256':hashlib.sha256('\n'.join(sorted(obs.index)).encode()).hexdigest(),
              'family_ids_sha256':hashlib.sha256('\n'.join(shared).encode()).hexdigest(),
              'gate_labels_exported_to_models':False,'all_methods_required':True,
              'unsupervised_scope':'all common cells, transductive, no held-out gate labels',
              'readout':'complete-geometry linear-kernel ridge; training-only centring and trace scaling; nested whole-mouse alpha CV',
              'alphas':ALPHAS,'fold_min_train':20,'fold_min_test':3,
              'target_definition':'positive fraction among measured cells only; n_measured >= min_cells',
              'sort_library_scope':'shared sorted sequencing libraries across mouse folds; not an independent-library generalisation test',
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','anndata','clone2vec']},
              'native_model_run':json.loads((OUT/dataset/'run.json').read_text()),
              'representations':json.loads((folder/'representation_audit.json').read_text())}
    (dest/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'dataset': dataset, 'methods': list(features), 'common_families': len(shared),
                       'readout': 'native full-geometry linear-kernel ridge', 'rows': sum(len(x) for x in scores)}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(TARGETS))
    ap.add_argument('--stage', choices=['profiles', 'clone2vec', 'native', 'all', 'score'], default='profiles')
    args = ap.parse_args()
    if args.stage == 'score':
        score(args.dataset)
    else:
        build_representations(args.dataset, args.stage)
