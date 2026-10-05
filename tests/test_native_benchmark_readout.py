"""Scientific readout contracts: missing labels and complete-family coverage."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('benchmark_score', Path(__file__).parents[1] / 'case_studies/score_native_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_unmeasured_probe_does_not_become_nonbinding():
    obs = pd.DataFrame({'clone_id': ['a', 'a', 'a', 'b'], 'donor': ['m1']*4,
                        'bait': ['positive', None, 'negative', None]})
    targets = benchmark.family_targets(obs, 'bait', 'positive')
    assert targets.loc['a', 'target'] == 0.5
    assert targets.loc['a', 'n_measured'] == 2
    assert 'b' not in targets.index


def test_native_node_pooling_weights_cells_and_excludes_partial_families():
    obs = pd.DataFrame({'clone_id': ['a', 'a', 'a', 'b', 'b']}, index=['c1','c2','c3','c4','c5'])
    # Native node 1 has two cells, node 2 one cell: family mean must be 2, not 3.
    cells = pd.DataFrame({'latent': [0., 0., 6., 1.]}, index=['c1','c2','c3','c4'])
    pooled = benchmark.pool_cells(cells, obs)
    assert pooled.loc['a', 'latent'] == 2
    assert 'b' not in pooled.index


def test_benisse_adapter_uses_squared_distance_without_squaring_it_again():
    points = np.array([[0.,0.],[1.,0.],[0.,2.],[2.,3.]])
    squared = ((points[:,None,:] - points[None,:,:])**2).sum(axis=2)
    z, audit = benchmark.squared_distance_features(squared, n_components=3)
    reconstructed = ((z[:,None,:] - z[None,:,:])**2).sum(axis=2)
    np.testing.assert_allclose(reconstructed, squared, atol=1e-8)
    assert audit['n_components'] == 2


def test_full_native_family_kernel_preserves_cell_weighted_geometry():
    points = np.array([[0.,0.],[1.,0.],[0.,2.]])
    distance = ((points[:,None,:] - points[None,:,:])**2).sum(axis=2)
    obs = pd.DataFrame({'clone_id':['a','a','a','b','b']}, index=list('vwxyz'))
    node = pd.Series([0,0,1,1,2], index=obs.index)
    kernel = benchmark.family_distance_kernel(distance, node, obs).to_numpy()
    means = np.stack([(2*points[0]+points[1])/3,(points[1]+points[2])/2])
    means -= means.mean(0)
    np.testing.assert_allclose(kernel, means @ means.T, atol=1e-12)


def test_kernel_readout_matches_primal_train_centred_ridge():
    from sklearn.linear_model import Ridge
    x = np.array([[1.,0.],[2.,1.],[3.,0.],[5.,2.],[20.,30.]])
    y = np.array([.1,.2,.5,.8,.9])
    train = np.array([True,True,True,True,False]); test=~train
    scale = np.sqrt(np.var(x[train],axis=0).sum())
    primal = Ridge(alpha=1).fit(x[train]/scale,y[train]).predict(x[test]/scale)
    pred = benchmark.fit_predict(x @ x.T,y,train,test,1)
    np.testing.assert_allclose(pred,primal,atol=1e-10)


def test_parallel_model_manifest_updates_preserve_both_completed_stages(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    native_spec=importlib.util.spec_from_file_location('native_adapter',Path(__file__).parents[1]/'case_studies/native_benchmark.py')
    native=importlib.util.module_from_spec(native_spec);native_spec.loader.exec_module(native)
    destination=tmp_path/'run.json'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i:native.stamp(destination,**{f'stage_{i}':'completed'}),range(24)))
    result=json.loads(destination.read_text())
    assert result=={f'stage_{i}':'completed' for i in range(24)}
