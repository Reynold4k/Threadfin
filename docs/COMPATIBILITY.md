# Verified compatibility

## Version 4.7.0 — 16 November 2026

* **Why this version.** Versions 4.5.3–4.6.2 held every installation at pynndescent < 0.7, because Scanpy 1.11
  cannot read the neighbour graphs of pynndescent 0.7: `sc.pp.neighbors` then fails on 8,192 or more cells with
  "'csr_array' object has no attribute 'getnnz'". Scanpy reads them from 1.12.5 on, a release that needs Python 3.12.
  umap-learn 0.6 needs pynndescent 0.7, so the pin also kept umap-learn below 0.6. Threadfin 4.7.0 keeps pynndescent
  below 0.7 only where pip cannot install a Scanpy that reads it.

### What pip installs

What `pip install --dry-run` resolves (pip 26.3, PyPI on 16 November 2026). Every row with Python 3.10–3.14 was also
installed from the wheel of this version in a new virtual environment: `pip check` passed, Scanpy's own neighbour
search, UMAP and Leiden ran on 12,000 cells, and the test suite passed (below).

| Python | `pip install threadfin` | `pip install "threadfin[dandelion]"` |
|---|---|---|
| 3.10 | Scanpy 1.11.5, pynndescent 0.6.0, umap-learn 0.5.12 | Scanpy 1.11.5, pynndescent 0.6.0, umap-learn 0.5.12, matplotlib 3.8.4 |
| 3.11 | Scanpy 1.11.5, pynndescent 0.6.0, umap-learn 0.5.12 | Scanpy 1.11.5, pynndescent 0.6.0, umap-learn 0.5.12, matplotlib 3.8.4 |
| 3.12 | Scanpy 1.12.6, **pynndescent 0.7.0, umap-learn 0.6.0** | Scanpy 1.11.5, pynndescent 0.6.0, umap-learn 0.5.12, matplotlib 3.8.4 |
| 3.13 | Scanpy 1.13.0, **pynndescent 0.7.0, umap-learn 0.6.0** | Scanpy 1.12.6, **pynndescent 0.7.0, umap-learn 0.6.0**, matplotlib 3.11.2 |
| 3.14 | Scanpy 1.13.0, **pynndescent 0.7.0, umap-learn 0.6.0** | Scanpy 1.12.6, **pynndescent 0.7.0, umap-learn 0.6.0**, matplotlib 3.11.2 |

Threadfin 4.6.2 resolves to pynndescent 0.6.0 and umap-learn 0.5.12 in every row; in bold what 4.7.0 changes. Every
pair that pip installs works with Scanpy's own neighbour search: pynndescent 0.7 only with Scanpy 1.12.5 or later.
An environment can still hold an older Scanpy with pynndescent 0.7 (after `pip install -U pynndescent`, or when other
packages keep Scanpy older); `import threadfin` then warns and names the fix.

The test suite of this version (`python -m pytest tests`, against the installed wheel), in those environments and
two more:

| Environment | Scanpy, pynndescent, umap-learn, numpy, pandas, anndata | Tests |
|---|---|---|
| Python 3.10, the oldest versions `pyproject.toml` allows | 1.10.0, 0.5.8, 0.5.5, 1.24.1, 2.0.0, 0.10.0 | 345 passed, 19 skipped |
| Python 3.10 | 1.11.5, 0.6.0, 0.5.12, 2.2.6, 2.3.3, 0.11.4 | 345 passed, 19 skipped |
| Python 3.11, `requirements-lock.txt` (the validated versions) | 1.11.5, 0.6.0, 0.5.12, 2.4.6, 2.3.3, 0.12.19 | 359 passed, 5 skipped |
| Python 3.11 | 1.11.5, 0.6.0, 0.5.12, 2.4.6, 2.3.3, 0.12.19 | 345 passed, 19 skipped |
| Python 3.12 | 1.12.6, 0.7.0, 0.6.0, 2.5.5, 3.1.1, 0.13.4 | 345 passed, 19 skipped |
| Python 3.12, Dandelion extra | 1.11.5, 0.6.0, 0.5.12, 2.5.5, 3.1.1, 0.13.4 | 359 passed, 5 skipped |
| Python 3.13 | 1.13.0, 0.7.0, 0.6.0, 2.5.5, 3.1.1, 0.13.4 | 345 passed, 19 skipped |
| Python 3.13, Dandelion extra | 1.12.6, 0.7.0, 0.6.0, 2.5.5, 3.1.1, 0.13.4 | 359 passed, 5 skipped |
| Python 3.14 | 1.13.0, 0.7.0, 0.6.0, 2.5.5, 3.1.1, 0.13.4 | 345 passed, 19 skipped |
| Python 3.14, Dandelion extra | 1.12.6, 0.7.0, 0.6.0, 2.5.5, 3.1.1, 0.13.4 | 359 passed, 5 skipped |

Five tests need the repository's `manuscript/` folder, which the wheel does not contain, or scirpy; without the
Dandelion extra, 14 more need Dandelion. In the repository with the validated versions, 380 tests passed and 1 was
skipped (scirpy is not installed). Every environment also passed `pip check`, Scanpy's neighbour search, UMAP and
Leiden on 12,000 cells, and `examples/tutorial.py`.

### Unchanged with the validated versions

The validated versions are those of [`requirements-lock.txt`](tutorials/requirements-lock.txt): Python 3.11,
pynndescent 0.6.0, umap-learn 0.5.12, Scanpy 1.11.5, sc-dandelion 0.5.7, matplotlib 3.8.4. With them, 4.7.0 gives every
result of 4.6.2:

* **The 16 validation data sets.** The user-path script of the 4.6.0 validation (`run_dataset.py`, unchanged) ran
  every data set with this version: every clone table, summary and overview figure is identical, byte for byte, to
  the 4.6.0 record, as are the cells, clones, clonal ICC, its P value, the programmes and the clone-map settings of
  each record. The Methods paragraphs differ in the version and in naming umap-learn 0.5.12 and, above 50,000 cells,
  pynndescent 0.6.0. The final code of this version, run again on all 16, gave the same.
* **Tutorial 5** ran end to end in an environment built as Step 0 says (conda-forge Python 3.11; the newest versions
  pip installs today are those of the lock file) with the real GEO download: every number it prints is the same as
  in the 4.6.2 copy, and the clone, programme and cell tables and the overview figure it saves are identical to
  those of 4.6.2 and 4.6.0. The differences are the version, timings and a line number in a warning. The published
  copy was executed with the final code of this version (Step 2 now imports the 10x reader in a form that works with
  every Scanpy from 1.10 to 1.13).
* **Documentation.** Every code block of the README, Tutorials 1, 3 and 4, `docs/RECLUSTERING.md` and
  `docs/STATE_DENSITY.md` (on the file Tutorial 5 saves) ran with this version and printed what the pages show; the
  blocks that 4.6.2 also had printed what 4.6.2 printed, apart from the version and the Methods paragraph. With the
  newest versions (Python 3.14, Scanpy 1.13, pynndescent 0.7, umap-learn 0.6), every block printed the same, apart
  from the versions that the Methods paragraph names and the last example of `docs/RECLUSTERING.md` (10 clone
  clusters instead of 11, from Scanpy's own neighbour search).

### pynndescent 0.6 and 0.7 on large data sets

Threadfin finds approximate neighbours in two places: when it smooths the cells of more than 50,000 cells (clone
profiles), and when UMAP lays out 4,096 or more profiled clones (the clone map). pynndescent 0.6 and 0.7
(umap-learn 0.5 and 0.6) find slightly different neighbours there. The seven validation data sets of these sizes ran
with both, everything else equal (the validated versions, with pynndescent 0.7.0 and umap-learn 0.6.0 in place of
0.6.0 and 0.5.12):

| Data set | Cells | Profiled clones | Between-clone distances, r | Clonal ICC (P) | Programmes, 0.6 → 0.7 (stability) | Same partition (ARI) | Clone map, 0.6 → 0.7 (n_neighbors, min_dist) | Conclusions that change |
|---|---:|---:|---:|---|---|---:|---|---|
| bone_marrow_pc | 115,144 | 6,583 | 0.9997 | 0.427 (0.005), identical | 3 → 3 (0.34, 0.80, 0.64 → 0.33, 0.75, 0.66) | 0.96 | 70, 0.95 → 70, 0.80 | none |
| covid_bcell | 61,423 | 588 | 0.9991 | 0.400 (0.005), identical | 1 → 1 (1.00 → 0.99) | 1.00 | 56, 0.95 → 32, 0.95 | none |
| ebv | 205,630 | 6,124 | 0.9997 | 0.158 (0.005), identical | 2 → 4 (0.79, 0.54 → 0.78, 0.65, 0.49, 0.04) | 0.60 | 65, 0.80 → 37, 0.80 | programmes 2 → 4; programme effects with FDR < 0.05: gfp 1 → 4, isotype 4 → 13 |
| flu | 123,693 | 2,082 | 0.9998 | 0.561 (0.005), identical | 5 → 5 (0.79, 0.90, 0.78, 0.87, 0.34 → 0.83, 0.90, 0.83, 0.83, 0.22) | 0.90 | 180, 0.80 → 180, 0.95 | none |
| larry_d46 | 44,664 | 6,174 | 1.0000 | 0.598 (0.005), identical | 8 → 8, identical | 1.00 | 66, 0.55 → 66, 0.55 (layout differs slightly) | none |
| ln_vaccine | 193,442 | 11,126 | 1.0000 | 0.326 (0.005), identical | 4 → 4 (0.81, 0.93, 0.60, 0.92 → 0.82, 0.94, 0.60, 0.91) | 0.83 | 37, 0.95 → 67, 0.95 | programme effects with FDR < 0.05: isotype 12 → 13 |
| malaria_late | 72,359 | 1,007 | 1.0000 | 0.454 (0.005), identical | 4 → 4 (0.80, 0.41, 0.73, 0.71 → 0.81, 0.41, 0.72, 0.65) | 0.96 | 32, 0.80 → 18, 0.95 | none |

Stabilities are those of the summaries, listed for programmes matched by their clones (programme numbers follow
size, so they can swap).
"r" is the correlation of the distances between clone profiles under the two versions (300,000 random pairs). The
clonal memory index and its P value are identical on every data set that has one (bone_marrow_pc, covid_bcell, ebv,
flu, ln_vaccine).

* **Unchanged:** the clonal ICC, its shuffled value and P value, and clonal memory (index, interval and P value),
  which are computed on unsmoothed cells; the P value of every label test.
* **Changed slightly:** the clone profiles (between-clone distances correlate 0.999 or more); the variance explained
  by each label (by at most 0.03 percentage points); the programme partitions and stabilities; the automatically
  chosen clone map, on six of the seven data sets. Programmes are numbered by size, so when sizes change, numbers can
  swap (bone_marrow_pc: P1 and P3).
* **Conclusions.** On six data sets, the number of programmes, the number of programmes flagged as unstable
  (stability below 0.6) and every label test are the same. On ln_vaccine, one programme effect more passes FDR 0.05
  (isotype: 12 → 13). On ebv (205,630 cells), the partition changes: with pynndescent 0.6 two programmes, of which
  P2 (2,431 clones) is unstable (0.54); with 0.7 four, of which P2 (2,002 clones, stability 0.65) is not flagged and
  P3 and P4 (391 and 159 clones) are. P1, the IgM-enriched majority, is the same programme (Jaccard overlap 0.84,
  stability 0.79 and 0.78). The programme effects follow the partition (FDR < 0.05: gfp 1 → 4, isotype 4 → 13). On
  ln_vaccine, the layout search found no setting with 0.7 that kept 98.5 % of clones in one piece in every layout
  (89.3 %, n_neighbors 67), where it found one with 0.6 (99.7 %, n_neighbors 37); the returned maps keep 98.6 % and
  99.7 % of clones in one piece.
* **Below these sizes.** On the nine smaller data sets, profiles, programmes, every statistic and the summaries are
  identical (profiles to the last digit). umap-learn 0.6 alone (checked with pynndescent 0.6) lays out some trial
  maps of the automatic search differently from 0.5: on larry_d2, those at n_neighbors 18 and 32 (the chosen
  min_dist changed from 0.80 to 0.95, n_neighbors 100 in both), and on spont_gc those at n_neighbors 56 (the same
  setting was chosen). The other eight clone maps are identical.
* **Newest stack.** With every other package at its newest version as well (Python 3.14.10, numpy 2.5.5, SciPy 1.18.1,
  pandas 3.1.1, anndata 0.13.4, Scanpy 1.12.6, scikit-learn 1.9.1, matplotlib 3.11.2, sc-dandelion 0.5.7, installed
  as `pip install "threadfin[dandelion]"`), all 16 data sets give the results of pynndescent 0.7 above, to within
  1.7e-14 (after matching the signs of principal components), with the same clone maps and programmes. One summary
  differs, because of the data preparation rather than Threadfin: the malaria loader writes the cell states with
  `astype(str)`, which under pandas 2 turns 298 missing states into the text "nan" (Threadfin warns about such a
  level) and under pandas 3 keeps them missing, so the programme tests of `cell_state` lose the level "nan" and one
  more programme effect passes FDR 0.05 (16 → 17).
* **Run time.** The smoothing step takes longer with pynndescent 0.7 (in a new session, 34 instead of 18 s on
  205,630 cells; 41 instead of 17 s on 61,423 cells). The complete `tf.run` varied by up to 40 % between runs on the
  shared cluster nodes, more than any difference between the versions.
* **To reproduce the published numbers exactly**, install the validated versions:
  `pip install -r docs/tutorials/requirements-lock.txt` (Python 3.11), or `pip install "pynndescent<0.7"`, which also
  brings back umap-learn 0.5, on any Python from 3.10 to 3.14. The Methods paragraph of each run names the versions
  of pynndescent and umap-learn it used.

### Python 3.10: SciPy and the clone map

On Python 3.10, pip installs SciPy 1.15.3, the last SciPy for that Python. UMAP starts every layout from a spectral
embedding that SciPy's eigensolver (ARPACK) computes. SciPy 1.15 and 1.16 return these eigenvectors with the
opposite sign of SciPy 1.17 (the validated version) and 1.18, so UMAP starts from the point-reflected layout, and its
stochastic optimisation ends in a slightly different map. The clone profiles, the neighbour graph that UMAP builds,
the programmes and every statistic are identical; the scores of the maps differ in the third decimal, and the
automatic search can settle on other settings:

| Simulated data of | SciPy 1.17 (validated; Python 3.11) | SciPy 1.15 (Python 3.10) |
|---|---|---|
| Tutorial 1 | n_neighbors 56, min_dist 0.95; 99.8 % of clones in one piece | n_neighbors 56, min_dist 0.95; 99.7 % |
| Tutorial 3, section 2 | quality score 0.739, coherence 0.998, local fidelity 0.578, global fidelity 0.900; six confirmed settings | 0.742, 0.997, 0.571, 0.913; five confirmed settings |
| Tutorial 4, step 3 | n_neighbors 100, min_dist 0.95 | n_neighbors 32, min_dist 0.95 |

The validated versions with SciPy 1.15.3 or 1.16.3 in place of 1.17.1 give the Python 3.10 numbers exactly (the
spectral starting layouts are each other's negatives to within 1.3e-15); with scikit-learn 1.7.2 or numpy 2.2.6,
the other versions that differ on Python 3.10, they give those of Python 3.11. The same holds on any Python with
SciPy 1.16 or earlier. The numbers of the tutorials and of the validation were obtained with SciPy 1.17.1, which
the lock file installs.

### Exact neighbour search (evaluated, not adopted)

Exact neighbours at every size would make the results independent of pynndescent: the smoothing as below 50,000
cells (scikit-learn), and UMAP given exact neighbours (`precomputed_knn`) for 4,096 or more clones. Run on the same
seven data sets with both versions:

| Data set | Smoothing, s: 0.6 / 0.7 / exact | `tf.run`, s: 0.6 / 0.7 / exact | Peak memory, GB: 0.6 / 0.7 / exact | Exact against 0.6: distances r; map | Exact against 0.7: distances r; map | Conclusions of exact search that differ from 0.6 |
|---|---|---|---|---|---|---|
| bone_marrow_pc | 16 / 34 / 12 | 397 / 404 / 383 | 6.9 / 7.2 / 7.0 | 0.9998; 70, 0.95 → 123, 0.95 | 0.9998; 70, 0.80 → 123, 0.95 | programme effects with FDR < 0.05: sort 10 → 12, isotype 8 → 9 |
| covid_bcell | 18 / 51 / 7 | 176 / 200 / 163 | 4.1 / 4.2 / 4.0 | 0.9992; 56, 0.95 → 32, 0.95 | 0.9992; the same map | none |
| ebv | 20 / 41 / 46 | 570 / 948 / 873 | 25.7 / 25.8 / 25.7 | 0.9998; 65, 0.80 → 37, 0.80 | 0.9998; the same settings | those of pynndescent 0.7: programmes 2 → 4; gfp 1 → 4, isotype 4 → 13 |
| flu | 17 / 33 / 16 | 312 / 315 / 338 | 8.1 / 8.3 / 8.4 | 0.9998; 180, 0.80 → 180, 0.95 | 0.9998; the same settings | programmes 5 → 4; programme effects with FDR < 0.05: time 5 → 4 |
| larry_d46 | 2 / 2 / 2 | 215 / 222 / 192 | 1.0 / 1.1 / 0.8 | 1.0000; the same settings | 1.0000; the same settings | none |
| ln_vaccine | 26 / 48 / 34 | 838 / 1,332 / 946 | 23.8 / 24.0 / 23.8 | 1.0000; 37, 0.95 → 67, 0.95 | 1.0000; 67, 0.95 → 37, 0.95 | those of pynndescent 0.7: isotype 12 → 13 |
| malaria_late | 19 / 36 / 12 | 338 / 308 / 284 | 6.6 / 6.8 / 6.6 | 1.0000; 32, 0.80 → 32, 0.95 | 1.0000; 18, 0.95 → 32, 0.95 | none |

Exact search ran with pynndescent 0.6 and with 0.7 (each with its umap-learn); the table shows the first, and the
second gave identical profiles, programmes, tests and summaries. larry_d46 has fewer than 50,000 cells, so only its
clone map changes method.

* **Time and memory.** Exact smoothing took 4 to 24 s per call (24 s on the 205,630 cells of ebv), against 1 to 9 s
  for pynndescent once compiled; pynndescent's first call in a session also compiles (15 to 20 s with 0.6, 31 to
  41 s with 0.7). Over the two smoothing calls of `tf.run`, exact search was faster on four of the six data sets
  above 50,000 cells and slower on the two largest (ebv: 46 against 20 s; ln_vaccine: 34 against 26 s). The complete
  `tf.run` took 0.8 to 1.5 times as long as with pynndescent 0.6 (repeated runs varied by up to 40 % on the shared
  nodes), and the peak memory changed by at most 0.4 GB (ebv: 25.7 GB with each).
* **Results.** With exact search, pynndescent 0.6 and 0.7 give identical profiles, programmes, tests and summaries;
  the clone maps are identical on five of the seven data sets (on bone_marrow_pc and ln_vaccine, umap-learn 0.6
  lays out the same neighbours slightly differently, and on ln_vaccine the search settles on n_neighbors 37 instead
  of 67). The exact results differ from either approximate version about as much as these differ from each other
  (between-clone distances correlate 0.999 or more), so adopting exact search would change the large-data results
  of the validation, and more of them than the change of pynndescent does: the number of programmes on ebv
  (2 → 4) and flu (5 → 4), and programme effects passing FDR 0.05 on bone_marrow_pc, ebv, flu and ln_vaccine. The
  clonal ICC, its P value and clonal memory do not change; the variance explained by each label changes by at most
  0.08 percentage points, with the same P values.
* Not adopted in 4.7.0; the default is unchanged.

### BLAS threads

Results can depend on the number of BLAS threads (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`), and
only through one step: the expression embedding that `tf.pp.prepare_embedding` builds when `tf.run` is given no
`basis`. Each step of `tf.run` was run with 8 threads and every other step with 1: with 8 threads in
`prepare_embedding` alone, a run gives exactly the result of 8 threads everywhere, and with 8 threads in any other
step alone, exactly the result of 1 thread. With 1 and 8 threads everywhere (same `n_jobs`; validated versions,
except for the spaclone row, which ran with the newest versions on Python 3.12):

| Data | Embedding | What differs between 1 and 8 threads |
|---|---|---|
| Tutorial 1 (simulated) | given (`basis="X_pca"`) | nothing: profiles within 2.2e-15, the same clone map, programmes and summary |
| GSE287123 (Tutorial 5) | built by Threadfin | embedding by up to 0.003; clonal ICC 0.16918594 / 0.16918596 and clonal memory 0.6490177 / 0.6490177, each with the same P value; variance explained by each label within 1e-6, with the same P values; the same three programmes (ARI 1.0) with stabilities 0.64, 0.52, 0.14 / 0.66, 0.55, 0.21; the same programme effects passing FDR 0.05; clone map n_neighbors 18, min_dist 0.80 / n_neighbors 10, min_dist 0.95 |
| AIR-SPACE, six lymph nodes, through spaclone 1.0.2's Threadfin bridge | built by Threadfin | embedding by up to 0.018; clonal ICC and its P value identical to six decimals; on day 3 the mean stability in the programme warning 0.67 / 0.63, the programme stabilities 0.593, 0.496 / 0.594, 0.498 |

No conclusion changed. Tutorial 5 run with 1 thread prints the same in 15 of its 18 code cells; the other three
differ in the clone-map settings, the programme stabilities above and the sixth decimal of a cosine similarity.
In spaclone's bridge, the bead-type scores, the counts and the annotations it passes to
Threadfin are identical with 1 and 8 threads; the difference arises in Threadfin's expression embedding, and 8
threads inside `prepare_embedding` alone reproduce the 8-thread result.

Making results independent of the thread count would mean running that step on a fixed number of threads (with
`threadpoolctl`). Fixing it at 1 costs little time (on GSE287123 the complete `tf.run` took 102 s with 1 thread
everywhere and 100 s with 8), but it would change the published numbers wherever Threadfin built the embedding,
because the validation and Tutorial 5 ran with 8 threads (GSE287123 with 8 threads gives the published clone map,
n_neighbors 10 and min_dist 0.95). Fixing it at 8 would keep them on processors and BLAS libraries like those of the
validation, but would oversubscribe machines with fewer cores. 4.7.0 changes neither: keep the thread count the same
between runs that must give the same programmes and map.

### Python 3.15

`pip install threadfin` does not complete on Python 3.15.2: h5py (which anndata needs), numcodecs and harmonypy have
no Python 3.15 wheels yet, and h5py does not build without the HDF5 library ("Unable to load dependency HDF5").
With the compiled packages from conda-forge (numpy 2.5.4, SciPy, pandas 3.1.1, h5py 3.16.0, numba 0.68.1, numcodecs
and others; harmonypy 2.1.0 built by pip) and the rest from pip (Scanpy 1.13.0, pynndescent 0.7.0, umap-learn 0.6.0,
anndata 0.13.4), `pip check` passed, Scanpy's neighbour search ran, the test suite passed (340 passed, 19 skipped: no
Dandelion or scirpy, and no `manuscript/`) and so did `examples/tutorial.py`. Python 3.15 is therefore not yet
listed as supported; it can be once h5py publishes Python 3.15 wheels.

### Dandelion 1.x (not supported in this version)

sc-dandelion 1.0.2 (16 November 2026) works through Threadfin's Dandelion adapter, but calls clones differently.
Dandelion 0.5.7 joins a junction to its neighbours within the identity threshold; 1.0 takes connected components
(single linkage) within each V–J–length group, so chains of similar junctions become one clone. Its clone membership
no longer depends on Python's hash seed (identical partitions under seeds 0, 1 and 2; the clone ids are numbered
differently). On five data sets with contigs, the same cells and the same embedding (validated stack; 1.0.2 on
Python 3.12):

| Data set | Clones (two or more cells), 0.5.7 → 1.0.2 | Largest clone | Adjusted Rand index | Profiled 0.5.7 clones kept exactly | Clonal ICC (P) | Programmes | Label tests: variance explained (P) | Clonal memory (P) |
|---|---|---|---|---|---|---|---|---|
| mouse_rbd (GSE287123, Tutorial 5) | 8,736 (1,881) → 7,208 (1,415) | 252 → 494 | 0.57 | 952 of 1,881 (929 merged) | 0.169 → 0.157 (0.005 / 0.005) | 3 → 2 | RBD bait 2.3 → 1.2% (0.0005 → 0.003); division 8.5 → 11.6%; zone 3.7 → 3.7%; isotype 4.6 → 5.1% | 0.65 → 0.53 (0.002 / 0.002) |
| gut_gc_reporter (GSE284622) | 6,963 (964) → 6,388 (749) | 139 → 321 | 0.61 | 507 of 964 (457 merged) | 0.401 → 0.362 (0.005 / 0.005) | 3 → 3 | GC sort 5.4 → 6.4%; tissue 51.4 → 49.3%; isotype 21.1 → 21.4% | 0.03 → 0.03 (0.018 / 0.014) |
| malaria (GSE286215, exp. 1) | 20,500 (523) → 20,492 (525) | 8 → 8 | 0.99 | 518 of 523 (5 merged) | 0.735 → 0.728 (0.005 / 0.005) | 4 → 4 | state 39.1 → 38.8%; isotype 15.2 → 15.3% | – |
| spont_gc (GSE203132) | 2,879 (1,053) → 2,679 (918) | 111 → 340 | 0.60 | 818 of 1,053 (235 merged) | 0.279 → 0.270 (0.005 / 0.005) | 3 → 3 | isotype 3.7 → 4.0% | – |
| gc_np_pc (GSE246382) | 609 (92) → 557 (100) | 11 → 19 | 0.64 | 69 of 92 (23 merged) | 0.049 → 0.045 (0.24 / 0.25) | 2 → 2 | fate 77.7 → 73.6% | – |

Label-test P values are 0.0005 in both unless shown. Clone-map settings changed on every data set.

No 0.5.7 clone is split by 1.0.2; every change merges clones. Supporting 1.x would change every number that rests on
clone calls, including those of the manuscript and of the tutorials (Tutorial 5: 7,208 instead of 8,736 clones),
so the extra keeps sc-dandelion < 1.

### harmonypy on Windows

No release of harmonypy publishes Windows wheels (2.1.0 has wheels for Linux and macOS only), and 2.x does not
compile with Microsoft's compiler, so Windows keeps harmonypy < 2 (pure Python).

### Two changes in the newest dependencies

* **umap-learn 0.6** has a second layout pipeline (recursive initialisation, the Adam optimiser and optionally the
  RINND neighbour search). It still uses the pipeline of umap-learn 0.5 by default, but warns that a later release
  will not; Threadfin asks for the 0.5 pipeline (`compatibility_layout=True`). It gives the maps of umap-learn 0.5
  for most neighbour graphs, not all (the trial maps of larry_d2 and spont_gc above).
* **anndata 0.13** keeps `X` among the layers, under the key `None`, so `for k in list(adata.layers): del
  adata.layers[k]` also deletes `X`. Threadfin then says that `adata.X` is empty instead of failing in NumPy. The
  manuscript's loader of ln_vaccine does this; for the newest-stack run above, deleting the key `None` was skipped
  while the data were read, which gives the loader's result of anndata 0.12.

## Version 4.6.2 — 16 November 2026

* **Why this version.** A first-time user followed the public 4.6.1 release in new environments, with the public
  documentation only: `pip install "threadfin[dandelion]"` on Python 3.11 and 3.12 and `pip install threadfin` on
  Python 3.14, the README Quick start, Tutorials 1–5 and `examples/tutorial.py`. Every number they printed matched
  the documentation; what else the test found is fixed in this version ([changelog](../CHANGELOG.md)).
* **Tests.** In the validation environment (Python 3.11.18, sc-dandelion 0.5.7, pandas 2.3.3, numpy 2.4.6, anndata
  0.12.19, scanpy 1.11.5) 360 passed and 1 was skipped (scirpy was not installed). With the newest releases of the
  dependencies (Python 3.12.16, pandas 3.1.1, numpy 2.5.5, anndata 0.13.4, scanpy 1.12.6), 346 passed and 15 that
  need Dandelion or scirpy were skipped, and the same with the oldest versions that `pyproject.toml` allows,
  installed as continuous integration installs them (Python 3.10, numpy 1.24.1, pandas 2.0.0, anndata 0.10.0,
  scanpy 1.10.0).
* **Dandelion on Python 3.13 and 3.14.** Scanpy 1.13.0 (13 November 2026) removed `scanpy.plotting._tools`, which
  sc-dandelion 0.5.7 imports. The Dandelion extra therefore installs Scanpy below 1.13; continuous integration
  runs the Dandelion tests on Python 3.12, 3.13 and 3.14 with it. Without the extra, Threadfin runs with Scanpy
  1.13.
* **Simulated data.** Every code block of Tutorials 1, 3 and 4, of `docs/RECLUSTERING.md` and of the README (apart
  from the call on the reader's own files) ran, and so did `examples/tutorial.py`. Tutorials 1, 3 and 4 and
  `examples/tutorial.py` printed the same as with 4.6.1 apart from the version, and Tutorial 1's clone table and
  overview figure are identical, byte for byte. A new virtual environment with the wheel of this version and the
  newest dependencies (Python 3.12) passed `pip check` and printed the same in `examples/tutorial.py`.
* **The tutorial notebook** ran end to end in a new environment built as Step 0 describes (Python 3.11 from
  conda-forge; this version, sc-dandelion 0.5.7, matplotlib 3.8.4), with the download from GEO: no error, 6.4
  minutes on 8 cores, 7.6 GB. Twelve of its 18 code cells print what the published 4.6.0 copy printed. The other
  six differ in the version, the quality-control plot (now shown), the corrected log message, the time of the
  map search and a line number, and the cosine similarity, now printed in full (0.8962615719357584). The clone
  and programme tables, the cell table and the overview figure it saves are identical, byte for byte, to those of
  the 4.6.0 run. Every package version is in `docs/tutorials/requirements-lock.txt`.
* **Tutorial 2 on public data.** Every code block of Tutorial 2 ran on GSE287123 with the substitutions its first
  section lists, in the environment of the notebook (with scirpy 0.22.5 for its scirpy block): 34,175 of the
  36,188 singlets passed the quality control, and 8,660 clones were called, 1,859 of them with two or more cells,
  the same through scirpy. After the later `how="fraction:IGHA"` test of section 7, the result rebuilt from the
  saved file printed the summary saved before the test (with 4.6.1 it reported that later test instead).
* **Saved results.** Files written by 4.6.0 and 4.6.1 load as before. Threadfin 4.6.0 reads files written by 4.6.2.

## Version 4.6.0 — 30 October 2026

* **The user's path.** A new environment was built as Tutorial 2 describes (Python 3.11, `pip install
  "threadfin[dandelion]"` from this version: pynndescent 0.6.0, numba 0.68.1, scanpy 1.11.5, sc-dandelion 0.5.7,
  matplotlib 3.8.4), and `tf.run` was called on the same 16 datasets, with `strata_key="sample"` for the 14 BCR
  datasets: no error, and every clone table — clone calls, reliabilities, clone-map coordinates and programmes — is
  identical, byte for byte, to that of 4.5.2, as is every clonal ICC. The shuffled ICC of the coherence test now
  comes from shuffles within samples; every test that was significant stays at P = 0.005 (200 permutations). In
  GSE171703 three libraries pool several donors; the strata there are donor × library (a fix this run found).
  Every code block of Tutorials 1, 3 and 4 ran (23 of 23) and printed what the tutorials show, apart from timings.
* **The tutorial notebook** ran end to end in a separately built conda environment without `PYTHONHASHSEED`:
  no error, 5.6 minutes on 8 cores including the download, 7.2 GB; the same 8,736 clones, 1,881 of them profiled,
  and the same ICC (0.169) as with 4.5.3. Every package version of that run is in
  `docs/tutorials/requirements-lock.txt`.
* **Tests.** In that environment 359 passed and 1 was skipped (scirpy was not installed). With the newest releases
  of the dependencies (Python 3.12, pandas 3.1, numpy 2.5, anndata 0.13, scanpy 1.12), 345 passed and 15 that need
  Dandelion or scirpy were skipped; continuous integration passed on Python 3.10–3.14, macOS, Windows, the oldest
  allowed dependencies and with Dandelion on Python 3.12–3.14.
* **Python 3.13 and 3.14 with Dandelion.** The Dandelion extra no longer asks for matplotlib < 3.9 on Python 3.13
  and later, where importing Threadfin restores the `matplotlib.cm.get_cmap` that Dandelion 0.5.7 needs. With
  sc-dandelion 0.5.7 and matplotlib 3.11.2, clone calls on three datasets (GSE246382, GSE287123 RBD, GSE286215
  d0-14) are identical on Python 3.11, 3.13 and 3.14, and identical with and without `PYTHONHASHSEED=0`; the
  Dandelion job of continuous integration installs `threadfin[dandelion]` on Python 3.12, 3.13 and 3.14.

## Version 4.5.3 — 28 October 2026

* **Why this version.** Every tutorial was followed again in new environments that saw nothing of the testing
  account (no settings, caches or packages; Python packages from PyPI, tutorial files from GitHub). The notebook
  stopped at Step 8: pynndescent 0.7.0, released the day before, returns sparse arrays that scanpy 1.11.5 cannot
  read (`sc.pp.neighbors` on 8,192 or more cells). On the same 60,000 simulated cells and seed, pynndescent 0.7.0
  also returned other approximate neighbours than 0.6.0 (22% of the rows of the 15-neighbour graph identical) and
  another UMAP layout; with pynndescent 0.6.0, numba 0.68.0 and 0.68.1 gave identical results. Threadfin now
  requires pynndescent < 0.7.
* **The release candidate.** A new environment (Python 3.11 from conda-forge only, `pip install
  "threadfin[dandelion]"` from this version: pynndescent 0.6.0, numba 0.68.1, scanpy 1.11.5, sc-dandelion 0.5.7,
  matplotlib 3.8.4) gave clone tables and summaries identical, byte for byte, to those of 4.5.2 for the three
  datasets whose smoothing uses pynndescent (61,423, 72,359 and 115,144 cells; the last also lays out 6,583 clones,
  for which UMAP uses pynndescent too); the tests passed (268, 1 skipped without scirpy);
  every code block of Tutorials 1, 3 and 4 ran (22 of 22); the regenerated notebook ran end to end (7.8 minutes on
  8 cores including the download, 7.6 GB) and printed the same numbers as with 4.5.2.
* **Python 3.14.** `pip install threadfin` installed on Python 3.14 (numpy 2.5, pandas 3.1, anndata 0.13, scanpy
  1.12, matplotlib 3.11); every code block of Tutorials 1, 3 and 4 and `examples/tutorial.py` ran and printed the
  same numbers as on Python 3.11. `pip install "threadfin[dandelion]"` failed there, because matplotlib < 3.9,
  which Dandelion 0.5 needs, has no packages for Python 3.13 or later: the Dandelion extra needs Python 3.10–3.12.

## Version 4.5.2 — 27 October 2026

* **The user's path.** A new environment was built as Tutorial 2 describes (Python 3.11,
  `pip install "threadfin[dandelion]"` from this version: sc-dandelion 0.5.7, numpy 2.4, pandas 2.3, anndata 0.12,
  scanpy 1.11, matplotlib 3.8), and `tf.run` was called on the same 16 datasets: every clone table is identical,
  byte for byte, to that of 4.5.1; the summaries differ only in the sentence about a single programme (3 datasets),
  and the Methods paragraphs only in the version numbers. Every code block of Tutorials 1, 3 and 4 ran (22 of 22).
* **The tutorial notebook** ran end to end in a separately built conda environment: no error, 6.0 minutes on 8
  cores, 7.6 GB; every printed number is the same as with 4.5.1.
* **Tests.** In that environment 268 passed and 1 was skipped (scirpy was not installed). With Python 3.12,
  sc-dandelion 0.5.7 and scirpy 0.25.1, the setting of the new continuous-integration job, all 269 passed and the
  tutorial on simulated data ran.

## Version 4.5.1 — 25 October 2026

* **The user's path, repeated with Dandelion pinned below 1.0.** A new environment was built as Tutorial 2
  describes (Python 3.11, `pip install "threadfin[dandelion]"`, which now installs sc-dandelion 0.5.7 rather than
  1.0; numpy 2.4, pandas 2.3, scipy 1.17, scikit-learn 1.9, anndata 0.12, scanpy 1.11, umap-learn 0.5.12, harmonypy
  2.1, matplotlib 3.8), and `tf.run` was called on the same 16 datasets: every clone table is identical, byte for
  byte, to that of 4.5.0, and the summaries differ only where 4.5.0 wrote "1 cells". Every code block of Tutorials
  1, 3 and 4 ran (22 of 22).
* **The tutorial notebook** ran end to end in a separately built conda environment: no error, 5.5 minutes on 8
  cores, 7.2 GB; 8,736 clones, as with Dandelion 0.5.x before.
* **Tests.** 266 passed; 1 skipped (scirpy was not installed in that environment).
* **Continuous integration** (Linux with Python 3.10–3.14 and with the oldest allowed dependencies, macOS,
  Windows): passed.

## Version 4.5.0 — 24 October 2026

* **The user's path on real data.** A new environment was built as Tutorial 2 describes (Python 3.11,
  `pip install "threadfin[dandelion]"` from this version: numpy 2.4, pandas 2.3, scipy 1.17, scikit-learn 1.9,
  anndata 0.12, scanpy 1.11, umap-learn 0.5.12, harmonypy 2.1, sc-dandelion 0.5.7, matplotlib 3.8), and `tf.run`
  was called on the 14 public BCR datasets of the case studies and the two LARRY datasets with every setting at
  its default, followed by `summary()`, `methods()`, `plot()` and saving: no dataset raised an error, and every
  clone count and clonal ICC equals that of 4.4.2. Every code block of Tutorials 1, 3 and 4 ran (22 of 22)
  ([`tutorial_path`](../manuscript/analyses/technical_validation/tutorial_path/README.md)).
* **The tutorial notebook** (`docs/tutorials/05_end_to_end_tutorial.ipynb`, which downloads GSE287123 from GEO)
  was executed end to end in a separately built conda environment: no error, 5.6 minutes on 8 cores, 7.6 GB.
* **Tests.** The full suite (261 tests) passed with none skipped in one environment that has both optional
  dependencies (Python 3.11, sc-dandelion 0.5.7, scirpy 0.22.5).
* **T cells and scirpy.** scirpy's example T-cell dataset ran end to end from the scirpy MuData
  ([`t_cells`](../manuscript/analyses/technical_validation/t_cells/README.md)).

## Version 4.4.1 — 23 October 2026

Continuous integration now also installs the package and runs the test suite and the tutorial on macOS and
Windows (Python 3.12). Windows found one problem: harmonypy 2.x publishes no Windows wheels and its source does
not compile with Microsoft's compiler, so the installation failed; Windows now receives the pure-Python
harmonypy (< 2). The Harmony step is tested with harmonypy 0.0.9, 0.2.0 and 2.1.0.

## Version 4.4.0 — 23 October 2026

The package was installed from the source tree with `uv` into new, isolated environments (no inherited
`PYTHONPATH` or user site-packages), and the full test suite and the tutorial on simulated data were run in
each (2 tests are skipped when an optional dependency is absent):

| Python | dependency versions | tests | tutorial |
|---|---|---|---|
| 3.10 | the **oldest** versions `pyproject.toml` allows: numpy 1.24.1, pandas 2.0.0, scipy 1.10.0, scikit-learn 1.3.0, anndata 0.10.0, scanpy 1.10.0, matplotlib 3.7.0, umap-learn 0.5.5, pynndescent 0.5.8, igraph 0.10.0, leidenalg 0.10.0, joblib 1.2.0, harmonypy 0.0.9 | 245 passed | passed |
| 3.11 | the benchmark environment (numpy 2.2, pandas 2.3, scanpy 1.11) | 245 passed | passed |
| 3.13 | current releases | 244 passed, 1 expected failure (the legacy v3 repeatability check) | passed |
| 3.14 | current releases: numpy 2.5, pandas 3.1, scipy 1.18, scikit-learn 1.9, anndata 0.13, scanpy 1.12, matplotlib 3.11, numba 0.68 | 244 passed, the same expected failure | passed |

The expected failure was a fault of the test, not of the package, and is fixed since 4.4.0: with pandas 3, `astype(str)` keeps missing values missing, and the test compared the labels of cells without a clone as NaN == NaN. `clonotype_recluster` gives identical clusters on every stack tested (issue #2).

This check found two problems, both fixed in 4.4.0: umap-learn < 0.5.5 and pynndescent < 0.5.8 import
`pkg_resources`, which new environments do not include (the minimum versions were raised), and a
manuscript script used `DataFrame.map`, which pandas 2.0 does not have. Continuous integration now runs
Python 3.10–3.14 and the oldest-versions installation for every change to `main`.

The tutorial notebook (`docs/tutorials/05_end_to_end_tutorial.ipynb`), which downloads GSE287123 from GEO, was
executed end to end in a new conda environment built as the notebook describes, with this version installed.

`n_jobs` (clone-map search and programme bootstrap) gives the same clone map and programmes as one process and
as the released 4.3.1; this is checked in `tests/test_parallel_and_robustness.py` and was checked on GSE287123.

## Version 4.3.0 — 18 October 2026

The package was installed with `pip` from the source tree into a new, isolated Python 3.11 environment
and run outside the repository: `threadfin.run()` completed on a simulated repertoire, including the
automatic choice of the clone-map settings, and the result was written to and read from `.h5ad`. The full
test suite (232 tests) passed in the development environment (Python 3.11). Continuous integration runs
the suite and the tutorial on Python 3.10, 3.11 and 3.12 for every change to `main`.

## Version 4.1.0 — 9 October 2026

The final 4.1.0 wheel was installed into the three isolated Linux environments
and tested from outside the repository. All 25 package modules matched the
source, wheel and installed copies byte-for-byte. This check found and fixed
an older AnnData serialization case in which an absent None-valued context
parameter prevented model reconstruction.

| Python | Installed-wheel full suite | Dependency check | Tutorial | Simulated quick start |
|---|---|---|---|---|
| 3.10.22 | 198 passed | Passed | Passed | Passed |
| 3.11.7 | 198 passed | Passed | Passed | Passed |
| 3.12.15 | 197 passed, 1 expected failure | Passed | Passed | Passed |

The separate current-development-environment suite also passed all 198 tests.
The Python 3.12 expected failure remains the legacy-v3 repeatability check,
not a passing test or a new state-density failure. Scientific warnings were
retained (33–37 per installed suite).
See [the 4.1.0 verification record](compatibility_results_2026-10-09.json)
for hashes, dependencies and every stage exit code.

## Historical check — 6 October 2026

Checked on 6 October 2026 using clean Linux environments without inherited
system-site packages. Tests imported the installed wheel from `site-packages`
outside the repository. All 23 package Python modules were also
compared byte-for-byte with the wheel and all three installed copies.

| Python | Full test suite | Dependency check | Tutorial | Simulated quick start |
|---|---|---|---|---|
| 3.10.22 | 161 passed | Passed | Passed | Passed |
| 3.11.7 | 161 passed | Passed | Passed | Passed |
| 3.12.15 | 160 passed, 1 expected failure | Passed | Passed | Passed |

The Python 3.12 expected failure is the existing legacy-v3
`clonotype_recluster` repeatability check. It is not counted as a passing test.
The v4 `tf.run` tests and examples passed on all three environments. The suites
emit 33–37 warnings, including scientific warnings about unstable programme
partitions; those are retained rather than hidden.

All 17 reclustering-control tests passed. They cover explicit overrides,
unchanged default partitions, independent display controls, AnnData round-trip
serialization, missing clone IDs, invalid parameter/distance rejection and
exact agreement with the historical Scanpy graph/Leiden recipe, including
independent UMAP and graph/Leiden random seeds. The optional
embedding graph is tested alongside unchanged default distance partitions.

Exact dependency versions, the tested wheel's SHA-256 and stage exit codes are
in [the verification record](compatibility_results_2026-10-06.json). The same
tests, optional sequence dependencies and simulated quick start are covered
by the repository's Python 3.10/3.11/3.12 GitHub Actions matrix.

## User-facing errors

Input validation now checks requested metadata columns before analysis,
rejects missing or blank donor/sample/batch labels, reports duplicate or
unmatched BCR barcodes and unusable heavy-chain calls, and checks finite
cell-by-feature embeddings and non-negative counts. Requested test columns
are not silently skipped. PCA dimensions are capped to the selected matrix
size. Errors are in English and include concrete correction steps.

See the [troubleshooting table of Tutorial 2](tutorials/02_your_data.md#9-troubleshooting) for examples. Missing measured
state, probe or time labels remain missing observations; they are not filled
as negative measurements.

## Verification limits

This verifies the tested dependency stacks (Linux for the real-data runs; Linux, macOS and Windows in
continuous integration since 4.4.1) and the documented example paths, not every real-data configuration.
Record package versions alongside scientific outputs; since 4.6.0, `threadfin.run` records them in
`adata.uns['threadfin']['run']`.
