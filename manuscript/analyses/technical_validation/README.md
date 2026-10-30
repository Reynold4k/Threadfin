# Technical validation

Checks of the method itself rather than of biology: how the clone map's settings are chosen, how well the
optional state-density model recovers known distributions, how fast each representation is, and a few
input audits. Until 2026-10-20 this material was kept outside the repository; it is published here so that
every analysis behind the manuscript can be inspected. The systematic comparison with other tools is in
[`../../benchmarking/`](../../benchmarking/README.md).

Most scripts were written to run from the repository's former `case_studies/` folder or from a local
validation workspace; paths at the top of each script record where their inputs were. They are provided as a
record of exactly what was run.

## Questions a user asks first (2026-10-24)

| Folder | Question | Answer |
|---|---|---|
| [`tutorial_path/`](tutorial_path/README.md) | Does the documented path work on real data, as a new user would follow it? | `tf.run` on all 14 public BCR datasets and the two LARRY datasets, in a new environment, with every setting at its default: no errors; every code block of Tutorials 1, 3 and 4 runs; every figure in a [gallery](tutorial_path/figures/README.md). |
| [`clone_definition/`](clone_definition/README.md) | Do the results depend on how clones are called? | Junction identities 0.75–1.0 and Dandelion's contig check, on two case studies: every conclusion holds. |
| [`embedding/`](embedding/README.md) | Do the results depend on how the expression embedding is built? | No: 2,000–5,000 variable genes and 20–50 components leave the excess clonal ICC within 0.115–0.126 and every label effect significant; leaving the receptor genes in would inflate the RBD coherence (excess 0.118 → 0.176). |
| [`power/`](power/README.md) | How many clones does the coherence test need? | Calibrated; 80 % power for a clonal ICC of 0.1 with about 10 clones of two or more cells, 0.05 with about 40, 0.02 with 160–320. |
| [`scalability/`](scalability/README.md) | Does it scale to atlas-size data? | A million cells (71,376 clones of two or more cells) in about an hour on 8 cores and 8.3 GB; time grows about in proportion to the number of cells. |
| [`t_cells/`](t_cells/README.md) | Does it work for T cells, from a scirpy object? | Yes, with the same call: 3,000 T cells of 14 patients (Wu et al. 2020) in one minute; clone identity explains 26.5 % of state variation (3.0 % shuffled). |

## Clone-map settings chosen automatically (`clone_map_settings/`, 2026-10-18)

**Question.** UMAP draws a continuum of clones as one picture only for suitable settings
(`n_neighbors`, `min_dist`), and the suitable settings differ between datasets. Can they be chosen
automatically, without any annotation?

**How the method was arrived at** (each step forced by a failure of the previous one):

1. *Scoring* (`scripts/grid_eval.py`, `analyze_grid.py`): 10 datasets × 54 settings × 2 seeds. Of several
   objectives, the one kept is the mean of local fidelity (preservation of each clone's 30 nearest
   neighbours) and global fidelity (Spearman correlation of distances), minus 5 × the shortfall of coherence
   (fraction of clones in one connected piece) below a target given by the profiles themselves (the largest
   component of their mutual 15-nearest-neighbour graph, capped at 0.99). This objective ranked the settings
   the authors had chosen by hand at or near the top.
2. *Gradient search* (Adam with simultaneous-perturbation gradients) found reasonable regions but wandered on
   flat optima and missed distant coherent regions; it is kept as an option (`method="adam"`).
3. *Robustness* (`validate.py`, `validate2.py`): settings that were most faithful on one layout often broke
   apart on another seed (one hand-picked setting kept only 57 % of clones connected on a new seed). Settings
   with `min_dist` ≥ 0.5 were much more robust, and 200 optimisation epochs scored layouts almost exactly like
   the default number while running 2.4× faster (`epochs_test.py`).
4. *Final method* (`tune_all.py`): a ladder of 6 × 4 settings, two shared seeds each, worst-case coherence,
   Gaussian smoothing over neighbouring settings, confirmation on six layouts, weaker UMAP repulsion
   (0.5, then 0.25) when nothing holds together (`repulsion_test.py`), a final check of the delivered map, and
   a subsample search for more than 3,000 clones (`validate4.py` → `validate5.py`, needed for the
   11,141-clone human lymph-node dataset).

**Result** (`final_results.csv`, `gallery_v3.png`).

| Method | Runs | Delivered map met the coherence target | Median time |
|---|---|---|---|
| automatic (ladder; 3 seeds × 10 datasets) | 30 | 30 / 30 | 54 s |
| previous fixed default (`n_neighbors=50, min_dist=0.8`) | 10 | 9 / 10 | 2 s |
| settings chosen by hand for the figures | 10 | 6 / 10 | 2 s |

Automatic choices were close to the hand-chosen ones (gut: 32/0.55 in two of three seeds, against 30/0.3
by hand). Datasets with more than 10,000 clones need 3–7 minutes. This is the procedure of
`threadfin.tl.tune_clone_map` in version 4.3.0 ([Methods](../../../docs/METHODS.md#5-the-clone-map-embed_clones-tune_clone_map)).

## The earlier parameter sweep (`clone_map_sweep_2026-10-11/`)

`clone_map_sweep.py` laid every dataset out over a grid of settings (galleries `*_gallery.png`);
`choose_map_params.py` chose the fixed default of version 4.2 from it. Superseded by the automatic choice
above; the 324 saved layouts (36 MB) are not stored.

## Optional state-density model (`state_density_model/`)

`threadfin.StateDensityModel` ([STATE_DENSITY.md](../../../docs/STATE_DENSITY.md)) describes a clone by
counts of its cells in regions of a frozen reference, with a conjugate prior. It is an optional alternative,
not the default representation, and these checks do not validate the default kernel profiles.

* **Simulation** (`validate_density_model.py`; `density_simulation*.csv/json`): 200 reference clones,
  500 evaluation clones per regime, query sizes 1–100 cells. Simultaneous sampling intervals covered at
  least 97.8 % of true region fractions; marginal credible intervals can under-cover when the prior is
  misspecified (minimum 0 % in boundary regimes), which is why they are documented as conditional.
* **LARRY split-cell reconstruction** (`density_larry_*`): 250 evaluation profiles (216 barcodes) whose
  region fractions are estimated from a few cells and compared with an independent half of the same clone.
  Mean squared error at 2 / 4 / 8 cells: raw counts 0.0169 / 0.0098 / 0.0062; with the reference-calibrated
  prior 0.0118 / 0.0082 / 0.0058 — the prior helps most for the smallest clones (−30 % at 2 cells).
* **Prior strength** (`prior_calibration.csv`): the out-of-sample log score peaks at strength 1–2, the
  automatic choice.
* **Runtime** (`benchmark_representation_runtime.py`; `representation_runtime.csv`): for 50,000 cells in
  5,000 clones, mean profiles take 0.06 s, kernel profiles 6.1 s and density profiles 8.3 s (one CPU core;
  peak memory 0.3–0.6 GB). End-to-end runtimes of every tool are in the benchmark.
* The algorithm-revision tools (`integrate_algorithm_revision.py`, `rerun_algorithm_revision.py`,
  `validate_paper_revision.py`, `validate_density_reporter.py`) record how the density model was integrated
  and checked against the division-reporter data.

## Bone-marrow sort gates (`bone_marrow_gate_check/`)

GSE253857 sorted bone-marrow plasma cells and memory B cells and blood memory B cells from the same donors.
`bone_marrow_pure_gate_check.py` counts receptors shared between sorted populations of one donor, with
exact heavy-chain and exact heavy + light definitions (`donor_validation.csv`; e.g. donor 1681: 225 receptor
groups shared between bone-marrow plasma cells and bone-marrow memory B cells under the paired definition).
It confirmed that clones genuinely span the sorted compartments, so the dataset's clone structure is not an
artefact of one gate.

## First attempt to run related tools natively (`native_method_adapters_2026-10-05/`)

Adapters that exported the documented inputs of Benisse and BiGCN for the RBD dataset, ran their published
entry points and scored the outputs with a donor-held-out ridge readout. BiGCN did not finish within the
allocated time and the attempt was set aside; its readout was tied to an older clone definition. It is
superseded by the end-to-end [benchmark](../../benchmarking/README.md), which runs Benisse, Clonotrace,
clone2vec and others on current clone definitions.

## Other checks (`filter_comparison/`)

`filter_comparison.py` compared clone-filtering rules (it motivated removing the reliability filter);
`review_figure2_biology.py` audited which datasets and labels Figure 2 could support; `run_gc_reclustering.py`
reproduced the legacy version-3 reclustering presets ([RECLUSTERING.md](../../../docs/RECLUSTERING.md)).
