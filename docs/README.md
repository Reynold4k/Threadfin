# Threadfin documentation

| Page | Content |
|---|---|
| [Tutorial 1 — Getting started](tutorials/01_getting_started.md) | the whole analysis on simulated data, with its output |
| [Tutorial 2 — Your own data](tutorials/02_your_data.md) | from Cell Ranger output to results, with several samples and donors |
| [Tutorial 3 — The clone map](tutorials/03_clone_map.md) | automatic and hand-set layouts, and how to check a map |
| [Tutorial 4 — Step by step](tutorials/04_step_by_step.md) | the individual functions in `threadfin.tl` |
| [Tutorial 5 — From GEO to results](tutorials/05_end_to_end_tutorial.ipynb) | a Jupyter notebook, tested end to end in a fresh environment: environment ([`environment.yml`](tutorials/environment.yml)), download of GSE287123, hashtag demultiplexing, clones, embedding, `threadfin.run`, every output explained (cells are generated from [`make_tutorial_notebook.py`](tutorials/make_tutorial_notebook.py)) |
| [Methods](METHODS.md) | the statistical model behind each step, with references |
| [Function reference](API.md) | every public function and its arguments, generated from the docstrings |
| [State densities](STATE_DENSITY.md) | the optional frozen-reference region-count model |
| [Reclustering (v3)](RECLUSTERING.md) | the legacy clonotype reclustering presets, kept for compatibility |
| [Compatibility](COMPATIBILITY.md) | Python versions (3.10–3.14) and dependency versions on which the package was installed and tested |
| [The manuscript](../manuscript/README.md) | publication status, datasets, analyses, the benchmark against related tools |

**Which tutorial first?** New to Threadfin: Tutorial 1 (a minute, simulated data), then Tutorial 5 (real data,
end to end). Ready to analyse your own experiment: Tutorial 2, with Tutorial 3 for the clone map. Building a
custom pipeline: Tutorial 4 and Methods.

Every function also has a docstring (`help(threadfin.tl.embed_clones)`), collected in the [function reference](API.md).
