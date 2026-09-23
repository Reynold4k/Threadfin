# Threadfin multi-dataset biological validation

Standardized validation of Threadfin across independent public paired
scRNA-seq + scBCR-seq datasets. Each dataset is described by a JSON config in
`configs/`; running one command produces QC, parameter-grid runs, basis
robustness, null models, held-out validation and figures.

## Datasets

| name | biology | external (non-transcriptomic) labels |
|---|---|---|
| `stephenson2021` | COVID-19 PBMC B cells | author cell-type labels |
| `flu_gse175522` | influenza vaccination, young vs older adults | vaccination timepoint (pre/d7), age group |
| `ebv_organoid_gse317492` | primary EBV infection in tonsil organoids | **GFP± infection status**, timecourse d0–d21 |
| `tonsil_king2021` | human tonsil GC B-cell maturation | author cell types (incl. DZ GC) |
| `ln_vaccine_gse195673` | SARS-CoV-2 mRNA vaccine lymph-node GC | tissue (LN vs blood), timepoint, author labels |

Per-dataset sources, URLs and checksums: see `../datasets_manifest.tsv`.

## Run

```bash
VENV=/data/scratch/projects/punim1236/threadfin_data/venv/bin/python  # any env with threadfin+scanpy
cd benchmarks/biological_validation
PYTHONPATH=. $VENV run_validation.py configs/<dataset>.json
```

Heavy datasets (EBV, LN vaccine) should go through SLURM, see `run_all.sbatch`.

## What each output answers

| output | question |
|---|---|
| `qc.json` | is the dataset even suited for clonotype-level analysis (enough expanded clones)? |
| `runs.json` | how do communities change with basis / resolution / sequence blending? |
| `robustness.json` | do PCA and UMAP bases agree (clone-level ARI)? |
| `null_permutation.json` | is clone state-purity above the within-donor permutation null? |
| `null_random_communities.json` | do size-matched random communities produce as many "significant" enrichments? |
| `heldout.json` | do held-out cells of a clone reproduce its community assignment? |
| `figures/` | cell state map, clone map, community map-back, signature heatmap, null histogram, basis comparison |
| `report.md` | human-readable summary |

## Null models — what each preserves

* **Null 1 (clone-label permutation)**: shuffles clone ids among cells *within
  donor*; preserves clone-size distribution, donor composition and state
  composition; destroys the clone→state link.
* **Null 3 (random communities)**: permutes community labels across *clones*;
  preserves community sizes and the clone→state composition; destroys the
  community→state link.
* **Null 2 (sequence-only grouping)** is realised via the parameter grid
  (`cdr3_weight = 1.0` runs in `runs.json`): communities built from CDR3
  distance only, compared against expression-based communities.
