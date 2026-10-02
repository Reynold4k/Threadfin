# Case studies on public data

Seven published paired single-cell RNA + BCR datasets, re-analysed with
Threadfin. Each one asks a biological question about B-cell clones that the
original study could not ask at clone level. The step-by-step interpretation,
written for readers without a computational background, is in
[`../report/PUBLIC_DATASETS_REPORT.md`](../report/PUBLIC_DATASETS_REPORT.md).

| case study | data | biological question |
|---|---|---|
| `ln_vaccine` | human lymph node and blood after SARS-CoV-2 mRNA vaccination (Kim et al. 2022, *Nature*) | Do germinal-centre clones commit to plasma-cell or memory fates, and do spike-binding clones differ? Do clones keep their state for months? |
| `mouse_np` | mouse germinal centres with a cell-division reporter, NP-OVA immunisation (Merkenschlager et al. 2025, *Nature*) | Is the number of divisions a clone has made written into its transcriptional state? |
| `mouse_rbd` | mouse germinal centres sorted by division, antigen binding and dark/light zone (Merkenschlager et al. 2025) | Are dark- and light-zone states clonally fixed, or do all clones cycle between them? |
| `flu` | human blood before and 7 days after influenza vaccination (Wang et al. 2023) | Which clones are recruited into the day-7 plasmablast burst? |
| `ebv` | human tonsil organoids infected with Epstein-Barr virus (Mitul et al. 2026, *PNAS*) | Does EBV infection select particular clones? (exploratory: pooled donors) |
| `tonsil` | human paediatric tonsil (King et al. 2021, *Sci Immunol*) | Do isotype-switched clones occupy different states from IgM/IgD clones? |
| `stephenson` | human blood in COVID-19 (Stephenson et al. 2021, *Nat Med*) | Do clone states differ with disease severity? (donor-level comparison) |

## Running

```bash
bash download.sh /path/to/data          # ~12 GB; sources in datasets_manifest.tsv
export THREADFIN_DATA=/path/to/data
python run_case_study.py ln_vaccine     # one dataset -> results/ln_vaccine/
python mouse_gc_deep_dive.py            # extra controls for the two mouse studies
python summarize.py                     # cross-dataset tables -> results/
```

Every case study runs the same five steps (`run_case_study.py`):

1. **Clones** are defined from heavy-chain sequences within each donor.
2. **Does clone identity shape cell state?** Share of transcriptional
   variation explained by clone, against clones shuffled within samples.
3. **Which clones behave alike?** Clone programmes, or "continuum" when
   clones differ without forming distinct groups.
4. **Do measured labels explain how clones differ?** Antigen binding,
   isotype, mutation load, sort gate, tissue or time point, tested with
   clones (not cells) as the unit.
5. **Do clones keep their state, and which genes are clonally inherited?**
   Clonal memory across time points, tissues or gates; gene-level clonal
   heritability.

## Output per dataset (`results/<dataset>/`)

| file | content |
|---|---|
| `summary.json` | all key numbers of the run |
| `clone_table.csv` | one row per clone: size, profile reliability, programme, labels |
| `programmes.csv`, `programme_markers.csv`, `programme_composition.csv` | programmes, their marker genes (clones as replicates) and cell-state make-up |
| `label_effects.csv` | how much of the difference between clones each label explains |
| `programme_associations.csv` | which programmes are enriched for each label |
| `memory_pairs_<key>.csv` | each clone's state change between time points / tissues / gates |
| `gene_heritability.csv`, `geneset_heritability.csv` | clonal inheritance of every gene and of marker gene sets |
| `cells.csv.gz` | per-cell clone, programme, labels and UMAP coordinates |
| `figures/` | coherence, clone map, label effects, memory, heritability |
