# Case studies on public data

Paired single-cell RNA and BCR sequencing datasets analysed with Threadfin. Each case study asks a
biological question of the clones in one dataset and answers it with the same pipeline, so the datasets
can be compared. The datasets marked † are analysed in the manuscript; the others are additional
applications. A summary of every dataset — cells, clones, clonal coherence, programmes and where it appears in
the manuscript — is in [`../dataset_overview.md`](../dataset_overview.md).

**New to Threadfin?** Read [Tutorial 1](../../docs/tutorials/01_getting_started.md) first; the
[end-to-end notebook](../../docs/tutorials/05_end_to_end_tutorial.ipynb) runs one of these case studies
(`mouse_rbd`) from the GEO download onwards.

| Case study | Data | Question |
|---|---|---|
| `gut_gc_reporter` † | Siniscalco *et al.* 2025 *Immunity* (GSE284622): oral NP-OVA or OVA plus cholera toxin; mesenteric lymph node, Peyer's patches and lamina propria; GC-fate reporter mice | Which clones connect gut germinal centres with IgA plasma cells, and where do clones without GC-fate labelling sit? |
| `spont_gc` † | Castrillon *et al.* 2023 *eLife* (GSE203132): GC-experienced (AID-EYFP⁺) splenic B cells from mice with spontaneous germinal centres and from immunized controls | Do clones differ in how their cells are distributed among GC, antibody-secreting and memory states? |
| `malaria_late` † | Skinner *et al.* 2026 *Nat. Immunol.* (GSE286215, experiment 2): spleen, days 10–42 of *Plasmodium chabaudi* infection | Are plasmablasts sampled together with GC B cells of the same clone, and does that vary between clones? |
| `malaria` † | GSE286215, experiment 1: spleen at days 0–14 of *P. chabaudi* infection | How does clonal structure develop through the first two weeks of infection? |
| `mouse_rbd` † | Merkenschlager *et al.* 2025 *Nature* (GSE287123): RBD protein or mRNA immunization, GC B cells sorted by division reporter, RBD bait and dark/light zone | Are division history, antigen binding and zone written into clone profiles? |
| `mouse_np` † | Merkenschlager *et al.* 2025 (GSE287123): NP-OVA immunization, H2B-mCherry division reporter | Is the number of divisions a clone has made reflected in its state? |
| `gc_np_pc` † | ElTanbouly *et al.* 2024 *J. Exp. Med.* (GSE246382): NP-OVA day 14, sorted GC zones and plasma cells (Smart-seq2) | Do clones that produce plasma cells differ from clones that stay in the germinal centre? |
| `np_cgg_wt_d14` † | Manakkat Vijay *et al.* 2026 *Nat. Immunol.* (GSE305877): NP-CGG immunization, day 14, one wild-type mouse (analysed by `../figures/reserve/source/analyse_np_cgg_wt_d14.py`) | Which clones keep GC gene expression? |
| `ln_vaccine` † | Kim *et al.* 2022 *Nature* (GSE195673): human lymph node fine-needle aspirates and blood after SARS-CoV-2 mRNA vaccination | Do GC clones commit to plasma-cell or memory output, and do spike-binding clones differ? |
| `covid_bcell` | Dugan *et al.* 2021 *Immunity* (GSE171703): antigen-bait-sorted B cells after SARS-CoV-2 infection | Is antigen specificity visible on the clone map? |
| `flu` | Wang *et al.* 2023 (GSE175522/GSE175523): blood before and 7 days after influenza vaccination | Which clones are recruited into the day-7 plasmablast response? (heavy chain only: no light chain was deposited, so its clones are not comparable with the paired-chain datasets) |
| `flu_lung` | GSE317692: mouse lung and mediastinal lymph node, day 20 after influenza A infection, with haemagglutinin tetramers | Do lung clones, and clones that bind the infecting or both haemagglutinins, occupy different states? |
| `tonsil` | King *et al.* 2021 *Sci. Immunol.* (E-MTAB-9005/9003): paediatric tonsil | Do isotype-switched clones occupy different states from IgM/IgD clones? |
| `ebv` | Mitul *et al.* 2026 *PNAS* (GSE317492): tonsil organoids infected with Epstein–Barr virus | Does infection select particular clones? (exploratory: donors are pooled) |
| `bone_marrow_pc` | GSE253857: human bone-marrow plasma cells and memory B cells, with blood; some sorted on SARS-CoV-2 spike or tetanus toxoid | Do clones keep a shared state across bone marrow and blood, and does the antigen matter? |

## How the case studies are run

Each dataset goes through four stages. The first two prepare the inputs once; the last two are the analysis.

1. **Download** the public files (`download.sh`; sources and MD5 checksums in `datasets_manifest.tsv` and
   `md5_manifest.txt`). `datasets.py` holds one *loader* per dataset that turns the deposited files into an
   AnnData of raw counts plus a per-cell BCR table, with the donor, sample and every annotation the authors
   recorded.
2. **Define clones** with Dandelion's paired heavy- and light-chain definition, within each donor
   (`dandelion_clones.py`, writing `results/dandelion_clones/<dataset>.csv`). This is the same clone
   definition `threadfin.run` uses; it is precomputed here so that the analysis does not need Dandelion.
   Dandelion also groups cells that have no heavy chain by their light chain alone (ids `B_VJ_…`); these
   groups join unrelated cells and are not clones, so `run_case_study.py` leaves such cells without a clone.
   (Until 2026-10-20 they were kept for the four datasets whose loaders read 10x per-cell tables —
   `covid_bcell`, `tonsil`, `ebv`, `bone_marrow_pc` — whose results were then re-run; in `covid_bcell` they had
   made up 1,689 of 2,281 "clones" with two or more cells.)
3. **Run Threadfin** with the same steps on every dataset (`run_case_study.py <dataset>`): clone profiles
   relative to each donor; clonal coherence against clone labels shuffled within samples; the clone map;
   clonal programmes with split tests and bootstrap stability; clone-level tests of the measured labels
   (antigen binding, isotype, mutation load, sort gate, tissue, time point); clonal memory; and clonally
   inherited genes. Dataset-specific settings (the sampling context, labels to test, fixed map settings used
   for the figures) are in `CONFIGS` at the top of `run_case_study.py`.
4. **Ask the dataset-specific questions** with the scripts in the table below.

```bash
bash download.sh /path/to/data          # sources and checksums in datasets_manifest.tsv / md5_manifest.txt
export THREADFIN_DATA=/path/to/data
python dandelion_clones.py ln_vaccine   # paired heavy + light clones with Dandelion
python run_case_study.py ln_vaccine     # one dataset -> results/ln_vaccine/
python summarize.py                     # cross-dataset tables -> results/
```

> **Why the donor is the sampling context here.** Every clone is described relative to the other cells of
> its donor. Tissue, time point and sort gate are *not* used as contexts because they are what the
> case studies compare between clones (see [Tutorial 2](../../docs/tutorials/02_your_data.md#5-choosing-donor_key-sample_key-and-batch_key)).

| Script | Analysis |
|---|---|
| `gut_plasma_branches.py` | the plasma-cell branches of the gut clone map: isotype, tissue and GC-fate labelling |
| `spont_gc_fates.py`, `p3_subsplit_analysis.py` | state diversity of spontaneous-GC clones and the split of one programme |
| `malaria_timecourse.py`, `pcas_arm_analyses.py` | clonal structure through the *Plasmodium* infection; convergent receptors shared between mice and the arms of the clone map |
| `clone_state_sharing.py` | co-observation of states within clones in the *Plasmodium* data |
| `mouse_gc_deep_dive.py` | division history, antigen binding and zone in the model-antigen germinal centres, with library controls |
| `spike_gc_trends.py`, `spike_binding_audit.py` | spike binding and somatic hypermutation in the vaccine lymph node, and what the public spike label does and does not measure |
| `covid_bcell_antigen.py` | antigen specificity on the clone map |
| `lineage_trees.py`, `lineage_vs_state.py`, `run_gctree.sh` | somatic-hypermutation lineage trees (GCtree) and state along the tree |
| `sequence_selection.py` | selection on the antibody sequence (replacement versus silent mutations against each clone's own germline), measured independently of expression |
| `summarize.py`, `summarize_clonal_information.py` | cross-dataset tables |

The analyses that go beyond Figures 1–4 — and what each of them found — are explained in
[`../analyses/`](../analyses/README.md).

## Output per dataset (`results/<dataset>/`)

| File | Content | How to use it |
|---|---|---|
| `summary.json` | all key numbers of the run | the first file to open; coherence, programmes, tests and settings |
| `clone_table.csv` | one row per clone: size, profile reliability, programme, labels, map coordinates | plot the clone map yourself, or join clone-level labels |
| `cells.csv.gz` | per-cell clone, programme, labels and UMAP coordinates | project programmes onto cells |
| `programmes.csv`, `programme_markers.csv`, `programme_composition.csv` | programmes, their marker genes (clones as replicates) and cell-state composition | describe each programme |
| `label_effects.csv` | how much of the difference between clones each label explains (R², permutation null, P) | which annotations line up with clone profiles |
| `programme_associations.csv` | which programmes are enriched for each label | only meaningful for stable programmes |
| `memory_pairs_<key>.csv` | each clone's change of state between time points, tissues or gates | the clonal-memory analysis |
| `gene_heritability.csv`, `geneset_heritability.csv` | clonal inheritance of every gene and of marker gene sets | which genes are shared within clones |
| `figures/` | coherence, clone map, label effects, memory, heritability | quick visual check of each run |

`results/dandelion_clones/` holds the clone assignment of every cell in every dataset.

**Reading `label_effects.csv`.** `r2` is the share of variation between clone profiles explained by the
label, `null_mean` the share explained by labels shuffled among clones of the same donor, `excess_r2` their
difference and `p_value` the permutation P value. Labels derived from the transcriptome (author cell types)
are not independent of the profiles; labels that are constant within a donor are tested without
stratification and cannot be separated from other donor differences.
