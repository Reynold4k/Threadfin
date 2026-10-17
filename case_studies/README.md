# Case studies on public data

Paired single-cell RNA and BCR sequencing datasets analysed with Threadfin. Each case study asks a
biological question of the clones in one dataset. The datasets marked † are analysed in the manuscript;
the others are additional applications.

| Case study | Data | Question |
|---|---|---|
| `gut_gc_reporter` † | Siniscalco *et al.* 2025 *Immunity* (GSE284622): oral NP-OVA or OVA plus cholera toxin; mesenteric lymph node, Peyer's patches and lamina propria; GC-fate reporter mice | Which clones connect gut germinal centres with IgA plasma cells, and where do clones without GC-fate labelling sit? |
| `spont_gc` † | Castrillon *et al.* 2023 *eLife* (GSE203132): GC-experienced (AID-EYFP⁺) splenic B cells from mice with spontaneous germinal centres and from immunized controls | Do clones differ in how their cells are distributed among GC, antibody-secreting and memory states? |
| `malaria_late` † | Skinner *et al.* 2026 *Nat. Immunol.* (GSE286215, experiment 2): spleen, days 10–42 of *Plasmodium chabaudi* infection | Are plasmablasts sampled together with GC B cells of the same clone, and does that vary between clones? |
| `malaria` | GSE286215, experiment 1: spleen at days 0–14 of *P. chabaudi* infection | How does clonal structure develop through the first two weeks of infection? |
| `mouse_rbd` † | Merkenschlager *et al.* 2025 *Nature* (GSE287123): RBD protein or mRNA immunization, GC B cells sorted by division reporter, RBD bait and dark/light zone | Are division history, antigen binding and zone written into clone profiles? |
| `mouse_np` † | Merkenschlager *et al.* 2025 (GSE287123): NP-OVA immunization, H2B-mCherry division reporter | Is the number of divisions a clone has made reflected in its state? |
| `gc_np_pc` † | ElTanbouly *et al.* 2023 *J. Exp. Med.* (GSE246382): NP-OVA day 14, sorted GC zones and plasma cells (Smart-seq2) | Do clones that produce plasma cells differ from clones that stay in the germinal centre? |
| `np_cgg_wt_d14` † | Manakkat Vijay *et al.* 2026 *Nat. Immunol.* (GSE305877): NP-CGG immunization, day 14, one wild-type mouse (analysed by `paper/figure_plan/reserve/source/analyse_np_cgg_wt_d14.py`) | Which clones keep GC gene expression? |
| `ln_vaccine` † | Kim *et al.* 2022 *Nature* (GSE195673): human lymph node fine-needle aspirates and blood after SARS-CoV-2 mRNA vaccination | Do GC clones commit to plasma-cell or memory output, and do spike-binding clones differ? |
| `covid_bcell` | Dugan *et al.* 2021 *Immunity* (GSE171703): antigen-bait-sorted B cells after SARS-CoV-2 infection | Is antigen specificity visible on the clone map? |
| `flu` | Wang *et al.* 2023 (GSE175522/GSE175523): blood before and 7 days after influenza vaccination | Which clones are recruited into the day-7 plasmablast response? |
| `flu_lung` | GSE317692: mouse lung and mediastinal lymph node, day 20 after influenza A infection, with haemagglutinin tetramers | Do lung clones, and clones that bind the infecting or both haemagglutinins, occupy different states? |
| `tonsil` | King *et al.* 2021 *Sci. Immunol.* (E-MTAB-9005/9003): paediatric tonsil | Do isotype-switched clones occupy different states from IgM/IgD clones? |
| `ebv` | Mitul *et al.* 2026 *PNAS* (GSE317492): tonsil organoids infected with Epstein–Barr virus | Does infection select particular clones? (exploratory: donors are pooled) |
| `bone_marrow_pc` | GSE253857: human bone-marrow plasma cells and memory B cells, with blood; some sorted on SARS-CoV-2 spike or tetanus toxoid | Do clones keep a shared state across bone marrow and blood, and does the antigen matter? |

## Running

```bash
bash download.sh /path/to/data          # sources and checksums in datasets_manifest.tsv / md5_manifest.txt
export THREADFIN_DATA=/path/to/data
python dandelion_clones.py ln_vaccine   # paired heavy + light clones with Dandelion
python run_case_study.py ln_vaccine     # one dataset -> results/ln_vaccine/
python summarize.py                     # cross-dataset tables -> results/
```

`run_case_study.py` runs the same steps on every dataset: clones defined within donors with Dandelion's
paired heavy- and light-chain definition; clone profiles relative to each sample; clonal coherence against
clone labels shuffled within samples; the clone map; clonal programmes with split tests and bootstrap
stability; clone-level tests of the measured labels (antigen binding, isotype, mutation load, sort gate,
tissue, time point); clonal memory; and clonally inherited genes. Dataset-specific analyses are in their
own scripts:

| Script | Analysis |
|---|---|
| `gut_plasma_branches.py` | the plasma-cell branches of the gut clone map: isotype, tissue and GC-fate labelling |
| `spont_gc_fates.py`, `p3_subsplit_analysis.py` | state diversity of spontaneous-GC clones and the split of one programme |
| `malaria_timecourse.py`, `pcas_arm_analyses.py` | clonal structure through the *Plasmodium* infection |
| `mouse_gc_deep_dive.py` | division history, antigen binding and zone in the model-antigen germinal centres |
| `spike_gc_trends.py`, `spike_binding_audit.py` | spike binding and somatic hypermutation in the vaccine lymph node |
| `covid_bcell_antigen.py` | antigen specificity on the clone map |
| `clone_state_sharing.py` | co-observation of clones and states in the *Plasmodium* data |
| `lineage_trees.py`, `lineage_vs_state.py`, `run_gctree.sh` | somatic-hypermutation lineage trees (GCtree) and state along the tree |
| `sequence_selection.py` | selection on the antibody sequence, measured independently of expression |

The clone maps of the published figures were drawn with fixed settings recorded in `run_case_study.py`
(`map_params`). Without them, `embed_clones` chooses the settings for each dataset automatically.

## Output per dataset (`results/<dataset>/`)

| File | Content |
|---|---|
| `summary.json` | all key numbers of the run |
| `clone_table.csv` | one row per clone: size, profile reliability, programme, labels, map coordinates |
| `programmes.csv`, `programme_markers.csv`, `programme_composition.csv` | programmes, their marker genes (clones as replicates) and cell-state composition |
| `label_effects.csv` | how much of the difference between clones each label explains |
| `programme_associations.csv` | which programmes are enriched for each label |
| `memory_pairs_<key>.csv` | each clone's change of state between time points, tissues or gates |
| `gene_heritability.csv`, `geneset_heritability.csv` | clonal inheritance of every gene and of marker gene sets |
| `cells.csv.gz` | per-cell clone, programme, labels and UMAP coordinates |
| `figures/` | coherence, clone map, label effects, memory, heritability |

`results/dandelion_clones/` holds the clone assignment of every cell in every dataset.
