# Negative results, withdrawn claims and corrected defects

A record of what was tested during the development of the manuscript and did **not** hold, and of the
defects found in our own code and estimators. It is kept because every one of these shaped a decision in
the current manuscript, and because a reader re-analysing the data should not have to rediscover them.

Numbers marked *(historical)* were computed between 2026-10-02 and 2026-10-13, several of them with the
earlier heavy-chain-only clone definition that was replaced by Dandelion's paired heavy- and light-chain
definition on 2026-10-11; they describe why a claim was dropped and are not current results. Numbers without
the mark come from the current analyses in this folder or in [`../case_studies/results/`](../case_studies/results).

---

## 1. Claims that were withdrawn

### 1.1 "Germinal centres form a continuum; mixed compartments form discrete groups"
*Withdrawn on 2026-10-04 after auditing every dataset's split tests.* The dichotomy was false: the
*Plasmodium* infection (days 10–42) and the human lymph node after vaccination both contain GC B cells and
formed groups, whereas the tonsil (6 % plasmablasts) formed a continuum. What holds is narrower: sorted GC
B cells alone gave one programme in two experiments, and the programme split test is conservative
(section 2.6), so a single programme is not evidence of a continuum. The manuscript reports programme
*stability* and does not use the number of programmes as a finding.

### 1.2 "Cell state is not clonal: it collapses when features and labels come from different cells"
*Withdrawn on 2026-10-10, the day it was written.* A disjoint-cell test (features from half of each clone's
cells, labels from the other half) had been applied to whatever categorical columns each GEO submission
contained. The "collapse" rested on three labels — two division gates (a 36-h H2B–mCherry reporter) and one
zone gate — none of them a clone-level fate; the two *Plasmodium* datasets, which have the clearest
GC/plasmablast/memory structure, contributed no cell-state label at all. Tested properly, the composition
of a clone over the GC, plasmablast and memory compartments was reproducible across disjoint halves in four
datasets and two systems (disjoint-half reliabilities 0.06–0.62 depending on dataset and compartment *(historical)*). The corrected statement:
which compartment a clone's cells occupy is clone-determined; cell-cycle position and light-/dark-zone
location are re-decided within a clone faster than one somatic mutation accumulates.

### 1.3 "Clonal dominance predicts clonal coherence" (frozen prediction P6)
*Falsified on 2026-10-10 against a prediction frozen beforehand.* Across ten datasets, Spearman's ρ between the
Gini coefficient of clone sizes and the reproducibility of the GC arm was −0.42 (P = 0.23; dataset-bootstrap
interval crossing zero) *(historical)*. The human lymph node, the second most dominated repertoire, was as
coherent as the most even one. A four-dataset ρ = −1 that had motivated the prediction was a coincidence of
which datasets had been analysed first.

### 1.4 "Threadfin adds information beyond the RNA centroid of a clone" (frozen prediction P1, synergy)
*Not supported on 2026-10-10.* In held-out prediction of 15 measured clone-level targets, kernel profiles
and the plain RNA centroid gave overlapping confidence intervals in all 15 *(historical)*. The manuscript
therefore makes no claim that kernel profiles predict better than centroids; the benchmark in
[`../benchmarking/`](../benchmarking/README.md) repeats this comparison on current clone definitions with
several external tools.

### 1.5 "Clonal bursts explain shared states among sequence-identical cells" (P3)
*Falsified on 2026-10-10.* The prediction was that cells with identical mutated receptors (burst progeny)
share state more in the rapidly dividing compartment. The opposite was observed: identical-genotype state
sharing was stronger in the slowly dividing compartment (ICC 0.48–0.59 against 0.08–0.27) *(historical)*.

### 1.6 "The RBD dataset is dominated by a 791-cell clone"
*Corrected on 2026-10-11.* The 791-cell clone was an artefact of defining clones on the heavy chain alone.
With paired heavy and light chains the largest RBD clone has 254 cells, and the clone-size Gini coefficient
falls from 0.65 to 0.51. Adding the light chain increased clonal coherence in all ten datasets that had a
light chain (by 0.005–0.14 in ICC) *(historical)*, which is why the paired definition is used throughout.

### 1.7 "The gut immunisation shows two biologically distinct plasma-cell arms"
*Reinterpreted on 2026-10-16.* The lower plasma-cell branch of the gut clone embedding consists almost
entirely of clones from one mouse (136 of 148), whose sampled response was dominated by lamina-propria plasma
cells; centring every clone on its own mouse shifts all of that mouse's clones together. Without centring,
or with composition-balanced centring, the two branches merge (Jaccard 0.82 and 0.81). The manuscript keeps
the path from GC clones through bridging clones to plasma-cell clones (which survives every centring) and
reports the branch as an individual difference (Supplementary Fig. 5). Material:
[`gut_all_B_and_centring/`](gut_all_B_and_centring).

### 1.8 "Nothing about a clone predicts its fate in another animal" (first-freeze P5)
*Falsified on 2026-10-10.* In the *Plasmodium* late time course (76 clones; animals held out whole; features
from one half of each clone's cells, composition measured in the other half), a clone's GC and memory
composition in held-out animals was predicted from its context-centred RNA mean (Threadfin `mean` profile)
at R² 0.45 and 0.55. A plain RNA centroid did as well (0.44 and 0.50), the kernel profile worse (0.21 and
0.30), clone2vec 0.12 and 0.23, V(D)J usage 0.04 and 0.18 and clone size ≤ 0.004 *(historical; heavy-chain
clones)*. Two lessons were carried forward: what is predictable across animals here is predictable from a
clone-level centred mean, so the kernel and shrinkage machinery did not earn their keep on this task; and the
manuscript makes no claim of superior prediction. The analysis predates the paired clone definition and was
not re-run; it is listed in the [manuscript overview](../README.md) as a candidate analysis to repeat.

### 1.9 Converting the clone's mutation gap into days of export lag
The mutation gap between a clone's output cells and its own GC cells is real and one-directional
([`export_lag/`](export_lag)); converting it into days assumes that every clone matures at the population's
average rate, which has not been validated. It is reported as an exploratory analysis, not as a finding.

---

## 2. Defects found in our own code and estimators

| Date | Defect | Effect | Fix |
|---|---|---|---|
| 2026-10-03 | AIRR `sequence_alignment` and `germline_alignment` may start at different positions; `v_sequence_start` indexes the raw read, not the alignment | mutation frequency read as 74 % instead of 0.6 % (GSE287123) | `tl.mutation_frequency(align="auto")` locates the start of one string in the other |
| 2026-10-04 | `clonal_memory` returned a collapsed interval from one clone | spurious certainty for 5- and 6-clone points | the function refuses to run below `min_pairs=10` |
| 2026-10-10 | a reliability estimator correlated clone means only within one sample at a time | the *Plasmodium* reliability rested on 8 clones while 801 were quoted | sample-centred pooled estimator; coverage and evidence base reported separately |
| 2026-10-10 | an empty pair class returned ICC = 1.0 with interval [1, 1] | a meaningless perfect score | empty classes return no estimate |
| 2026-10-10 | the pairwise lineage statistic is confounded by mutation depth | a trait that only depends on depth looked heritable | a phylogenetic mixed model with depth as a fixed effect (`tl.lineage_heritability`); the pre-specified clone × mouse units were invalid (they leave a clone × gate interaction) and clone × sample units are used; likelihood-ratio intervals were invalid on these data and are documented as such; nothing survives beyond identical genotypes |
| 2026-10-10 | the programme split test cannot detect two groups four within-group SDs apart; the obvious repair (null covariance estimated within the candidate groups) has a false-positive rate of 1.0 | a single programme looked like evidence of a continuum | conservative test kept; documented; a single programme is reported as "no split supported" |
| 2026-10-11 | `min_reliability=0.5` as a default | removed 75 % (RBD), 90 % (NP–OVA) and 100 % (GSE246382) of profiled clones; GSE246382 then appeared to have no clone structure | no reliability filter anywhere; shrinkage already weights small clones |
| 2026-10-11 | the Dandelion adapter filled a missing `junction_aa` column with empty strings, and Dandelion groups on `junction_aa` | every heavy-chain-only influenza cell received the same key (a 972-cell "clone") | the junction is translated from nucleotides; the adapter refuses tables with > 5 % unusable junctions |
| 2026-10-11 | hand-tuned clone-map layouts were dropped when coordinate tables were retired | figures silently fell back to default layouts | layout settings live in `run_case_study.CONFIGS` and are recorded in each run |
| 2026-10-18 | stored results contained Python lists and tuples | `adata.write_h5ad` failed after a full analysis | results are converted to writable types |
| 2026-10-20 | `.gitignore` patterns for quick-start outputs also matched `docs/_static/` | the homepage images were never committed | patterns anchored to the repository root |
| 2026-10-20 | `tl.export_lag` fits its clock with a Theil–Sen slope over all cell pairs | memory grows with the square of the number of GC cells (out of memory for 40,000 cells) | the analysis in [`export_lag/`](export_lag) fits per-visit medians; fixed in threadfin 4.3.1 (above five million pairs the slope is the median over a fixed-seed random sample of five million pairs) |
| 2026-10-20 | Dandelion also groups cells without a heavy chain by their light chain alone (ids `B_VJ_…`); `define_clones_dandelion` returned these groups as clones, and the case-study loaders that read 10x per-cell tables kept light-chain-only cells | in GSE287123 run from scratch, 5,276 of 21,939 cells and 740 of 2,626 groups of two or more cells; in the stored case studies, 1,689 of 2,281 groups in GSE171703, 419 of 780 in the tonsil, 1,198 of 8,114 in the EBV organoids, 67 of 6,855 in bone marrow, and one two-cell group among the 1,887 profiled RBD clones (the manuscript's datasets were otherwise unaffected); found when Benisse received receptor sequences for only 29 % of the GSE171703 cells | `require_heavy_chain=True` in threadfin 4.3.1; `run_case_study.prepare` drops the groups; the four affected additional case studies, their cross-dataset tables and the benchmark's GSE171703 input were rebuilt and re-run |
| 2026-10-20 | Dandelion's clone assignment depends on Python's string-hash seed | the same contigs gave 8,729–8,734 clones in different sessions | `PYTHONHASHSEED=0` documented for reproducible ids; the seed in use is stored in the result |
| 2026-10-23 | Cell Ranger names contigs after the cell barcode (`<barcode>_contig_1`), so a donor sequenced in several libraries can have two cells with the same contig ids; `define_clones_dandelion` passed them on and Dandelion mixed those cells' chains | 18 of 21,939 cells in GSE287123 RBD, 8 of 4,855 in NP and 20 of 51,837 in GSE171703 (none in the other datasets); the RBD and NP "light-chain-only groups" counted on 2026-10-20 were this artefact; corrected RBD clones change the partition little (adjusted Rand index 0.9987) and Figure 4 not at all (division excess R² 0.088 → 0.085, crossover cosine 0.887 → 0.896, 10 of 10 mice) | duplicated ids are renamed after their cell in threadfin 4.4.0; [`rbd_complete_pairs/`](rbd_complete_pairs/README.md) |
| 2026-10-23 | `define_clones_dandelion` required a `cell_id` column before applying `cell_key` | a table whose barcode column had another name was rejected with a misleading message (found while calling the GSE203132 paired clones) | the required columns follow `cell_key` (4.4.0) |
| 2026-10-23 | `pyproject.toml` allowed umap-learn < 0.5.5 and pynndescent < 0.5.8, which import `pkg_resources`, absent from new Python environments | an install resolved to the oldest allowed versions could not draw the clone map (27 of 243 tests failed) | minimum versions raised; the oldest allowed versions are now installed and tested ([`../../docs/COMPATIBILITY.md`](../../docs/COMPATIBILITY.md)) |
| 2026-10-23 | an analysis requested from `tf.run` that could not be done on the data (a label without values, a time axis on which no clone is seen twice) was only written to the log | with `verbose=False` the summary did not say that it was missing | `ThreadfinResult.skipped`, listed under "Not run" in `summary()` (4.4.0) |

---

## 3. Design facts that limit what can be concluded

* **Sort gates sequenced in their own libraries are library labels.** In GSE287123, the NP–OVA division
  gate, the RBD bait gate and the RBD zone gate are each completely determined by the sequencing library;
  only the RBD division gate is crossed with libraries between the two cohorts, which the manuscript uses as a
  control (Fig. 4e). In GSE195673, the lymph-node/blood "tissue" label is also a library label.
* **Shared-cell circularity.** When a clone's profile and its label fraction are computed from the same cells,
  any per-cell association, technical ones included, reappears at clone level. The association tests of the
  manuscript are in-sample; the benchmark adds a disjoint-cell version in which the representation and the
  label come from different cells.
* **Annotations derived from RNA** (author cell types, module scores) are not independent of the clone
  profiles; they describe programmes but cannot validate them.
* **Isotype purity.** 63–85 % of clones with at least four isotyped cells carry a single isotype; among
  isotype-mixed clones, isotype still agreed between disjoint halves (0.30–0.62) except in the *Plasmodium*
  late time course *(historical)*.
* **One sampling time per animal** (the *Plasmodium* time course, the gut and RBD cohorts): co-occurrence of
  states within a clone cannot establish ancestry, migration or direction.
