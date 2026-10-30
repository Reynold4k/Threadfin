# Following the tutorials on every public dataset

**Question.** Can a user who follows the tutorials reproduce the analysis of each public dataset of the
manuscript without a single error?

**What was done.** A new Python 3.11 environment was built exactly as the README says (`pip install
"threadfin[dandelion]"`, threadfin 4.6.0, nothing else installed). For each of the 14 public BCR datasets of the case studies,
[`run_dataset.py`](run_dataset.py) did what Tutorial 2 asks of a user — read the deposited raw counts, add the
donor and sample of each cell, and pass the deposited contig-level BCR table with the barcodes renamed to match
(section 4) — and then made the documented call with every setting at its default:

```python
result = tf.run(adata, bcr=contigs, donor_key="donor", sample_key=..., strata_key="sample", test=[...], n_jobs=8)
print(result.summary()); result.methods(); result.plot(...); result.clones.to_csv(...); adata.write_h5ad(...)
```

`strata_key="sample"` makes the coherence test shuffle clone labels within each sample — the library, sort gate
or tissue sample — as the case studies did (Tutorial 2, section 5). The two LARRY datasets have lineage barcodes
instead of a BCR and follow section 6 of Tutorial 2 (clone ids defined elsewhere). Every exception and every warning was recorded. [`run_tutorial_blocks.py`](run_tutorial_blocks.py)
ran every Python block of Tutorials 1, 3 and 4 in order, as written, and compared what they print with the
output shown in the tutorials; Tutorial 5 (the notebook) was executed end to end in a fresh environment.

**Result.** No error on any dataset, in any of the 23 tutorial blocks or in the notebook (5.5 minutes, 7.2 GB).
Every clone table — clone calls, profile reliabilities, clone-map coordinates and programmes — is identical, byte for
byte, to that of threadfin 4.5.2, and so is every clonal ICC. The clones and the ICC agree with the case studies (in
brackets); the small differences in clone numbers come from Dandelion's dependence on the hash seed in the case
studies and from contig ids that threadfin ≥ 4.4 renames when a barcode occurs in two libraries. What 4.6.0 changes
here is the null of the coherence test: shuffled within samples instead of donors, the shuffled ICC rises where a
donor's samples differ (GSE195673 0.001 → 0.164, GSE253857 0.046 → 0.245, GSE284622 0.023 → 0.278), and every
test that was significant remains at the smallest P value that 200 permutations can give (0.005); GSE246382 remains
not significant (P = 0.24). Every figure that Threadfin draws for these runs is in the [figure gallery](figures/README.md)
(the last column of the table links each dataset's page).

<!-- BEGIN:table -->

| dataset | errors | cells | contigs matching a cell | clones ≥ 2 cells (case study) | clonal ICC (case study) | minutes | peak memory | figures |
|---|---|---|---|---|---|---|---|---|
| GSE253857 bone-marrow plasma cells | 0 | 115,144 | 70 % | 6,583 (6,583) | 0.427 (0.427) | 5.3 | 6.85 GB | [32](figures/bone_marrow_pc/README.md) |
| GSE171703 SARS-CoV-2 bait-sorted B cells | 0 | 61,423 | 91 % | 588 (588) | 0.400 (0.400) | 2.2 | 3.82 GB | [12](figures/covid_bcell/README.md) |
| GSE317492 EBV tonsil organoids | 0 | 205,630 | 100 % | 6,124 (6,124) | 0.158 (0.158) | 7.7 | 25.53 GB | [21](figures/ebv/README.md) |
| GSE175522 blood, influenza vaccine | 0 | 123,693 | 87 % | 2,082 (2,082) | 0.561 (0.561) | 4.5 | 8.17 GB | [22](figures/flu/README.md) |
| GSE317692 influenza lung | 0 | 4,121 | 24 % | 204 (204) | 0.162 (0.162) | 1.0 | 1.13 GB | [14](figures/flu_lung/README.md) |
| GSE246382 GC + plasma cells (Smart-seq2, TRUST4) | 0 | 884 | 85 % | 92 (96) | 0.049 (0.048) | 0.5 | 0.55 GB | [19](figures/gc_np_pc/README.md) |
| GSE284622 gut GC-fate reporter | 0 | 16,619 | 93 % | 964 (958) | 0.401 (0.401) | 1.1 | 1.59 GB | [31](figures/gut_gc_reporter/README.md) |
| GSE140802 LARRY day 2 (clone ids given) | 0 | 28,249 | — | 1,401 (—) | 0.680 (—) | 0.9 | 0.62 GB | [11](figures/larry_d2/README.md) |
| GSE140802 LARRY days 4/6 (clone ids given) | 0 | 44,664 | — | 6,174 (—) | 0.598 (—) | 2.7 | 0.9 GB | [11](figures/larry_d46/README.md) |
| GSE195673 lymph node, mRNA vaccine | 0 | 193,442 | 76 % | 11,126 (11,141) | 0.326 (0.326) | 11.1 | 23.66 GB | [26](figures/ln_vaccine/README.md) |
| GSE286215 Plasmodium d0-14 | 0 | 40,623 | 91 % | 523 (523) | 0.735 (0.736) | 1.8 | 3.64 GB | [21](figures/malaria/README.md) |
| GSE286215 Plasmodium d10-42 | 0 | 72,359 | 100 % | 1,007 (1,015) | 0.454 (0.451) | 4.1 | 6.5 GB | [21](figures/malaria_late/README.md) |
| GSE287123 NP reporter | 0 | 10,109 | 47 % | 470 (471) | 0.117 (0.120) | 0.9 | 1.62 GB | [12](figures/mouse_np/README.md) |
| GSE287123 RBD reporter (Tutorial 5) | 0 | 36,188 | 44 % | 1,881 (1,887) | 0.169 (0.169) | 2.2 | 7.9 GB | [32](figures/mouse_rbd/README.md) |
| GSE203132 spontaneous GCs | 0 | 22,898 | 100 % | 1,053 (1,051) | 0.279 (0.279) | 0.9 | 2.22 GB | [16](figures/spont_gc/README.md) |
| E-MTAB-9005 paediatric tonsil | 0 | 22,478 | 80 % | 361 (361) | 0.253 (0.253) | 0.9 | 2.43 GB | [20](figures/tonsil/README.md) |

<!-- END:table -->

**What this found and fixed** (threadfin 4.6.0): in GSE171703 three of the 36 libraries pool cells of several
donors, so shuffling clone labels within libraries moved them between donors; strata now stay within the centring
groups (here donor × library, with a warning), which raised the shuffled ICC from 0.035 to 0.052 there and changed
nothing in the other datasets. A label fixed within each sample was reported twice by `tf.run` (once by each label
test); it is now reported once. The gallery's descriptions of the label tests, the clone-weighted composition and
the memory index were brought up to 4.6.0.

**Earlier** (threadfin 4.5.0): a subset made without `.copy()` lost its results and stopped with
a misleading message; a contig table with both a 10x `barcode` and an AIRR `cell_id` column could end up with two
cell-id columns; three warnings that had nothing to do with the user's data (a notice printed when Dandelion is
imported, a scanpy notice about densifying a sparse matrix, and one about duplicated gene names) are gone; the
expected share of matching BCR contigs in Tutorial 2 now comes from these datasets; the warning that programmes
did not reach the stability threshold, given in 12 datasets, now appears only when two or more programmes remain
(10 datasets).

**Files.** `results/tutorial_path.csv` (one row per dataset), `results/<dataset>.json` (everything recorded,
including each warning), `results/<dataset>_summary.txt` (the printed summary), `results/<dataset>_methods.txt`
(the Methods paragraph), `results/tutorial_blocks.json` (every tutorial block with its printed output).
