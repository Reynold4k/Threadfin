# What Threadfin finds in published data

*A step-by-step walkthrough for readers without a computational background.
Eight published datasets, human and mouse, re-analysed with the same script
(`case_studies/run_case_study.py`). Each section says what the experiment was,
what the analysis asked, what came out, and what it does not show.*

For the germinal-centre experiments with model antigens, a longer walkthrough
is in [`GERMINAL_CENTRE_CASE_STUDIES.md`](GERMINAL_CENTRE_CASE_STUDIES.md).

---

## How to read these results

**A clone** is a family of B cells descended from one ancestor, recognised by
their shared receptor sequence. Threadfin treats the clone, not the cell, as
the thing being measured.

**Five questions** are asked of every dataset:

1. *Is B-cell state inherited within clones?* The share of the variation in
   gene expression that is explained by which clone a cell belongs to, compared
   with the same clone labels shuffled among cells of the same sample. Shuffling
   within samples matters: clones are usually confined to one sample, so a
   looser comparison would call ordinary differences between samples "clonal".
2. *Do clones fall into distinct groups?* Clones are grouped only when the split
   between groups is better than one continuous spread. Otherwise the answer is
   "a continuum", which is a real answer, not a failure.
3. *What explains the differences between clones?* Each label measured in the
   experiment (antigen binding, isotype, mutation load, sort gate, tissue, time)
   is summarised per clone and tested with clones as the replicates.
4. *Do clones keep their state?* The same clone sampled at two times, in two
   tissues or in two sort gates, compared with unrelated clones of the same
   donor.
5. *Which genes are inherited within clones?*

**Two numbers accompany every claim**: how many clones it rests on, and what
the same analysis gives when the labels are shuffled. A clone sampled as two
cells says little; the reports state how many cells a clone needed in each
dataset before its profile was reliable.

---

## 1. Human lymph node after mRNA vaccination (Kim et al. 2022, *Nature*)

**The experiment.** Eight people vaccinated against SARS-CoV-2; lymph-node
needle aspirates and blood taken repeatedly from the first week to six months.
193,442 B cells, 153,049 with a receptor sequence, 8,435 clones with at least
two cells.

**Is state inherited within clones?** Yes: 28.8% of the variation between cells
of the same sample is explained by clone identity, against 14.4% for shuffled
clones (p = 0.002). Three cells were enough for a reliable clone profile here.

**Which clones behave alike?** Four groups of clones, two of them stable:
a germinal-centre group (*MS4A1*, *MEF2B*, *LRMP*), an antibody-secreting group
(*JCHAIN*, *MZB1*, *TXNDC5*, *IGHG1*), a second secreting group with
*BHLHA15* and *DNAJB9*, and a resting group (*BANK1*, *FCMR*, ribosomal genes).

**What distinguishes them?** Clones whose antibodies bound the spike protein
were strongly over-represented in the germinal-centre group and in one
secreting group (odds ratios around 3.2 and 3.5 across donors), and almost
absent from the other secreting group (odds ratio 0.03). Isotype and mutation
load also separated the groups.

**Do clones keep their state?** Over months, partly: the memory index across
time points was 0.26 (95% interval 0.20-0.31, 420 clones). Across tissue it was
0.0 (-0.03 to 0.01, 159 clones): the same clone's cells in blood and lymph node
did **not** resemble each other. The simplest reading is that where a cell is
matters more than which clone it came from, so clone-level claims about state
should always say which tissue they refer to.

---

## 2-3. Mouse germinal centres after model-antigen immunisation

Covered in detail in the germinal-centre walkthrough. In brief: clone identity
explained 9.3% (NP-OVA) and 14.5% (RBD vaccine) of B-cell state; clones formed
a **continuum** rather than distinct groups in both; how many times a clone's
cells had divided was the strongest explanation of how clones differ (17.6% and
8.0%), with the dark/light-zone sort comparable (9.0%); and about half of what
distinguishes a clone was retained when its cells were caught in a different
part of the cycle.

---

## 4. Human blood before and after influenza vaccination (Wang et al. 2023)

**The experiment.** Six adults, blood B cells before vaccination and seven days
after. 123,693 cells, 1,997 clones with at least two cells.

**Is state inherited within clones?** The largest value we measured: 55.8%
versus 3.0% shuffled (p = 0.002). This is expected rather than surprising -
blood contains naive cells, memory cells and a short-lived burst of
antibody-secreting cells, and a clone is usually of one kind.

**Which clones behave alike?** Four groups: a naive-like group (*TCL1A*,
*FCER2*, *IL4R*), two memory-like groups (one with ribosomal and *LTB*, one with
MHC class II genes), and an antibody-secreting group (*MZB1*, *XBP1*,
*TNFRSF17*, *FKBP11*).

**What distinguishes them?** Isotype explained 15.1% of the differences between
clones and mutation load 9.1% (both p = 0.0005). The secreting group was
dominated by class-switched, mutated clones and was almost entirely a day-7
population; the naive-like group was unmutated, IgM/IgD, and present on both
days. This is the expected signature of a recall response: the cells that burst
out at day 7 come from clones that had already been through a germinal centre.

**Caution.** Day 0 and day 7 are different blood draws, so "time" and "sample"
are the same thing here; a clone seen only at day 7 cannot be distinguished
from a clone that was simply missed at day 0.

---

## 5. Human tonsil (King et al. 2021, *Sci Immunol*)

**The experiment.** Six paediatric tonsils, 22,478 cells, 670 clones with at
least two cells.

**Results.** Clone identity explained 16.6% of B-cell state against 2.6%
shuffled (p = 0.002), but no split between groups of clones was significant:
tonsil clones vary along a continuum. Isotype still explained 12.7% of the
differences between clones (p = 0.0005), so the continuum is not featureless -
it is partly ordered by class switching.

This dataset is a good illustration of why the group test matters: clustering
would happily have returned several "programmes" here.

---

## 6. Human tonsil organoids infected with Epstein-Barr virus (Mitul et al. 2026, *PNAS*)

**The experiment.** Tonsil cells from pooled donors grown as organoids and
infected with EBV carrying a green fluorescent marker; sampled from day 0 to
day 21, with infected (GFP+) and uninfected cells sorted separately at the last
two time points. 205,630 cells, 6,923 clones with at least two cells.

**Results.** Clone identity explained 15.7% of state against 4.2% shuffled
(p = 0.002). Three groups of clones were found but **none was stable** under
resampling (0.46, 0.39, 0.15, all below the 0.6 threshold for interpretation),
so we do not describe them. Whether a clone's cells were infected explained
9.5% of the differences between clones, and isotype 14.3% (both p = 0.0005).
Clones kept their state moderately across time (0.31) and between infected and
uninfected cells (0.33).

**Caution, and why this dataset is reported but not emphasised.** The donors
were pooled before infection, so "donor" cannot be used to stratify the tests,
and clones from different people may be mixed. The programme groups were
unstable. We report the dataset as exploratory.

---

## 7. Human blood in COVID-19 (Stephenson et al. 2021, *Nat Med*)

**The experiment.** A 5,000-cell subset spanning 25 donors across a severity
gradient; only 83 clones have two or more cells.

**Results.** Clone identity explained 34.6% of state against 2.3% shuffled.
Two groups of clones: a resting B-cell group and an antibody-secreting group.
Isotype explained 21.0% of the differences between clones and mutation load
9.1%. Clinical severity appeared to explain 18.4% - but severity is a property
of the donor, not of the clone, so this cannot be separated from any other
difference between those donors, and it is reported as such.

**Caution.** With 83 clones, this dataset is a demonstration rather than
evidence.

---

## What held across datasets

* **Clone identity always explained some of B-cell state** (9% to 56%), and
  always far more than shuffled clones. It never explained most of it: a clone
  biases what its cells do.
* **Distinct groups of clones are not the rule.** Four of eight datasets gave a
  continuum rather than separate groups. Where groups were found, they
  corresponded to recognisable biology (secreting versus resting, naive versus
  memory) rather than to subtle structure.
* **Isotype and mutation load explain clone differences in blood and tonsil;
  division history and zone explain them in germinal centres.** In other words,
  the clone-level signal tracks the stage a clone is at.
* **Clonal state is partly, not fully, retained** - about 0.3 over months in a
  vaccinated lymph node, about 0.5 across germinal-centre compartments - and is
  **not** retained across tissues.

## What this cannot show

These are published observational datasets. Threadfin's outputs are hypotheses
about clones, not demonstrations of mechanism: the associations above are
consistent with several explanations, and distinguishing them needs experiments
that follow or perturb the same clones. Sorted compartments are often
sequenced as separate libraries, so comparisons between them carry a technical
component; where that mattered most (the mouse experiments) the analysis was
repeated within single libraries. Labels fixed for a donor cannot be separated
from the donor. Finally, every number here depends on how many cells of each
clone were sequenced, which is a property of the experiment, not of the biology.
