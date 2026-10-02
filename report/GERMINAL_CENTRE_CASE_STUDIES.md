# Germinal-centre clones after model-antigen immunisation

*A step-by-step walkthrough for readers without a computational background.
Everything here comes from published mouse experiments in which a defined
model antigen (NP-OVA, or an RBD protein/mRNA vaccine) was given and
germinal-centre B cells were sorted by what they had been doing. Numbers are
reproduced by `case_studies/run_case_study.py`.*

---

## Why clones, and why the germinal centre is awkward

A germinal centre is a cycle. B cells divide and mutate their receptor in the
**dark zone**, test the new receptor in the **light zone**, and are then either
sent back for another round or leave as plasma cells or memory cells. Nothing
in this cycle is a beginning and nothing is an end, so ordering single cells
along a "pseudotime" asks a question the biology does not answer: a cell's
state tells you where it is in the cycle, not how far it has come.

A clone is different. All the cells of one clone descend from a single
ancestor, so a clone has a history even when its individual cells do not have
an order. Threadfin therefore compares **clones with one another**: each clone
is summarised by the whole distribution of states its cells occupy, relative to
the cells it was sampled alongside, and clones are then placed on a common map.
The question becomes "which clones look as though they were recently selected,
and which look as though they were sent back to divide again?"

## The experiments

| study | antigen | what was sorted | why it is useful here |
|---|---|---|---|
| Merkenschlager et al. 2025, *Nature* (NP arm) | NP-OVA | germinal-centre B cells split by a fluorescent **division reporter** (most- vs least-divided) | the number of divisions a cell has made is measured, not inferred |
| Merkenschlager et al. 2025, *Nature* (vaccine arm) | RBD protein or mRNA | division reporter **and** dark/light zone **and** antigen-binding bait | three independent readouts of selection in one experiment |
| ElTanbouly et al. 2023, *J Exp Med* | NP-OVA | dark zone, light zone, Myc-GFP+ light zone (recently selected) and **plasma cells** | the fate a clone's cells took is observed directly |

In each dataset clones were defined within each mouse from the heavy-chain
sequences, with hypermutated relatives kept together (distance thresholds
0.14-0.16, chosen from the data).

## Step 1. Is a B cell's state inherited within its clone?

Clone identity explained **9.3%** of the variation in gene expression between
germinal-centre cells of the same sort gate in the NP-OVA experiment, and
**14.5%** in the vaccine experiment, against 3.5% and 4.6% when the same clone
labels were shuffled among cells of the same gate (both p = 0.002, 373 and
1,414 clones). So sister cells of a clone resemble each other more than
unrelated cells sampled beside them, but most of the variation between cells is
*not* explained by which clone they belong to: a clone biases what its cells
do rather than dictating it.

In the smaller Smart-seq2 experiment (113 clones with at least two cells, 388
cells) the same measurement gave 4.3% against 2.8% shuffled (p = 0.22). We read
this as too few cells per clone to measure the effect, not as evidence against
it; a reliable profile there would need about 23 cells from a clone, and almost
no clone was sampled that deeply.

## Step 2. Do clones fall into distinct programmes?

No. In all three experiments, grouping the clones and then testing whether any
split was better than one continuous spread of clones returned **a single
group**: germinal-centre clones differ from one another along a continuum
rather than falling into separate types. This is worth stating plainly, because
clustering always returns groups if you ask it to; the test is what stops a
continuum from being reported as discrete programmes.

## Step 3. What explains the differences between clones?

Each clone was summarised by one value per label (for example, the fraction of
its cells in the most-divided gate), and labels were shuffled among clones of
the same mouse to see how much could be explained by chance.

| what was measured | NP-OVA | RBD vaccine |
|---|---|---|
| divisions since labelling | **17.6%** (p = 0.0005) | **8.0%** (p = 0.0005) |
| dark- vs light-zone sort | not sorted | **9.0%** (p = 0.0005) |
| antigen binding (bait) | not sorted | 1.3% (p = 0.013) |
| somatic mutation load | 2.1% (p = 0.19) | 0.9% (p = 0.022) |
| isotype | 5.1% (p = 0.093) | 0.8% (p = 0.018) |
| high-affinity W33L mutation | 2.3% (p = 0.98, only 19 clones) | not applicable |

Division history is consistently the strongest explanation of how clones
differ, and the dark/light-zone sort is comparable. Antigen binding and
mutation load explain much less. One reading, which these data support but do
not prove, is that the clone-level signal mostly reflects **where a clone
currently sits in the selection cycle** rather than a fixed property of the
clone: clones caught in a state of recent division look alike, and so do clones
caught resting in the light zone.

The two strongest effects also survive a stricter control. Each sort gate was
sequenced as its own library, so a difference between gates could in principle
be technical. Re-measuring with every clone compared only against cells of its
own library still gave 16.0% for divisions in the NP-OVA experiment
(p = 0.0002) and 11.8% and 11.6% in the two vaccine arms (p = 0.0002 and
0.003), with the zone sort at 5.1% (p = 0.026).

## Step 4. Does a clone keep its state across the cycle?

If a clone's cells are found in two different compartments, do those two groups
of cells still resemble each other? The index below is 0 when a clone's cells
in one compartment say nothing about its cells in the other, and 1 when they
are as alike as two samples of the same thing.

| compared across | clones | index (95% interval) |
|---|---|---|
| dark- vs light-zone cells (RBD vaccine) | 57 | 0.53 (0.32-0.73) |
| antigen-binding vs non-binding cells (RBD vaccine) | 105 | 0.49 (0.33-0.65) |
| most- vs least-divided cells (RBD vaccine) | 173 | 0.47 (0.34-0.60) |
| most- vs least-divided cells (NP-OVA) | 70 | 0.28 (0.03-0.49) |

About half of what distinguishes a clone is retained when its cells are caught
in a different part of the cycle. A clone is therefore neither locked into one
state nor reset at every round: it carries something recognisable with it. We
cannot tell from these snapshots whether that something is inherited
transcriptional state, the receptor itself, or the clone's position in the
tissue; distinguishing those would need an experiment that follows the same
clone over time.

## Step 5. Which genes are inherited within clones?

Ranking genes by how much of their expression is explained by clone identity
puts surface receptors and interferon-response genes near the top in both
experiments (*Cd72*, *Cd38*, *Ccr6*, *Ly6d*, *Usp18*, *Ifi213*), while
cell-cycle and dark-zone genes score low relative to genes of similar
expression level. This is the pattern expected if the cycle itself is something
every clone passes through, while the clone-specific part is closer to how the
cell senses and responds to its surroundings. These are associations in
observational data and would need perturbation to be called causal.

## Step 6. An independent check from the receptor sequences

Everything above uses gene expression. The receptor sequences give a separate
handle on the same question. Within a clone, cells carrying more somatic
mutations arose later in that clone's history, so a mutation-based family tree
orders a clone's cells without using their expression at all. Trees were built
for each clone (GCtree, abundance-aware parsimony, rooted at the unmutated
germline sequence): 213 trees over 1,790 cells in the NP-OVA division-reporter
experiment, and 45 trees over 240 cells in the sorted-fate experiment. Each
cell's position was then compared with the compartment it was sorted from,
**within clones**, so differences between clones cannot produce the result.

| comparison | clones | mutations from the ancestor | p |
|---|---|---|---|
| most- vs least-divided cells (NP-OVA) | 145 | -1.9 | 0.14 |
| plasma cells vs their germinal-centre sisters | 28 | -0.6 | 0.13 |
| dark-zone vs light-zone sisters | 16 | +1.3 | 0.16 |

None of these reaches significance, and we report them as such. Two points are
worth noting anyway. First, the directions are the ones the original studies
describe: plasma cells carry slightly fewer mutations than the germinal-centre
cells of the same clone, and dark-zone cells slightly more than their
light-zone sisters. Second, cells from the most-divided gate are **not**
further from the germline than their least-divided sisters, which is what
would be expected if divisions and mutations accumulated together; the one
difference that did reach the 5% level was that the most-divided cells more
often occupy internal positions in their clone's tree, that is, positions with
observed descendants (+3.3 percentage points, p = 0.036, 145 clones). This is
consistent with the original report that high-affinity clones divide more but
mutate less per division, though with these numbers it is a hint rather than a
result.

## Step 7. Is the clone's behaviour just a readout of its antibody?

A sceptical reading of everything above is that the expression differences
between clones simply reflect how good each clone's antibody is, in which case
the receptor sequence alone would be enough and gene expression would add
nothing. This can be tested.

Somatic hypermutation scatters mutations across the V region; those that change
an amino acid are visible to selection, those that are silent are not. A clone
whose antibody has been selected for therefore carries more amino-acid-changing
mutations than mutation alone would produce. How many to expect is not a
universal constant - it depends on the clone's own germline sequence, because
the genetic code makes some positions more likely to change a residue than
others - so the expectation is computed per clone by mutating its own germline
at random (`case_studies/sequence_selection.py`).

Across all three experiments the antibodies do show the expected signature of
selection: 78-79% of mutations change an amino acid, against 77.3-77.5%
expected by chance. But this sequence-level selection score is **largely
unrelated** to what the clone's cells were doing:

| experiment | compared with | clones | correlation | p |
|---|---|---|---|---|
| RBD vaccine | fraction of cells most divided | 1,101 | +0.08 | 0.009 |
| RBD vaccine | fraction of cells binding the bait | 820 | -0.03 | 0.32 |
| RBD vaccine | fraction of cells sorted dark zone | 281 | +0.09 | 0.15 |
| NP-OVA, divisions | fraction of cells most divided | 343 | +0.05 | 0.36 |
| NP-OVA, sorted fates | fraction of cells that are plasma cells | 94 | -0.13 | 0.20 |

Only the largest comparison reaches significance, and even there the
correlation is weak. We read this cautiously, in two directions. It is a
reminder that a weak correlation over a thousand clones is not a strong
biological effect. And it suggests that the clone-level signal Threadfin
measures in gene expression is **not simply a restatement of the antibody
sequence**: if it were, the two measurements would agree far more than they do.
That is the clearest argument we have for needing expression at the level of
clones at all.

## What we are not claiming

* These are associations in published observational data. Threadfin's output is
  a hypothesis about clones, not a demonstration of mechanism; testing it needs
  experiments that follow or perturb the same clones.
* "Recently selected" and "sent back to divide" are interpretations of sort
  gates and division reporters, which are themselves indirect.
* Clone-level measurements depend on how many cells of a clone were sampled.
  Every number above is accompanied by the number of clones it rests on, and
  the smallest dataset is reported as underpowered rather than negative.
* Sorted compartments were sequenced as separate libraries in these
  experiments; the within-library re-analysis is a control, not a guarantee.
