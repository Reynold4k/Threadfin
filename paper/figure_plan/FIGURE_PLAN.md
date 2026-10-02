# Figure plan

Each main figure is one A4 page, laid out on a 183 mm wide area (Nature
double column). `make_figures.py` draws every panel that comes from data and
leaves a dashed box wherever a panel still has to be drawn by hand; the boxes
say what the panel will show. Rebuild everything with:

```bash
python make_figures.py        # Figure_1..5 here; method and benchmark figures go outside the repository
```

Technical panels (clone-calling thresholds, the test that decides whether
clones form groups, simulations) are deliberately **not** in the main figures.
They are rendered to `internal_validation/figures/` instead.

---

## Figure 1 — Reading a germinal centre at the level of clones

The concept figure. It has to make one idea obvious before any result: a
germinal centre is a cycle with no beginning and no end, so cells cannot be
ordered along it, but clones can be compared with one another.

| panel | content | source | status |
|---|---|---|---|
| a | the germinal-centre cycle: dark zone -> light zone -> re-enter or exit; why a cell's state says where it is, not how far it has come | schematic | drawn |
| b | a clone is a family sampled repeatedly around the cycle, so it is a distribution over states, not a point | schematic | drawn |
| c | the resulting map of clones: one point per clone, related clones near each other, ordered from "recently selected" to "sent back to divide" | schematic | drawn |
| d | workflow: sequences -> clones within donors -> embedding without immunoglobulin genes -> clone profiles -> the four questions | schematic | **to draw** |
| e | two clones with the same average state but different spreads, and why the distribution distinguishes them | simulated example | drawn |
| f | how many cells of a clone are needed before its profile is reliable, per dataset | case studies | drawn |
| g | clonal memory: the same clone sampled twice versus a random clone, after removing sampling noise | schematic | **to draw** |
| h | programmes or a continuum: clones are grouped only when the split is real | schematic | **to draw** |
| i | outputs at a glance | schematic | **to draw** |

## Figure 2 — Germinal-centre clones differ along a continuum of selection

Model-antigen immunisation only (NP-OVA; RBD protein and mRNA vaccine), where
cells were sorted by what they had been doing.

| panel | content | source | status |
|---|---|---|---|
| a | experimental design: immunisation, sorting by division reporter / zone / antigen bait / plasma cells, paired RNA + BCR sequencing | schematic | **to draw** |
| b | how much of B-cell state is explained by clone identity, against shuffled clones, in each experiment | `case_studies/results/*/summary.json` | drawn |
| c | the map of clones, coloured by how much the clone's cells had divided; no split between groups was significant | `clone_table.csv` | drawn |
| d | what explains the differences between clones: divisions, zone, antigen binding, mutation load, isotype | `label_effects.csv` | drawn |
| e | clonal memory across the sorted compartments | `summary.json` | drawn |
| f | position of a cell in its clone's mutation-based family tree, by compartment - an independent check that uses only the receptor sequences | `lineage/within_clone_tree_comparisons.csv` | drawn once trees are built |
| g | biological interpretation: clones caught dividing look alike, clones caught in the light zone look alike, and about half of what distinguishes a clone is carried across the cycle | schematic | **to draw** |
| h | what would settle it: an experiment that follows or perturbs the same clones over time | schematic | **to draw** |

## Figure 3 — A vaccinated human lymph node over six months *(to assemble)*

| panel | content | source |
|---|---|---|
| a | design: lymph-node aspirates and blood, eight donors, week 0 to month 6 | schematic, **to draw** |
| b | the groups of clones found, with their stability and marker genes | `ln_vaccine/programmes.csv`, `programme_markers.csv` |
| c | spike-binding clones across the groups (odds ratios per donor) | `programme_associations.csv` |
| d | clonal memory over months, and the absence of it between blood and lymph node | `summary.json` |
| e | interpretation: tissue overrides clone identity | schematic, **to draw** |

## Figure 4 — Clone states across human blood and tonsil *(to assemble)*

Influenza vaccination (recall burst at day 7), tonsil (a continuum ordered by
isotype), and COVID-19 blood as a smaller demonstration. Panels: groups of
clones per dataset; what explains clone differences; the day-7 secreting
burst; and a placeholder for the biological summary.

## Figure 5 — Which genes are inherited within clones *(to assemble)*

Gene-level clonal inheritance across datasets, gene sets compared with genes of
similar expression level, and a placeholder for the interpretation (the cycle
is shared by all clones; what differs is closer to how cells sense their
surroundings).

## Extended data *(to assemble)*

Per-dataset quality control, the same analyses with the sampling context
changed, and the comparison with the authors' own clone calls.

---

## Conventions

* 183 mm wide, A4 page, Liberation Sans (Arial metrics), panel letters in bold
  lower case.
* One accent colour (blue) for Threadfin results, grey for comparisons and
  shuffled controls, orange only where a second category is needed.
* Every panel that rests on a statistical test states the number of clones and
  the value expected by chance.
* Figures never claim mechanism; interpretation panels are labelled as
  interpretation.
