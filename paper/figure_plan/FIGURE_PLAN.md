# Figure plan

Five main figures and four supplementary figures, drawn from the committed
outputs of `case_studies/results/` by `make_figures.py` and
`make_supplementary.py`. Every page is 183 mm wide (Nature double column) and
as tall as its content; nothing is hand-placed outside those two scripts.

Run both after any case study is re-run:

```bash
python make_figures.py          # Figure_1..5
python make_supplementary.py    # Supplementary_1..4
```

## The argument the figures make

| Figure | Question | What it shows |
|---|---|---|
| **1** | What is a clone, and what can be measured about one? | The three settings where clones matter, the workflow, the four questions, why a distribution beats a centroid, and how reliability grows with clone size |
| **2** | Among sorted germinal-centre B cells, where selection is measured, what orders the clones? | A continuum, ordered by division history and zone rather than antigen binding; about half the state survives a round of selection; the mutation tree inside a clone predicts nothing in the two experiments able to detect it |
| **3** | Does any of it survive a live infection? | Influenza, with a genetic perturbation that moves the measurement; a Plasmodium time course from day 0 to 42, in which clonal structure peaks in week one and the dominant axis shifts from plasmablast fate to germinal centre and mutation load |
| **4** | What is a clone when no germinal centre is organising it? | Human bone-marrow plasma cells and the extrafollicular first fortnight of infection; where clones form groups and where a continuum across ten datasets (groups appear in samples that span compartments, always with an antibody-secreting group; this is partly built in for sorted bone marrow); how much a clone keeps its state; and the regime where no clone carries any sequence diversity |
| **5** | Which parts of the B-cell programme are inherited? | Gene sets ranked against genes expressed at the same level |

| Supplementary | Question |
|---|---|
| **1** | Are the clones defined correctly, and how big are they? |
| **2** | Is the comparison fair, and is there enough data? |
| **3** | The model-antigen germinal centre in detail, including every lineage tree |
| **4** | What other tools can and cannot do, on the same clones |

## Panels that are drawn rather than computed

All schematics live in `schematics.py` and are vector line art, not images:
the workflow (Fig. 1c), clonal memory (1g), groups versus continuum (1h), and
the three experimental designs (2a, 3a, 4a), plus the benchmark design
(Supplementary 4a). They carry no data and are safe to edit freely.

## Conventions

* Blue is the observed quantity, grey the same quantity with clones shuffled
  within a sample; a dashed outline marks the shuffled control drawn on top of
  a bar.
* Yellow marks a label that is a property of the animal or donor and so cannot
  be separated from other differences between them.
* Every panel that reports a test states the number of clones and a p value.
* Where a dataset is too small to test, the figure says so instead of leaving
  the panel out.
