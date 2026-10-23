# T cells, end to end

**Question.** Threadfin was built and validated on B cells. Does the same call work for T cells, with the
receptor data in a scirpy object as many T-cell users keep them?

**Data.** Wu et al. 2020 (*Nature* 579:274): 3,000 T cells from tumour, normal adjacent tissue and blood of 14
patients, as distributed with scirpy (`scirpy.datasets.wu2020_3k`, a MuData with the receptor chains in
`obsm["airr"]`).

**Analysis.** One call, with the scirpy object passed as the receptor data:

```python
result = tf.run(mdata["gex"], bcr=mdata, donor_key="patient", sample_key="sample", test=["source"])
```

Clones are called by Dandelion from beta and alpha chains with identical junctions (Dandelion's default for T
cells, whose receptors do not mutate), within each patient. The clone calls are also compared with and without
Dandelion's contig check.

```
PYTHONHASHSEED=0 python tcr_end_to_end.py results     # needs scirpy and sc-dandelion; 1 minute
```

**Result** ([`results/tcr_end_to_end.json`](results/tcr_end_to_end.json),
[summary](results/summary.txt), [Methods paragraph](results/methods.txt)).

* The receptor was recognised as TCR (97 % of the IG and TR contigs; the other 3 % are immunoglobulin contigs).
  2,793 of 3,000 cells received a clone; 191 clones have two or more cells. The 207 cells without a beta chain
  were left without a clone: alpha chains alone are not diverse enough, and invariant alpha chains
  (for example of MAIT cells) are shared by unrelated T cells.
* Clonal coherence: clone identity explains 26.5 % of the variation of cell states within samples, against
  3.0 % for shuffled clones (P = 0.005, the smallest possible with 200 permutations). The tissue of origin
  explains 1.7 % of the variation between clone profiles, against 1.1 % for shuffled labels (P = 0.08), and the
  clones vary along a continuum (no split significant).
* 231 cells (8 %) carry two productive chains of one kind (127 two beta chains, 104 two alpha chains) and so a
  clone id that joins both. With Dandelion's contig check, the two-beta cells keep their dominant chain or are set
  aside (168 contigs), the two-alpha cells keep both (dual-alpha T cells are common, since TCR alpha rearrangement
  is not allelically excluded), and the clones with two or more cells rise from 191 to 199 (cells in them from 611 to
  636).

**Conclusion.** T cells work with the same call; nothing in the analysis after clone calling depends on the
receptor type. The run takes a minute for 3,000 cells.
