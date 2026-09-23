# Threadfin validation report: ln_vaccine_gse195673

SARS-CoV-2 mRNA vaccine, axillary lymph-node FNA + blood, 8 donors, 5 timepoints (Kim/Zhou et al. 2022, Nature). Author-integrated h5ad + Change-O BCR tables (Zenodo 5895181).

## Dataset QC

- cells: 193,442; BCR-positive: 153,049 (79%)
- clonotypes: 92,761; expanded (>=3 cells): 4,396; (>=5): 2,488
- clone size: median 1.0, max 918

- donor: {'368-10': 49394, '368-02a': 34719, '368-07': 25528, '368-22': 20821, '368-01a': 19382, '368-20': 17609, '368-04': 13015, '368-13': 12974}
- timepoint: {'d60': 74805, 'd110': 46163, 'd28': 29248, 'd201': 27515, 'd35': 13762, 'd28+d35': 1949}
- state: {'GC': 62156, 'RMB': 42255, 'Naive': 38686, 'PB': 27231, 'LNPC': 12299, 'B & T': 10797, 'PB-like': 18}

## Threadfin runs (parameter grid)

| run | clusters | NMI vs state | ARI vs state | #sig enrichments |
|---|---|---|---|---|
| tf__pca__r0.3__cw0.0 | 11 | 0.196 | 0.152 | 19 |
| tf__pca__r0.3__cw0.3 | 9 | 0.016 | 0.022 | 17 |
| tf__pca__r0.8__cw0.0 | 22 | 0.202 | 0.111 | 34 |
| tf__pca__r0.8__cw0.3 | 14 | 0.038 | 0.035 | 24 |
| tf__umap__r0.3__cw0.0 | 14 | 0.271 | 0.136 | 22 |
| tf__umap__r0.3__cw0.3 | 12 | 0.016 | 0.029 | 22 |
| tf__umap__r0.8__cw0.0 | 24 | 0.242 | 0.086 | 37 |
| tf__umap__r0.8__cw0.3 | 15 | 0.060 | 0.050 | 28 |
| tf__joint__r0.3 | 23 | 0.202 | 0.100 | 35 |

## Robustness and statistical support

- **Joint GEX+BCR embedding** (lam=0.5): latent-vs-GEX distance Spearman **0.243**, latent-vs-BCR **-0.123** over 65433 graph edges; modality contribution GEX 0.89 / BCR 0.11
- **Basis robustness** (PCA vs UMAP): clone-level ARI **0.234**, NMI 0.439 over 4396 clones. Low ARI means fine-grained community boundaries shift between bases; check whether the biologically significant enrichments replicate across bases.
- **Null 1 (clone-label permutation, within donor)**: observed mean clone state-purity **0.849** vs null 0.478 ± 0.002; p = 0.0050
- **Null 3 (size-matched random communities)**: observed 19 significant state enrichments vs null 13.40 ± 2.44; p = 0.0149
- **Held-out split-clone validation**: sibling halves of the same clone co-cluster at **0.93** ± 0.01 vs chance 0.09 (1517 clones x 5 repeats, basis=X_pca)

Runtime: 18.3 min.
