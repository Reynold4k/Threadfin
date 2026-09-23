# Threadfin validation report: ebv_organoid_gse317492

Tonsil organoid primary EBV infection, d0/d4/d7/d14/d21, GFP+/- sorted at d14/d21 (Mitul et al. 2026, PNAS). GFP status = experimentally infected cells (external, non-transcriptomic label).

## Dataset QC

- cells: 205,630; BCR-positive: 123,362 (60%)
- clonotypes: 89,757; expanded (>=3 cells): 4,564; (>=5): 2,350
- clone size: median 1.0, max 302

- timepoint: {'d21': 75519, 'd14': 64893, 'd7': 22786, 'd0': 21786, 'd4': 20646}
- condition: {'GFP+': 71576, 'GFP-': 68836, 'uninfected': 65218}
- state: {'0': 20201, '1': 19546, '2': 19006, '3': 13824, '4': 12082, '5': 10736, '6': 8541, '7': 8203, '8': 7819, '9': 7589, '10': 7239, '11': 6820, '12': 6163, '13': 6130, '14': 5863, '15': 5816, '16': 4915, '17': 4486, '18': 4482, '19': 4259, '20': 4248, '21': 3313, '22': 3222, '23': 2208, '24': 2203, '25': 1728, '26': 1629, '27': 1590, '28': 1380, '29': 331, '30': 25, '31': 25, '32': 8}

## Threadfin runs (parameter grid)

| run | clusters | NMI vs state | ARI vs state | #sig enrichments |
|---|---|---|---|---|
| tf__pca__r0.3__cw0.0 | 13 | 0.668 | 0.506 | 32 |
| tf__pca__r0.3__cw0.3 | 10 | 0.016 | 0.001 | 34 |
| tf__pca__r0.8__cw0.0 | 16 | 0.657 | 0.455 | 32 |
| tf__pca__r0.8__cw0.3 | 14 | 0.026 | 0.005 | 48 |
| tf__umap__r0.3__cw0.0 | 10 | 0.375 | 0.223 | 52 |
| tf__umap__r0.3__cw0.3 | 13 | 0.029 | 0.005 | 46 |
| tf__umap__r0.8__cw0.0 | 21 | 0.394 | 0.164 | 91 |
| tf__umap__r0.8__cw0.3 | 16 | 0.044 | 0.007 | 51 |
| tf__joint__r0.3 | 20 | 0.592 | 0.326 | 39 |

## Robustness and statistical support

- **Joint GEX+BCR embedding** (lam=0.5): latent-vs-GEX distance Spearman **-0.002**, latent-vs-BCR **0.313** over 72526 graph edges; modality contribution GEX 0.92 / BCR 0.08
- **Basis robustness** (PCA vs UMAP): clone-level ARI **0.352**, NMI 0.467 over 4564 clones. Low ARI means fine-grained community boundaries shift between bases; check whether the biologically significant enrichments replicate across bases.
- **Null 1 (clone-label permutation, within donor)**: observed mean clone state-purity **0.791** vs null 0.314 ± 0.002; p = 0.0050
- **Null 3 (size-matched random communities)**: observed 32 significant state enrichments vs null 41.22 ± 4.51; p = 0.9851
- **Held-out split-clone validation**: sibling halves of the same clone co-cluster at **0.98** ± 0.00 vs chance 0.10 (1199 clones x 5 repeats, basis=X_pca)

Runtime: 36.5 min.
