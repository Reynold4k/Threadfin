# Threadfin validation report: tonsil_king2021

Human palatine tonsil B-cell maturation, 6 pediatric donors (King et al. 2021, Sci Immunol). Reference states: author cell-type annotations (CellTypeMetaData.txt).

## Dataset QC

- cells: 22,478; BCR-positive: 11,570 (51%)
- clonotypes: 10,473; expanded (>=3 cells): 200; (>=5): 51
- clone size: median 1.0, max 40

- donor: {'BCP005': 12105, 'BCP006': 2982, 'BCP009': 2974, 'BCP008': 2517, 'BCP003': 1377, 'BCP004': 523}
- state: {'Cycling B': 5827, 'Naive': 4613, 'GC': 3382, 'MBC': 1934, 'DZ GC': 1695, 'Plasmablast': 1435, 'MBC FCRL4+': 1176, 'Activated Naive': 640, 'preGC': 574, 'LZ GC': 542, 'Activated MBC': 255, 'prePB': 168, 'Activated MBC FCRL4+': 130, 'FCRL2/3high GC': 107}

## Threadfin runs (parameter grid)

| run | clusters | NMI vs state | ARI vs state | #sig enrichments |
|---|---|---|---|---|
| tf__pca__r0.3__cw0.0 | 1 | 0.000 | 0.000 | 0 |
| tf__pca__r0.3__cw0.3 | 2 | 0.003 | 0.003 | 0 |
| tf__pca__r0.8__cw0.0 | 5 | 0.100 | 0.075 | 4 |
| tf__pca__r0.8__cw0.3 | 7 | 0.029 | 0.015 | 0 |
| tf__umap__r0.3__cw0.0 | 2 | 0.066 | 0.023 | 3 |
| tf__umap__r0.3__cw0.3 | 1 | 0.000 | 0.000 | 0 |
| tf__umap__r0.8__cw0.0 | 6 | 0.078 | -0.017 | 4 |
| tf__umap__r0.8__cw0.3 | 5 | 0.063 | 0.035 | 3 |
| tf__joint__r0.3 | 1 | 0.000 | 0.000 | 0 |

## Robustness and statistical support

- **Joint GEX+BCR embedding** (lam=0.5): latent-vs-GEX distance Spearman **-0.024**, latent-vs-BCR **nan** over 3183 graph edges; modality contribution GEX 1.00 / BCR 0.00
- **Basis robustness** (PCA vs UMAP): clone-level ARI **0.000**, NMI 0.000 over 200 clones. Low ARI means fine-grained community boundaries shift between bases; check whether the biologically significant enrichments replicate across bases.
- **Null 1 (clone-label permutation, within donor)**: observed mean clone state-purity **0.662** vs null 0.489 ± 0.013; p = 0.0050
- **Null 3 (size-matched random communities)**: observed 0 significant state enrichments vs null 0.00 ± 0.00; p = 1.0000
- **Held-out split-clone validation**: sibling halves of the same clone co-cluster at **0.99** ± 0.02 vs chance 0.66 (20 clones x 5 repeats, basis=X_pca)

Runtime: 2.2 min.
