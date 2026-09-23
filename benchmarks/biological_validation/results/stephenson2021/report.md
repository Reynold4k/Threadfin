# Threadfin validation report: stephenson2021

COVID-19 PBMC, 5k BCR+ B-lineage cells (Stephenson et al. 2021, Nat Med; via scirpy). Reference state: initial_clustering (B_cell/Plasmablast).

## Dataset QC

- cells: 5,000; BCR-positive: 4,997 (100%)
- clonotypes: 4,823; expanded (>=3 cells): 22; (>=5): 11
- clone size: median 1.0, max 20

- donor: {'MH9143277': 437, 'MH9143275': 338, 'AP8': 295, 'MH9179826': 290, 'MH9179822': 283, 'MH9143424': 282, 'MH9143421': 247, 'MH9143273': 247, 'MH9143426': 244, 'MH9143420': 232, 'MH9143321': 226, 'MH9179824': 225, 'AP5': 212, 'MH9143325': 208, 'AP4': 203, 'CV0171': 197, 'MH9179823': 166, 'AP12': 154, 'MH9143323': 144, 'CV0068': 83, 'CV0284': 81, 'CV0231': 61, 'CV0262': 58, 'CV0155': 57, 'CV0201': 30}
- state: {'B_cell': 4077, 'Plasmablast': 923}

## Threadfin runs (parameter grid)

| run | clusters | NMI vs state | ARI vs state | #sig enrichments |
|---|---|---|---|---|
| tf__pca__r0.3__cw0.0 | 2 | 0.615 | 0.772 | 2 |
| tf__pca__r0.3__cw0.3 | 1 | 0.000 | 0.000 | 0 |
| tf__pca__r0.8__cw0.0 | 3 | 0.417 | 0.357 | 3 |
| tf__pca__r0.8__cw0.3 | 3 | 0.415 | 0.352 | 3 |
| tf__umap__r0.3__cw0.0 | 2 | 0.683 | 0.820 | 2 |
| tf__umap__r0.3__cw0.3 | 2 | 0.743 | 0.853 | 2 |
| tf__umap__r0.8__cw0.0 | 3 | 0.395 | 0.308 | 3 |
| tf__umap__r0.8__cw0.3 | 3 | 0.478 | 0.347 | 3 |
| tf__joint__r0.3 | 1 | 0.000 | 0.000 | 0 |

## Robustness and statistical support

- **Joint GEX+BCR embedding** (lam=0.5): latent-vs-GEX distance Spearman **0.144**, latent-vs-BCR **0.506** over 985 graph edges; modality contribution GEX 0.73 / BCR 0.27
- **Basis robustness** (PCA vs UMAP): clone-level ARI **0.944**, NMI 0.895 over 75 clones. Low ARI means fine-grained community boundaries shift between bases; check whether the biologically significant enrichments replicate across bases.
- **Null 1 (clone-label permutation, within donor)**: observed mean clone state-purity **0.973** vs null 0.817 ± 0.022; p = 0.0050
- **Null 3 (size-matched random communities)**: observed 2 significant state enrichments vs null 0.54 ± 0.89; p = 0.2736
- **Held-out split-clone validation**: sibling halves of the same clone co-cluster at **1.00** ± 0.00 vs chance 1.00 (10 clones x 5 repeats, basis=X_pca)

Runtime: 0.9 min.
