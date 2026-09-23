# Threadfin validation report: flu_gse175522

Seasonal influenza vaccination, sorted PBMC B cells, 6 donors x 2 timepoints (pre, day 7), young vs older adults (Wang et al. 2023, Aging). BCR: author AIRR tables with clone_id.

## Dataset QC

- cells: 123,693; BCR-positive: 84,040 (68%)
- clonotypes: 80,682; expanded (>=3 cells): 499; (>=5): 125
- clone size: median 1.0, max 66

- donor: {'141394': 24811, '141409': 24528, '141415': 24291, '120648': 20344, '120667': 16917, '141393': 12802}
- timepoint: {'d0': 70986, 'd7': 52707}
- condition: {'older': 73630, 'young': 50063}
- state: {'0': 10060, '1': 8472, '2': 7044, '3': 6727, '4': 5508, '5': 5420, '6': 5137, '7': 4999, '8': 4485, '9': 4234, '10': 3708, '11': 3493, '12': 3252, '13': 3179, '14': 3174, '15': 3145, '16': 3102, '17': 3101, '18': 2567, '19': 2553, '20': 2138, '21': 2114, '22': 1936, '23': 1856, '24': 1766, '25': 1763, '26': 1731, '27': 1726, '28': 1548, '29': 1363, '30': 1050, '31': 926, '32': 899, '33': 894, '34': 894, '35': 824, '36': 810, '37': 796, '38': 734, '39': 701, '40': 621, '41': 593, '42': 440, '43': 429, '44': 383, '45': 381, '46': 378, '47': 330, '48': 196, '49': 113}

## Threadfin runs (parameter grid)

| run | clusters | NMI vs state | ARI vs state | #sig enrichments |
|---|---|---|---|---|
| tf__pca__r0.3__cw0.0 | 3 | 0.435 | 0.371 | 33 |
| tf__pca__r0.3__cw0.3 | 4 | 0.059 | 0.031 | 11 |
| tf__pca__r0.8__cw0.0 | 5 | 0.535 | 0.491 | 35 |
| tf__pca__r0.8__cw0.3 | 9 | 0.121 | 0.048 | 25 |
| tf__umap__r0.3__cw0.0 | 5 | 0.483 | 0.416 | 35 |
| tf__umap__r0.3__cw0.3 | 2 | 0.034 | 0.018 | 12 |
| tf__umap__r0.8__cw0.0 | 8 | 0.504 | 0.345 | 39 |
| tf__umap__r0.8__cw0.3 | 6 | 0.099 | 0.056 | 19 |
| tf__joint__r0.3 | 6 | 0.509 | 0.378 | 34 |

## Robustness and statistical support

- **Joint GEX+BCR embedding** (lam=0.5): latent-vs-GEX distance Spearman **0.232**, latent-vs-BCR **-0.828** over 7886 graph edges; modality contribution GEX 0.81 / BCR 0.19
- **Basis robustness** (PCA vs UMAP): clone-level ARI **0.313**, NMI 0.371 over 499 clones. Low ARI means fine-grained community boundaries shift between bases; check whether the biologically significant enrichments replicate across bases.
- **Null 1 (clone-label permutation, within donor)**: observed mean clone state-purity **0.883** vs null 0.356 ± 0.006; p = 0.0050
- **Null 3 (size-matched random communities)**: observed 33 significant state enrichments vs null 8.82 ± 2.87; p = 0.0050
- **Held-out split-clone validation**: sibling halves of the same clone co-cluster at **0.97** ± 0.01 vs chance 0.26 (46 clones x 5 repeats, basis=X_pca)

Runtime: 9.5 min.
