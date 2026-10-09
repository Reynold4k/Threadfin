# Native benchmark：可解释范围与当前结果

## 这个 benchmark 测量什么

它测量的是：在不给表示学习阶段输入 reporter/sort 标签的前提下，家族表示是否能在留出整只 mouse 后，由训练 mouse 的 kernel-ridge readout 预测该家族中**已测** mCherry-low、RBD-probe 或 DZ gate 的细胞比例。家族是预先保存、donor-private 的 Threadfin IGH family；每个方法使用相同的非免疫球蛋白 RNA PCA100 输入、相同 family 交集和相同 readout。

它不测量 BCR 家系真值、细胞亲缘方向、GC re-entry、未来命运、亲和力或保护功能。预测的是同一快照中已测 gate 的 family fraction。表示可在无标签条件下看到全部 common cells，因此是 **label-held-out、transductive** 比较，不能写成“对未见 mouse 的端到端泛化”。

## NP-OVA 当前可复核结果

当前 `mouse_np` 的共同主分析集有 7 只 mouse、373 个共同 expanded families，所有七个非空表示均为 100% feature coverage。主分析使用 `min_cells=2`，即 family 至少有两个捕获 cell 且至少两个已测 division-gate cell；每个方法有 7 个 leave-one-mouse-out MAE。

| 方法 | 留鼠 MAE 中位数 |
|---|---:|
| RNA centroid（raw） | 0.1611 |
| RNA centroid（donor-centred） | 0.1654 |
| Threadfin mean | 0.1973 |
| Threadfin kernel | 0.2068 |
| clone2vec | 0.2381 |
| BiGCN | 0.2913 |
| Benisse complete family kernel | 0.3143 |
| Training-mean baseline | 0.3143 |

因此，在这个 NP-OVA division-fraction readout 上，Threadfin mean 或 kernel **没有超过 RNA centroid**。这不是 Threadfin 的 SOTA 证据，也不能推广为对其他 cohort、其他标签或其他任务的方法排序。donor-centred RNA centroid 与 raw centroid 的接近结果也表明，此处不应把 donor centring本身归因于 Threadfin kernel 的独有增益。

`min_cells=5` 是预先指定的灵敏度分析，选择 150 个 family；其 MAE 中位数依次为 raw RNA 0.0977、donor-centred RNA 0.0982、Threadfin mean 0.1076、Threadfin kernel 0.1389、clone2vec 0.1604、BiGCN 0.1985、Benisse 0.1914、training mean 0.2078。它改变的是纳入的 family 集合，不能和主分析混为一个效应量，也不应用来追求对 Threadfin 有利的任务或阈值。

## 为什么有两个 RNA baseline

raw RNA centroid 是家族内 PCA100 的简单均值。donor-centred centroid 先从每个 cell 的 PCA 表示移除 donor 均值，再取同样的 family mean。Threadfin mean/kernel 都以 `context_key=donor` 建立 profile；新增的 donor-centred baseline 将“移除 donor context”的作用与 Threadfin 的 shrinkage、kernel representation 和 smoothing 分开。合理比较是：

- raw 与 donor-centred RNA：donor centring；
- donor-centred RNA 与 Threadfin mean：context-adjusted profile/shrinkage；
- Threadfin mean 与 kernel：kernel representation 和 smoothing。

这些比较不要求 Threadfin 在任何一项上获胜。

## Native 输出与 readout 的边界

Benisse 的性能使用完整的 cell-weighted family kernel；30 维 classical MDS 只保留为输出审计，不参与性能读数。BiGCN 使用官方 pipeline 的单次 native 输出；上游实现没有可固定的随机 seed，所以这个结果是一次已保存 native run，而非其随机性的充分估计。保存的输出和训练折 readout 使同一输入下的 scoring 可复现，但不替代多 seed 或多 cohort 的稳健性评估。

kernel centring、trace scaling、target centring和 ridge alpha 的内层选择都在每个 outer training fold 内完成。Reporter 标签只用于目标构造和训练/评估 readout；未测 RBD/zone labels 保持缺失，绝不记作负值。

## 尚未报告的 RBD 结果

`mouse_rbd` 仍在等待所有七个表示完成并满足共同 family 交集条件。RBD bait 或 zone 结果在完成前不应以部分方法、部分覆盖或填零方式报告。

## 审计位置

每次完整评分应同时检查：

- `coverage.csv`：每种表示的 expanded-family coverage 与共同 family 数；
- `targets.csv`：每个 family 的已测细胞数和 target fraction；
- `heldout_scores.csv`：每个 held-out mouse 的 MAE、R2、样本数、target prevalence、alpha 和超出 `[0,1]` 的预测计数；
- `heldout_predictions.csv`：逐 family 预测；
- `benchmark_manifest.json`：cell/family ID hash、软件版本、native commit/status、表示审计和 scoring contract。

应以每 mouse 的配对 MAE 差异及其中位数/IQR解释结果，不能把大量 family 伪重复地作为独立性能样本。

分选 library 的边界：当前 NP 的 HI/Lo 和 RBD 的 A–D physical library 前缀与 division gate 对应；整鼠留出仍共享这些 library。它检验捕获 gate fraction 的读出，不能单独排除分选/测序 library 的技术影响，也不是独立 library 的泛化或总体分化概率验证。`source_library_gate_counts.csv` 保存该对应关系，标签仍未作为模型特征输入。
