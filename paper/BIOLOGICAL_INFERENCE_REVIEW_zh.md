# 生物学推断审阅：以 GC 为主线的证据边界

本文档供作者核对主张、图注和后续验证设计。它不是对算法能力的排他性评价：其他受体、系统发育或联合多组学方法也能回答部分克隆问题。这里的重点是当前数据实际允许的生物学表述。

## 推荐主线

稿件应围绕一个窄问题组织：**在实验上可锚定的 GC 反应中，BCR 定义的家族如何分布于已捕获的细胞状态；这种分布何时只是注释重现，何时构成可供验证的候选关系？**

四个问题需要四种不同证据，不能由同一张 UMAP 代替。

| 生物学问题 | 当前最强证据 | 可说的结论 | 不能说的结论 |
|---|---|---|---|
| GC 中近期分裂/区室经历是否和 clone profile 有关？ | Merkenschlager NP-OVA、RBD protein、RBD mRNA 的 H2B-mCherry、RBD bait 或 LZ/DZ FACS 门 | 在各独立 reporter/sort cohort 中，家族的**捕获状态分布**与已测 division gate/zone 有关。 | 单个细胞未来会分裂、回到 LZ 或退出 GC；蛋白与 mRNA cohort 可直接合并。 |
| 一个家族是否在同一感染鼠、同一终末时间点同时出现 GC、PB、memory-like 注释？ | GSE286215 的同鼠 paired scRNA/BCR、HTO，按鼠置换 | 可报告同日同鼠的受体关联状态共现，并据此选择候选家族。 | GC→PB、GC→memory、memory GC re-entry、跨日同一克隆的连续轨迹。 |
| 同一人中 GC-containing 家族能否在多个采样时间重复出现？ | 人 LN FNA/血液的重复取样 | 重复**捕获**到同一供者同一 BCR 家族的 GC 成员，支持观察到的 persistence。 | 某个 memory 细胞再进入 later GC，或某个 GC 细胞产生一个指定 plasmablast。 |
| 两个清楚的终末分选门之间，受体身份是否真的可一致？ | GSE253857 donor 1681/1684 的 pure BM PC、pure BM memory、blood memory；唯一 productive H+L | 完全相同 H+L 受体可在同一 donor 的不同纯 sort gate 中被检出。 | PC 与 memory 的因果转化、双向分化或共同前体的时间顺序。 |

因此，Fig. 2 是主线的实验锚；Fig. 3 是同鼠感染背景下的候选关系及其负对照；Fig. 4 是重复采样与独立 binding 标签；Fig. 5 汇总所有模型的 clonal expression signal 和合理 shuffle；Fig. 6 对比同类工具的任务及实际可比测量。骨髓的“受体身份跨纯门一致”验证移入 Supplementary 5。其余数据集留在 `tested_datasets/`，用于覆盖范围和失败/弱证据展示，不承载 GC 命运论证。

## Plasmodium：应突出的重要负发现

`case_studies/results/clone_state_sharing/biological_summary.csv` 的主统计量以每只鼠为单位，分母固定为该鼠所有至少两个 BCR-bearing cell 的 THREADFIN family；置换保持 family size 与状态总数，第二个 null 还在重链 isotype 内置换状态。这个设计检验“同一 family 中状态共同出现”是否超出可由鼠内采样、family size 和 isotype composition 解释的水平。

对 GC+PB，**没有跨阶段、跨治疗条件一致的正 excess**：

| 队列/条件 | observed equal-mouse rate | isotype-null rate | observed − isotype-null |
|---|---:|---:|---:|
| Exp1 early（19只可用鼠） | 0.0386 | 0.0882 | −0.0497 |
| Exp2 artesunate metadata arm（18只） | 0.0448 | 0.0573 | −0.0125 |
| Exp2 saline arm（18只） | 0.1191 | 0.1737 | −0.0546 |

这不是“没有看到任何 GC/PB 共享家族”。原始聚合计数仍为 Exp1 39 个、Exp2 145 个 THREADFIN GC+PB shared families；严格 exact H+L 也有 Exp1 11/437、Exp2 41/541 个共享 group（`exact_receptor_sharing.csv`）。但这些计数本身会随 clone size、状态丰度、isotype 和 BCR 捕获率升高。经 equal-mouse、同型 null 后，结果不支持把 GC+PB 共现写成普遍正富集或一个独立的 fate bifurcation 信号。

这应成为稿件的强处：系统把“画图中看起来连接的 GC/PB clone”与“超过合理采样模型的关联”分开。正确表述是：**观测到受体相关的 GC/PB、GC/memory-like 和 memory-like/PB 候选；在当前数据和同型控制下，GC+PB 没有一致正富集。** GC+memory-like 尤其稀少（Exp1 2 个 THREADFIN groups；Exp2 33 个），不应被包装为 early memory 机制。

可展示候选，但候选图必须同时给出：同一 mouse、同一天、每种注释的细胞数、THREADFIN family ID、exact IGH 与 exact H+L 是否一致，以及该候选只是用于后续实验。`selected_clones.csv` 中有此类小 family；选择规则必须独立于 embedding 位置。严禁把候选称为“GC re-entry”、谱系树、亲子关系或真实命运。

## 骨髓：设计校正与可保留证据

GSE253857 文件名数字不是 donor ID。manifest 校正后，主 profile 只保留 15 个 single-donor/single-tissue paired libraries、7 个已知 donor；pool、cross-donor 和 mixed-tissue libraries 默认排除。尤其 `BM_6_PC_Bmem` 是 donor 561，不能写成 donor 6；`BM_PCs-Bmem_3_556-blood555` 同库混合 donor 556 的 marrow 与 donor 555 的 blood，不能用于 donor-level 比较。

最干净的验证不是全库 clone-profile retention，而是 `bone_marrow_pure_gate_check/donor_validation.csv`：每细胞恰一条 productive IGH 与恰一条 productive IGK/IGL，缺失或多 contig 不强行配对。pure BM PC 对 pure BM memory 的 exact H+L 共享为 donor 1681 的 **225/862** eligible receptor groups（432 PC、257 memory cells）和 donor 1684 的 **31/677**（57、32 cells）。BM PC 对 blood memory 分别为 **124/1,012** 与 **115/934**。这些是跨清楚 FACS 门的身份一致性，不是 fate 验证。

Supplementary 5 应用“sort gate”而非“cell fate”标记轴；克隆颜色限定 pure PC 与 pure memory，其余门另标；exact identity 检验以 donor 为单位。PC+Bmem combined gate、antigen-first library 与 pool/mixed library可作为透明的敏感性/覆盖信息，不能制造 PC-versus-memory 生物学效应。

## mouse gene set 的措辞与质量控制

鼠数据的 signature 必须作为模块化表达描述，而不是物种间已经等价的细胞命运标签。当前流程以 human gene set 的大小写转换产生鼠符号；这不等于逐基因同源物审核。每个鼠 dataset 应在图注或源数据中给出 requested/present genes，并且：

1. 将“GC / DZ / LZ / plasma / memory signature”写为“以该模块为主的表达分数”；
2. 对缺失、非一对一同源或物种差异显著基因，不把低分解释成缺乏该生物学过程；
3. 不用同一 `log_norm` 矩阵建立 clone profile 后，又把同一 gene set 分数当作 profile/UMAP 的独立验证；它只能解释 profile 的表达组成；
4. 优先让 reporter gate、FACS gate、抗原 probe、BCR isotype 和独立功能测定承担外部锚定。

## 最有信息量的后续验证

1. **抗原特异候选的真实命运实验**：在 PcAS 中以 parasite-antigen bait 与 GC/PB/memory gates 分选，测 paired BCR 并表达候选单抗测结合/功能；这把“作者注释”推进为 antigen-specific readout，但仍不单独解决时间方向。
2. **prime–boost fate mapping**：先不可逆标记已形成的 memory 或 GC-derived memory，随后同源/异源 boost，在 GC、PB 与 memory compartment 检测标记细胞和 BCR。这是检验 memory re-entry 的最低设计标准，而不是在终末感染 UMAP 中寻找箭头。
3. **条形码/谱系记录与多时间点终末取材**：与抗原特异分选结合，预注册 GC→PB、GC→memory 的方向性判据；统计单位为动物，分选/测序深度为协变量。
4. **对骨髓候选作配对功能验证**：对 donor-matched exact H+L shared groups 表达抗体、检测 antigen specificity；要解释 PC/memory 的关系，需加入可追踪的来源或连续招募设计，不能由 shared receptor 推断。

## 关键来源

- Merkenschlager et al., *Nature* 2025, reporter division/SHM GC 模型，doi:[10.1038/s41586-025-08728-2](https://doi.org/10.1038/s41586-025-08728-2)。
- ElTanbouly et al., *J Exp Med* 2024，NP-OVA LZ/PC sort 与 affinity，doi:[10.1084/jem.20231838](https://doi.org/10.1084/jem.20231838)。
- Skinner, Asad et al., *Nature Immunology* 2026，PcAS 时序、HTO 与 treatment，doi:[10.1038/s41590-026-02563-x](https://doi.org/10.1038/s41590-026-02563-x)；对应 GEO [GSE286215](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE286215)。
- Kim et al., *Nature* 2022，纵向 LN GC 样本，doi:[10.1038/s41586-022-04527-1](https://doi.org/10.1038/s41586-022-04527-1)。
- Ferreira-Gomes et al., *Nature Communications* 2024，BMPC/Bmem sort 设计，doi:[10.1038/s41467-024-48570-0](https://doi.org/10.1038/s41467-024-48570-0)；对应 GEO [GSE253857](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE253857)。
- Mesin et al., *Cell* 2020，memory GC re-entry 的 fate-mapping/reboost 对照，doi:[10.1016/j.cell.2019.11.032](https://doi.org/10.1016/j.cell.2019.11.032)。
