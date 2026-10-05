# 同类工具的任务、成熟证据与比较边界

核查日期：2026-10-05。这里比较的是公开配对 scRNA/scBCR 分析工具，不宣称穷尽所有 V(D)J 重建、序列分析和单细胞整合软件。能力来自下列原始论文与官方代码；**能力表不是实际运行的性能分数**。单个任务没有适用输出，不能计为性能为零。

| 工具 | 最适合的问题 | 已有验证 / 官方实现 | 与 Threadfin 比较时需要统一的单位 |
|---|---|---|---|
| Benisse | 用表达信息细化 BCR clonotype 的潜在距离与网络 | 原文分析 13 个配对数据集，并检验受体与表达距离；官方提供预训练 CDR3 编码器及 R 图学习模型。[原文](https://www.nature.com/articles/s42256-022-00492-6)、[代码](https://github.com/wooyongc/Benisse) | 原生节点为 exact V–CDR3–J clonotype；映射到固定、同鼠 sequence family 后比较，不能把网络边直接当共同祖先。 |
| BiGCN | 将表达和 BCR 图融合为表示 | 双图模型及公开训练代码。[原文](https://doi.org/10.1002/smtd.202501919)、[代码](https://github.com/Lxc417/BiGCN) | 原生 clonotype 节点、Benisse 的 BCR 编码、crude V/J 图及两套 cosine 图都应保留。 |
| CoNGA | 受体邻域和表达邻域的共同结构 | 原始验证以 T 细胞为主；当前官方支持 `human_ig` / `mouse_ig`，也输出 clonotype 表达和受体降维及邻域关系。[原文](https://doi.org/10.1038/s41587-021-00989-2)、[代码](https://github.com/phbradley/conga) | 配对链、exact clonotype 及 GEX/TCRdist 结构；不能沿用旧稿“没有克隆层面输出”的判断。 |
| Ibex | 重/轻链 CDR3 的序列表示，并接入表达 WNN | 官方提供 encoder、geometric 和表达相关模型；可进入 Seurat/SCE。[代码](https://github.com/BorchLab/Ibex)、[论文分析代码](https://github.com/ncborcherding/Ibex.manuscript) | 细胞层面的受体表示需要明确后续 WNN 和 family 聚合步骤；不能将仅运行 geometric 编码称作整个联合模型。 |
| Dandelion | V(D)J 注释、克隆网络及发育关联 | 原文验证 V(D)J 特征空间和发育轨迹，支持 BCR/TCR，包含序列及 GEX 接口。[原文](https://www.nature.com/articles/s41587-023-01734-7)、[代码](https://github.com/tuonglab/dandelion) | 真正的原生 V(D)J feature/pseudobulk 工作流与 clone-state distributions 是不同输出；旧版自制 V/J one-hot 不能标为 Dandelion。 |
| Scirpy | 克隆定义、网络、表达图上的 clonal modularity | 官方已有 BCR 网络聚类教程及 clonotype modularity 的大小匹配图零模型。[BCR 教程](https://scirpy.scverse.org/en/v0.22.1/tutorials/tutorial_5k_bcr.html)、[modularity API](https://scirpy.scverse.org/en/latest/generated/scirpy.tl.clonotype_modularity.html) | modularity 是单个克隆的表达图连通性分数，不是与 family profile 同维度的嵌入；应在适用的 clonal resemblance 任务中比较。 |
| sciCSR | 结合 productive/sterile IgH 转录本估计 class-switch 动态 | 原生 CSR 分析及表达/VDJ 整合。[原文](https://doi.org/10.1038/s41592-023-02060-1)、[代码](https://github.com/Fraternalilab/sciCSR) | 需要 BAM/转录本位置等额外输入。常规过滤后 gene-count 矩阵无法替代这些输入；不可仅用 isotype 标签冒充 CSR 推断。 |
| scRepertoire | 克隆配对、扩增、跨样本追踪及 Seurat/SCE 整合 | 第 2 版含实际速度/内存评估及应用实例，可衔接 Ibex 等工具。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12204475/)、[官方](https://github.com/BorchLab/scRepertoire) | annotation/sequence-clustering 与 joint latent learning 不同；“整合”也包括可靠的 metadata 接入。 |
| Platypus | 受体特征、转录状态、SHM 和 repertoire 的统一工作流 | 原文有 COVID-19 配对数据应用。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046018/)、[代码](https://github.com/alexyermanos/Platypus) | 统一输入、QC 与克隆定义后比较实际共同输出；不因没有本包特定 profile 统计量而认定不能整合。 |
| clone2vec | 由表达邻域学习整个 clone 的表示 | 原生 skip-gram clone embedding；主要面向与 RNA 配对的 lineage tracing，不是专门的 BCR ancestry caller。[官方文档](https://clone2vec.readthedocs.io/en/latest/) | 可接收固定 BCR family 标签比较 clone representation；其原始条码来源与 BCR 推断家族要明确区分。 |
| Threadfin | family 内捕获状态分布、可靠性、上下文条件检验、重复采样状态保持 | 本稿的 reporter / GC infection / repeated human GC / pure-gate validation，并报告低覆盖和阴性结果 | sequence-only family、context-centred kernel distribution、reliability；未来命运、GC 再进入和定量 affinity 不属于已验证输出。 |

## 公平比较的关键决定

1. 不混为一张“最佳方法”排行榜。区分 ancestry calling、joint representation、clone resemblance、CSR/sequence trajectory；每个工具在其适用任务里评价。
2. 表达特征排除免疫受体基因；固定同鼠家族和同一细胞集合，避免某工具扩大 clone 后自动提高一致性。
3. 用独立 reporter/FACS 标签评价表达表示；保持整个 donor 在测试折，避免把同一 clone 的细胞分进训练和测试。
4. 成熟工具的全部原生模型要真实运行。只运行序列编码器、接口 import 或作者外自制代理不能代表原生工具。
5. 同时报告 coverage、配置、运行时间与样本范围；小模型和不同输出单位的运行成本不能外推为全规模优势。使用共同覆盖交集时也保留各方法损失了多少 cells / families 的记录。
6. 无直接 fate tracing 的公开数据不能评价真实 GC 再进入或最终分化准确率；本稿 benchmark 评价的是捕获状态解释力。

Figure 6 借鉴 [Yan 等的 benchmark 论文](https://www.nature.com/articles/s43588-026-00977-z) 的任务分解、数据集概览、性能与适用性并列呈现方式。其空间配准指标不适用于 BCR，因此不移植其排行榜或数值。
