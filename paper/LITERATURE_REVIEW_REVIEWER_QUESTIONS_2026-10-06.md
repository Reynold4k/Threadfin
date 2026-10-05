# 2026-10-06 文献审阅：最可能的审稿问题与建议

范围：截至 2026-10-06 的定向公开检索（原始研究、方法论文和官方工具文档），不是穷尽式系统综述；没有运行任何新分析。本清单以稿件现有定位为前提：Threadfin 描述 BCR 定义 family 的**已捕获状态分布**，不声称直接重建祖先/子代、CSR 方向、亲和力或未来 fate。

## 建议在回复审稿人与最终修订中优先处理的十个问题

| 优先级 | 审稿人可能问什么 | 证据与为何重要 | 当前已有控制 / 尚未解决 | 建议的未来验证（不在本轮执行） |
|---|---|---|---|---|
| P0 | **IGH-only family 是否把不同祖先误合并，因而制造跨状态 sharing？** | clonal-family inference 没有真值时本身仍未完全解决；比较研究显示方法、测序深度、突变负荷都会改变 family 及 downstream 统计，且 HC-only 有系统性局限。配对链的大型分析进一步显示仅重链可造成偏差。 | 已有：同鼠 family、exact H+L sensitivity、缺失/多链不强配，且正文已说 exact H+L 会漏掉 SHM-diverged relatives。未解决：主分析仍为 IGH-only，exact H+L 不是完整敏感性曲线。 | 预注册多种 family caller/阈值（Change-O/partis/SCOPer 与当前定义），报告 ARI、family-size/共享率变化；在可用库中以 paired H+L 或条码作误合并上界。主文保留“association within an inferred family”，不要写为确定 ancestry。 |
| P0 | **跨 family 的 UMAP 邻近是否被误读为共同祖先、共同抗原或生物学 convergent evolution？** | Benisse、BiGCN、CoNGA 都学习/检验 sequence–GEX 邻域关联；它们的边或近邻不是 B-cell 系统发育边。clone2vec 也明确是 clone composition 的探索表示，且截至日期仍是预印本。 | 已有：正文反复区分 family membership、profile similarity 与 lineage；表达嵌入排除了 receptor genes。未解决：图注/摘要任何“trajectory/evolution/lineage”措辞都会被当作过度推断。 | 对选定 family 使用 germline-aware SHM phylogeny（IgPhyML/dowser）仅作独立的谱系敏感性图；将跨-family 图的结论限为“similar captured state distributions”。真正的共同表位需抗原结合或竞争实验。 |
| P0 | **GC–PB/memory 共现能否支持 GC 输出、memory re-entry 或方向性？** | 终末 scRNA+scBCR 快照不含时间方向。Mesin et al. 的 fate mapping 发现 recall GC 主要来自无既往 GC 经验 B cells，memory re-entry 仅小部分；因此不能由 shared receptor 或跨日采样替代 fate mapping。 | 已有：正文及 Fig. 3 已拒绝 GC→PB/GC→memory、re-entry 结论；同型置换还给出 GC–PB 无一致正 excess。未解决：读者仍会把“shared-state family”理解为 fate path。 | prime–boost 中分别不可逆标记 GC-derived memory/GC cells，并在 GC、PB、memory 做 paired BCR + 数量学读出；以动物为统计单位、预定义 re-entry 判据。 |
| P0 | **isotype-preserving shuffle 是否控制了足够的抽样结构？** | isotype 与状态、时间、组织、pre-sort gate、library/capture rate 均可相关。sciCSR 说明 CSR 动态需要 productive/sterile IgH 转录本和 Markov model，不能从 isotype 标签或 clone 共现推出 CSR/方向。 | 已有：鼠内置换保留 family size/state totals，第二 null 在重链 isotype 内；每鼠报告；正文明确无 CSR 推断。未解决：同型 null 不控制 library、day、FACS/HTO、clone age 或 annotation error。 | 分层/限制置换于 mouse×day×library/sort×isotype，或用这些协变量的层级模型；同时给每层有效 family 数。若原始 BAM 充足，可独立运行 sciCSR，但其结果只能检验 CSR 动态，仍不能验证 cell fate。 |
| P0 | **reporter gate 与物理分库混杂时，预测/关联是否只是 library effect？** | NP/RBD 输入中 mCherry 或 sorting gate 与物理 library 对齐会使 cell composition、ambient RNA、深度或 batch 成为标签代理；leaving out mouse 但允许 transductive representation 不构成端到端新 mouse 泛化。 | 已有：Methods 中关于 reporter/library 混杂的说明 已直接承认该限制；同一 library shuffle、donor-centred RNA baseline、RBD 待全覆盖后才报告。未解决：现有数据不能把 reporter biology 与 library 完全分开。 | 需要同一 10x/library 内混合 gate 或 cell hashing 后再做 gate-held-out test；在独立实验日/文库重现。结果文字保持“label-held-out, transductive”，不得称 generalisation/SOTA。 |
| P1 | **clone size 和不均匀捕获是否同时驱动 reliability、跨状态概率与表达相似？** | clone-level distribution 的方差强烈随细胞数变化；clone2vec 也将 sparse-clone robustness 作为专门问题，建议以 clone 而非 cell 作 association unit。 | 已有：固定 all-family denominator、最小 2/3 cells sensitivity、reliability、clone-bootstrap、每鼠统计。未解决：不同 state/gate 的 BCR recovery 和 dropout 未被直接测量。 | 做按 clone size、BCR recovery、state abundance、library 的分层/匹配敏感性；binomial/Dirichlet-multinomial measurement model；预先规定小 family 只作 descriptive candidates。 |
| P1 | **donor/sample 层级是否被 cell 或 family 伪重复掩盖？** | scBCR–GEX 关联常被大量细胞放大；家族并非跨 donor 的独立复制。human 首两时间点只有一个 donor 的 SHM 对照，不能做 population maturation。 | 已有：infection 以 mouse 为 replicate、human donor 等权、benchmark leave-one-mouse-out，且已报告早期 donor 数。未解决：多个 cohort 来自同一论文/实验，不是独立 biological replications。 | 所有主效应给 mouse/donor-level effect、CI 和每层 n；用 leave-one-donor/mouse sensitivity，避免 pooled cell P 值；把 1-donor 点标为描述性。 |
| P1 | **“exact H+L across pure PC/memory gates” 是否仍可能是 doublet、index hopping、ambient VDJ 或技术重复？** | paired receptor identity 强化 membership，却不提供时序；scirpy 官方 IR model 也将 multi-chain cells 标为应审查对象。跨 sort library 的重复 receptor 需要库级 QC 才有解释力。 | 已有：仅唯一 productive IGH+IGK/IGL、donor-match、纯 FACS gates；pool/mixed-donor 库排除。未解决：未独立验证每个 shared group 的 cell barcode/UMI、doublet 与 ambient contamination。 | 对 shared groups 报每库 barcode/UMI/contig QC、双细胞分数和跨库复现；选少数 paired antibodies 重表达测 antigen specificity。仍只能称 receptor identity across gates，不能称 PC↔memory conversion。 |
| P1 | **SHM、Spike binding 与 affinity/future output 的边界是否清楚？** | SHM 是 context-dependent 且受选择，普通 phylogeny/序列距离不可直接代表时间或 affinity；GC PC fate 的 affinity evidence 也依赖特定实验系统。Merkenschlager 直接测到 affinity-enhancing variants 与 division/SHM regulation，不能由任意 family 的 mutation load 外推。 | 已有：正文明确 binary binding ≠ quantitative affinity、SHM ≠ affinity，且 reporter/affinity 数据分开。未解决：读者可能把 Fig. 4 SHM curve 或 Spike-positive label 当作 functional validation。 | 对候选配对链表达 mAb，做定量 binding/neutralisation；若研究 mutation order，以 germline-aware model 和已测 affinity variants 校准，并将 affinity 与 fate 实验分开预注册。 |
| P2 | **native tool benchmark 是否公平、可复现，且未把“任务不匹配”误当性能差？** | Dandelion/Scirpy/Platypus 主要覆盖 annotation、repertoire/trajectory workflow；CoNGA 原始验证以 TCR 为主；sciCSR需要原始 IgH 转录本；Benisse/BiGCN 是不同节点与目标。比较不同输出时，单一 division-fraction MAE 无法定义总体排名。 | 已有：METHOD_COMPARISON 已按任务拆分；共同 family/cell 集、RNA baseline、coverage/config、原生模型和 BiGCN 单 run 限制已写；NP 结果诚实显示 Threadfin 未胜 RNA centroid。未解决：RBD 未完成；clone2vec 是预印本，官方 doc 不能等同发表基准。 | 仅在每工具适用任务上作 matched comparison，完整报告版本/commit/seed/coverage/runtime；保留 no-result/不适用，而非填零。RBD 等所有原生输出与共同交集完成后再冻结结果；外部独立 cohort 才能讨论稳健性。 |

## 对 Threadfin 的科学定位建议

最稳妥的定位是：**以独立 BCR family call 为分组单位，量化一个实验/样本内被捕获细胞状态的组成、其可靠性和与外部 reporter/sort 标签的关联；用受约束置换检验 family 共现是否超出观测到的采样结构。** 这补足的不是 ancestry reconstruction、BCR embedding、CSR direction 或 fate prediction，而是这些问题之间的可审计边界。

建议在摘要、Fig. 1/3/4/6 legend 和讨论各保留一次以下限制句式：`shared state occupancy is an association among captured cells within an inferred BCR family; it does not establish ancestry direction, a cell-fate transition, antigen affinity, or recall GC re-entry.`

## 新增/应显式引用的关键来源

完整元数据、状态（published/preprint/software）与 URL 在 `internal_validation/literature_review_sources_2026-10-06.json`。优先加入：Dandelion、sciCSR、Platypus、CoNGA、Scirpy 官方文档、partis、IgPhyML、TRIBAL、clonal-family systematic evaluation、以及 Mesin fate mapping。clone2vec 必须标注为 **2026 bioRxiv preprint**，不能作为“已发表方法论文”。

## 已核对的稿件引用问题（正文与 DOCX 引用核查）

1. 现有 1–8 的题名、年、期刊、卷页/eID 与 DOI 在 PubMed/出版方可核实，未发现 DOI 错误；Skinner et al. 是已发表的 *Nature Immunology* 27(7):1502–1516 (2026)，不是预印本。
2. 现有第 9 条 clone2vec 只写官方 documentation/implementation，且作者字段“Isaev S, Kharchenko lab and Adameyko lab”不符合 bibliography 格式。若引用方法论文，应改为：Isaev S, Erickson AG, Adameyko I, Kharchenko PV. *Clonal embeddings allow exploratory analysis of lineage-resolved single-cell data.* bioRxiv. 2026;2026.04.30.720820. doi:10.64898/2026.04.30.720820，并明确为预印本；若仅为所用软件版本，保留 software citation 但不要称其为发表论文。
3. 当前正文引用仅覆盖研究所用数据与 Benisse/BiGCN/clone2vec；若 Fig. 6 保留 Dandelion/Scirpy/sciCSR/CoNGA/Platypus 的比较，必须在主参考文献或图注加入对应原始论文/官方文档，否则比较表的关键能力无可追溯来源。
4. 更正：Benisse 前六作者应为 Ze Zhang、Woo Yong Chang、Kaiwen Wang、Yuqiu Yang、Xinlei Wang、Chen Yao；clonal-family systematic evaluation 第一作者为 Daria Balashova；BiGCN 的页码/eID 是 **e01919**，不是 e202501919。Platypus 的正确出处为 *NAR Genomics and Bioinformatics* 3(2):lqab023 (2021), doi:10.1093/nargab/lqab023；此前误列的 10.26508/lsa.202000869 属于无关论文。

## P0：与 clone2vec + repertoire caller 组合相比，Threadfin 的新增贡献应怎样写

不能把“克隆 embedding”本身作为新增贡献。clone2vec 已提供基于表达邻域的 clone-distribution 空间、稀疏 clone 的稳健处理、clone–gene association 与跨数据集对齐；而当前 NP division-fraction benchmark 中 Threadfin mean/kernel 也未优于 RNA centroid。因此不应以更好的通用表示学习、稀疏 clone 表征或预测准确度定位 Threadfin。

可被审稿人检验、也与现有结果一致的定位是一个**审计层（audit layer）**：在外部 repertoire caller 固定 family 后，Threadfin (i) 明确把 family 的捕获状态组成与 ancestry/未来 fate 分开，(ii) 给出小 family 的 profile reliability，(iii) 使用 mouse/donor、family size、state total 与 isotype 条件化的 null 来问某个跨状态共现是否超出观测采样结构，(iv) 将 reporter/FACS/probe 作为外部锚定，并公开报告库混杂、阴性同型-null 和 RNA baseline。它回答的是“这一个 inferred family 中的状态共现/偏好在当前抽样设计下是否仍值得成为实验候选”，不是取代 clone2vec 的分布表示或 repertoire caller 的 family/lineage inference。

建议在稿中直接承认：Threadfin 可接在 partis/Change-O/Dandelion/Scirpy 等 caller 后；不同 caller 的稳定性是必需 sensitivity analysis。其可发表价值取决于 reliability、conditional null 与 GC 外部锚定能否改变对候选的解释，而非 UMAP 的新颖性或单一 reporter prediction 排名。

## 续跑复核补充：2026 年相邻方法与配对链证据

- **CoMBCR 已于 2026 年发表**：BCR/RNA co-learning 用于细胞层面功能表示，包含 Spike-binding 标签任务。审稿人会问为何只选 Benisse/BiGCN；建议在任务能力表中加入 CoMBCR，明确其不同输出和未来统一比较所需的 family 聚合与标签持出。本轮没有运行它，也没有填入性能分数。[原文与完整元数据](https://academic.oup.com/bioinformatics/article/42/3/btag115/8512509)。
- **重链 family 的风险应有直接配对链引用**：Wang 等人的大型配对链研究指出 chain-mixed clusters 和 naive-like pseudo-clonal clusters。审稿人可能要求检查轻链一致性及 public receptor 造成的伪扩增；目前的 exact H+L 控制只支持部分成员关系，不能替代主 family caller 的敏感性/特异性评估。[PLOS 原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014077)。
- **稿件状态已再次核实**：clone2vec 仍按 bioRxiv 预印本引用并明确未同行评审；[PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/42146635/)与[原始预印本](https://www.biorxiv.org/content/10.64898/2026.04.30.720820v1)支持当前状态。

两篇新增论文已加入正文引用及 RIS 注册表；原始 Crossref 元数据保存在 `internal_validation/crossref_metadata_2026-10-06_extension.json`。这是定向补充检索，不是穷尽所有公开文章的系统综述。上述科学建议均未在本轮执行。
