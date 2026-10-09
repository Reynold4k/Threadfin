# Figure 2–3 生物学逻辑与参数审阅（2026-10-07）

**结论：Figure 2D 有可信的生物学解释价值，但目前 C–E 主要证明家族表达表示与实测分裂表型有关，尚不足以证明 Threadfin 比简单 RNA 家族均值更有方法学增益。RBD 重算正确，建议主图保留既有默认参数；更展开的布局作为敏感性对照。Figure 3 已将 C 放在中央并显著放大，E 保留为纯 PB 家族内部的状态描述，F 改为逐鼠、分治疗组的 GC 富集家族比例。**

## 1. Figure 2 C–E 能说服到哪一步

| 面板 | 有效信息 | 当前证据的边界 |
|---|---|---|
| C：RBD 细胞 UMAP，36,188 个细胞 | 交代实测 mCherry 分裂门在细胞表达空间中的分布，为家族图提供背景 | 本身不是 Threadfin 输出，不能承担方法增益的论证；分选门来自分别建库的细胞，门效应与文库效应需区分 |
| D：381 个可靠家族的分裂门占比 | 同鼠受体定义家族的 RNA 表示与独立记录的分裂门有关；该关系跨参数保持 | 分裂门没有用于拟合表示，但与文库设计相连；家族占比不意味着所有成员共享完全相同的分裂历史，也不验证选择后的命运 |
| E：相同坐标上的 V 区 SHM | 当前表达状态与累计突变历史并非同一读出；全局线性突变轴较弱 | 不能写成“SHM 无关联”或“阴性对照”：70/85 个配置仍有名义显著的局部 SHM 关联 |

**最关键的来源修正：**H2B-mCherry NP/RBD reporter 为 **GSE287123**；Figure 2B 的 **GSE246382** 是另一项 NP-OVA day-14 分选研究。B 的 377 个点是 donor-restricted V–D–J receptor groups，其中 83 组包含不同 junction；D/E 的 381 个点是更严格的同鼠、IGH junction 序列相似性定义家族。两者不能统称为同一种克隆，也不能把 B 的分支直接对应到 D 的分裂门。B 的状态富集提供选择相关的解释，不构成无监督重建完整 GC 通路或未来命运的证据。

## 2. RBD 参数审计：计算正确，外观变化没有改变主要解释

重建链为：productive IGH → donor-private V/J 与 junction-length 分层内的 junction 序列家族 → 去除受体基因的 RNA 表示 → donor context 的 kernel family profiles → reliability ≥0.5。最终为 **381 × 30** 的家族表示。默认 UMAP 与保存主图坐标的两轴相关均近 1，最大绝对误差分别为 **4.73×10⁻⁷、1.19×10⁻⁷**。这里审计的是家族表示的可视化参数，不是重新定义家族或发现细胞分化轨迹。

| 用途 | neighbours / min_dist / spread / seed | 距离保真度 ρ | 分裂局部 excess EV | DZ 门局部 excess EV | SHM 局部 excess EV |
|---|---|---:|---:|---:|---:|
| **建议主图：保留既有配置** | **15 / 0.1 / 1 / 0** | **0.728** | **0.251** | **0.172** | **0.097** |
| 几何保真度较高的对照 | 15 / 0.9 / 1 / 0 | 0.784 | 0.282 | 0.195 | 0.106 |
| 较连续、展开的对照 | 50 / 0.5 / 1 / 0 | 0.777 | 0.276 | 0.199 | 0.139 |

这里保真度是抽样点对在原特征空间与二维图中距离的 Spearman 相关，**不是局部邻居保持率**；excess EV 是 kNN 读出 explained variance 减去置换均值，不能直接解读为总解释方差。原始扫描字段名 `knn_r2` 使用的是 `1 − Var(y−预测值)/Var(y)`，严格说是 EV，不是标准预测 R²；标准 R² 使用残差均方并惩罚平均预测偏差。新参数对照图和表已纠正此标注，保留原扫描表以供追溯。三个展示配置的分裂标准 kNN-LOO R² 已从保存坐标核算，依次为 0.171、0.199、0.191；与 EV 的差异较小，不改变推荐。这里按家族留一，不是整鼠留出验证。DZ 门只有 87 个具备该测量的家族，未测量家族不设为零。全扫描的最高保真度为 0.786（15 / 0.5 / 0.5 / 0）；它与表中两个几何对照很接近，没有独立证据表明其生物学结论更强。

85 个配置中，分裂线性 R² 为 **0.142–0.170**，局部 excess EV 为 **0.212–0.282**，85/85 达到名义 p≤0.05；42 对 seed 0/1 配置的线性 R² 差值绝对值中位数仅 **0.002**。SHM 的线性 R² 为 **0.008–0.040**，但局部 excess EV 为 **0.014–0.158**，70/85 达到名义 p≤0.05。因而“全局线性 SHM 梯度较弱”可靠，“SHM 不编码任何状态信息”不可靠。

本扫描只有 200 次置换，p 值为探索性、未经参数选择校正的名义值。置换按 donor 分层，不能自动排除同鼠分选文库影响。扫描包含 42 组 seed 0/1 配对，默认配置另有 seed 2，证明的是读出指标的有限种子敏感性，不是所有点位或枝形完全稳定。RNA 模块分数复用建图表达数据，不能和实测门混作独立验证；该扫描也没有证明蛋白与 mRNA 两臂各自重现所有二维关联。文稿中已有的两臂 within-library 全特征空间检验是另一项分析。

**图的推荐顺序：**默认图保留为主要报告，以避免根据读出挑选最漂亮的配置；若强调几何可读性，可明确采用 15 / 0.9 的敏感性图。不能仅凭它的颜色更顺滑，就声称生物学证据增强。三组图中，同一批家族均展现较稳定的分裂结构和较弱的全局 SHM 结构。

- [三配置 × 三实测读出对照图](../case_studies/results/mouse_rbd_embedding_audit/review/parameter_comparison.png)；[PDF](../case_studies/results/mouse_rbd_embedding_audit/review/parameter_comparison.pdf)
- [指标与参数](../case_studies/results/mouse_rbd_embedding_audit/review/parameter_comparison.csv)、[完整审计摘要与输入哈希](../case_studies/results/mouse_rbd_embedding_audit/review/review_summary.json)
- [复核与绘图脚本](../case_studies/review_rbd_embedding.py)，原计算入口 [mouse_rbd_embedding_audit.py](../case_studies/mouse_rbd_embedding_audit.py)

## 3. Nature Methods reviewer 最可能追问什么

期刊要求新方法经过充分验证，并与可用方法作严格比较；有解释力的应用图是必要支撑，不能替代方法增益证据。[Nature Methods aims and scope](https://www.nature.com/nmeth/submission-guidelines/about/aims)、[What makes a Nature Methods paper](https://www.nature.com/articles/s41592-022-01558-4)。

当前已有一个不能回避的基线：完成的 native benchmark 中，RBD division 的 min-2 家族、10 只整鼠留出读出（1,414 家族），RNA centroid 的 median MAE 为 **0.201**，donor-centred RNA 为 **0.214**，Threadfin mean 为 **0.225**，Threadfin kernel 为 **0.235**；该端点上不能宣称 Threadfin 胜过简单均值。此 benchmark 的家族入选规则与 D/E 的可靠性筛选不同，不能把 1,414 与 381 混用；其表征读出设计也不能夸大为完全归纳的新鼠泛化。

因此，建议把核心主张收敛为“统一受体身份、RNA 分布、上下文与采样可靠性的可审计家族分析”，再用证据说明哪些环节产生可测量增益。最值得补强的不是再调一张 UMAP，而是：

1. 在相同家族、相同输入、相同鼠/文库拆分下比较 kernel profile 与简单 RNA centroid；若指标没有优势，如实说明适用任务差异。
2. 展示可靠性是否校准采样误差：细胞下采样或独立分样中的家族表示稳定性，以及均值无法捕捉的分布差异。不能仅按 reliability 删点后宣称有效。
3. 将实测分裂/GC-zone 的定量比较落实到小鼠与文库层面，并报告效应量；参数扫描的名义 p 值不作为新增独立样本。
4. 对 Figure 2B 同时报告严格序列家族敏感性，防止较宽 V–D–J 组合产生的混合状态被当成真实克隆分支。

若以后重构 Figure 2 C–E，逻辑更强的三个角色是“实测 GC-zone 锚点 → 分裂家族图 → 同条件下对基线的逐鼠定量增益”，SHM 留作正交历史读出。本轮没有把尚未完成的基线比较伪装成新主图；保留 C–E，纠正其来源和解释。

## 4. Figure 3：已实现的重排与 E/F 决定

- **A 为紧凑的设计图**：保留用户补充的两个按真实日期间隔绘制的时间轴、早期逐日鼠数及晚期治疗条；统一蓝/绿浅色背景、左侧实验流程、字号和留白。终点取脾，每个日期是不同小鼠。
- **B 放大**：801 个可靠早期家族，按 PB 细胞占比展示输出偏向。
- **C 放在整页中央且绘图区最大**：1,183 个可靠晚期家族；保留已选 k=50、min_dist=0.5、spread=1、seed=0 的真实坐标，140×88 mm 绘图区。GC、PB、memory 标注表示状态富集，不能解释为分化方向。D 的细胞底图移到上方右侧作注释背景。
- **E 保留，但重新限定结论**：329 个纯 PB 家族仍有 cycling 程序差异。10 只具备 ≥5 个此类家族的小鼠内，距 GC 参考区与 cycling 分数的相关方向均为负，中位 ρ=−0.596；按天数量为 d7=1、d10=227、d14=101。它说明单一 PB 标签没有概括全部 RNA 状态差异。二维距离与模块分复用 RNA，尚不能独立验证成熟时间、GC 来源或“新近输出”。删去依赖任意近端阈值的 36%→82% 主张。
- **F 换成直接支撑 C 的逐鼠捕获组成**：36 只感染鼠、1,174 个可靠家族，盐水/给药分开；GC 富集定义为家族中 ≥50% 捕获细胞标为 GC，分母是该鼠所有可靠家族，点为鼠、线为中位数。其余 9 个家族来自 4 只 naive 对照，保留在 C 的全图而不纳入 F。晚期仍捕获 GC 富集家族；这是终点队列组成，不能写成同一个克隆持续到 d42，也不据此宣称药物因果效应。

原 F 的跨早晚队列程序条形图把状态组成、采样富集与程序强度混在一起；旧图注又描述了另一张共占据置换图。新 F 统一图、代码和文字。严格同鼠/同型匹配的 co-occupancy null 结果仍保留在正文及已有源表中，包括校正后没有一致正向 GC–PB excess 的结果；没有用更美观的图覆盖该限制。

**整图能支持的主线：**Threadfin 把受体定义家族映射到感染中的捕获状态分布，并进一步显示同一细胞注释内部的家族差异；时间/治疗标签提供解释背景。它没有证明家族间祖先关系、GC 选择节点、GC 回流或未来两个命运。模型抗原中的选择相关解释不移植到感染图。

## 5. 可复现文件

- 当前主图：[Figure 2](figure_plan/Figure_2.png)、[Figure 3](figure_plan/Figure_3.png)、[Figure 3 PDF](figure_plan/Figure_3.pdf)。
- 新布局与 E/F 计算：[figure3_infection.py](figure_plan/figure3_infection.py)；入口仍为 `paper/figure_plan/make_figures.py --figures 2 3`。
- [E/F 源表与解释统计](../case_studies/results/figure23_review/figure3_interpretation.json)，同目录保存逐家族与逐鼠 CSV；[figure_audit.json](figure_plan/figure_audit.json) 保存 C 坐标来源、哈希和布局尺寸。
- 同步阅读：[看图说话](看图说话.md)、[图注](FIGURE_LEGENDS.md)、[文稿源文件](MANUSCRIPT_source.md)。

本轮未重做细胞预处理、未篡改保存坐标、未改动 supplementary 图片。目录内旧的 benchmark 计划/历史评议可能仍记录作业未完成；本审阅的方法比较使用已完成的 [readout_summary.csv](../case_studies/results/native_benchmark/readout_summary.csv)，不沿用旧状态或旧排名。
