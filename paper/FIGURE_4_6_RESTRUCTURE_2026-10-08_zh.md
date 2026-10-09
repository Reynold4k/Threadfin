# Figure 4–6 重构与 Clonotrace 复核（2026-10-08）

目前最有说服力的主线是：**GC 应用发现问题 → 真实谱系条码检验跨时间联系 → 纵向 TCR 检验可推广性 → 用独立测量和分样实验界定方法增益。** 原 Figure 4 的 non-GC 覆盖和 Figure 5 的跨数据集汇总值得保留，但不足以各自占一张主图，现移至 Supplementary 14、15；人类 GC 的重复采样及结合标签保留在 Supplementary 9。

## 指定论文的逻辑及可借鉴之处

阅读对象为用户提供的 `2025.09.01.673503v1.full.pdf`，即 [Clonotrace v1](https://doi.org/10.1101/2025.09.01.673503)，而非假定其为已发表的 Nature Methods 论文。

其证据顺序是：方法与目标任务 → 模拟数据中已知分支/时间及候选基因真值 → 治疗细胞系的非单调时间过程 → 真实造血谱系条码 → 肿瘤 TCR 应用 → 纵向免疫治疗。值得借鉴的是每张图回答一个更强、可检验的问题。模拟中可观察真值与真实样本的表达相关性不能混称；其 NSCLC 段也明确指出 clone distribution shift 不一定是分化。

本次实际重分析该 PDF 明确给出公开 accession 的两个数据集：

| 数据 | 本次分析范围 | 与原文的区别 |
|---|---|---|
| [GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802) | LARRY in-vitro：130,887 个细胞，49,302 个细胞有唯一谱系标签 | 使用整个公开 in-vitro 数据；不是照搬原文的特定 monocyte/neutrophil 子集、聚类或 OT 坐标 |
| [GSE266219](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266219) | 十个公开患者对象；195,685 个 TCR 注释细胞中，124,534 个有唯一 TRA/TRB 配对 | 原文的临床响应比较选八名患者；本次不猜测八人的选择或响应映射，不作疗效复现声明 |

指定 PDF 的 melanoma treatment 和 glioblastoma 部分未给出可直接复现的 GEO accession。本次没有复现这两个应用，也没有把本次结果描述成完整复现 Clonotrace。下载地址、体积、SHA256 在原始数据目录的 `sources.json`；代码、表格及运行记录随本次交付保留。

## 新主图与实测结果

**Figure 4：条码联系是否能支持早期状态与后续观测之间的关联？**

- 展示作者的细胞 SPRING 背景及 7,575 个 Threadfin clone-day 表征；连线仅连接同一已知条码。
- 预测表征完全单独建立，只使用第 2 天细胞；第 6 天细胞和状态标签不参与该 PCA/表征模型。
- 1,401 个第 2 天多细胞条码中，172 个满足第 6 天至少 4 个细胞的配对条件。3 次五折交叉验证的所有方法使用相同条码划分，训练折内三折选择 ridge 参数。
- 中性粒细胞比例：kernel 的中位折 R² 为 0.199，RNA 均值为 0.163；单核细胞比例分别为 0.163 和 0.099。
- 绝对误差上的增益较小，单核细胞比例的 kernel MAE 为 0.227，RNA 均值为 0.225。不能只展示更有利的 R² 后声称全面占优。
- 这验证的是后续采到的 RNA 定义状态组成，不是完整发育潜能；仅 172 个配对条码、采样选择及同实验内的 transductive 表征限制推广性。

**Figure 5：同一个真实 TCR 克隆跨时点是否仍带有状态信息？**

- 10,428 个 clone-cycle 表征，按患者去背景；不把 cycle 当作需要消去的批次。
- 精确、唯一的 TRA 与 TRB CDR3 氨基酸配对在同一患者内定义生物学克隆。患者编号、周期字符串及来源细胞严格对应；多 TRA/TRB 的歧义细胞不用于联系。
- ≥4 细胞/时点的相同克隆，在 29 个患者–相邻观测周期组合中，与同患者、同周期、捕获数量分箱内的其他克隆比较。
- 跨患者中位距离比为 kernel 0.608、Threadfin mean 0.637、RNA mean 0.660。三者均能检测同克隆状态保持，因此不能把“有信号”当成 kernel 独有。
- 公共数据中确有 C3、C5、C7 等周期，按原标签保留，未强制改成 C1/C2/C4/C6。
- RNA 模块色图解释状态；独立配对身份及置换检验提供不同层面的证据。分布改变也可能来自扩增、收缩或迁移，没有临床响应对应表就不推断治疗获益。

**Figure 6：什么收益来自表达信息，什么收益来自收缩估计？**

- 主体改为同鼠留出 reporter 预测、全部实测 readout 的 R²、分样验证；移除原主图占大面积且证据不足的“Threadfin 独有功能”矩阵。
- 保留全部八种方法/对照与共同评价对象。现有 reporter endpoint 上，简单 RNA 均值仍优于 Threadfin；这必须在图文中直说。
- LARRY 分样中，用不同生物学条码估计方差与特征变换，再在 250 个至少 16 细胞的 clone-day-well 单元上评价；每次预测样本与参考样本互不重叠，关闭邻居平滑。
- 每单元 20 次分样，采 2、4、8 个细胞时，平均误差相对未收缩 kernel mean 分别降低 **27.2%、14.2%、6.5%**。
- 这是收缩估计的实证收益；不是绝对 reliability 数值已经校准的证明，也不等于命运预测能力。

## 对投稿说服力的判断

此次重构比“更多 clone UMAP + 更多数据集覆盖”更有说服力，但尚不能仅凭这些结果支持全面优于现有方法的结论。面向 Nature Methods，主要尚缺：

1. 核验用户另行提供 DOCX 中记载的 **Clonotrace 官方实现直接比较**（302 家族、10 项比较未见显著优势），用户确认仅作参考；是否追加同任务实验由本机可复现证据决定。本次是用 Threadfin 分析其公共数据，不是直接胜过 Clonotrace 的 benchmark。
2. 将核特征、邻居平滑、背景去除和收缩分别拆开的预先定义消融。当前 mean/kernel 对照同时改变非线性特征和平滑，不能归因于单一组件。
3. 独立实验/供者、完整训练折内预处理的归纳式外推；当前报告的转导式分析已经明确限制。
4. reliability 对真实重测稳定性的定量校准，以及配对条码覆盖率、克隆大小和未观察后代的系统敏感性分析。
5. BCR 生物学层面的独立命运/功能验证。采样分布、SHM、二值结合标签与真正的亲和力、分化方向应继续区分。

三臂 Supplementary 13 已补全 GC、PB、Memory 的同坐标对照及逐鼠时间曲线。统计解释、图注、“看图说话”和 Word 导出同步更新；详细验证记录见 `case_studies/results/clonotrace_revision_validation.json`。

## 复现入口

```bash
sbatch case_studies/clonotrace_larry.sbatch
sbatch case_studies/clonotrace_nsclc.sbatch export
sbatch case_studies/clonotrace_nsclc.sbatch analyse
sbatch case_studies/clonotrace_figures.sbatch
```

原主图归档在项目上一级 `internal_validation/paper_673503/figures_before_restructure/`。
成功计算的源作业：LARRY `32524038`；NSCLC 导出 `32524037`，表征 `32524411`（最后汇总报类型错误），汇总续跑 `32524816`。续跑复用了已完成的表征，未重复计算或把失败状态计作完成。

## 10 月 9 日补充：回应 SPRING 与简单统计基线

B/C 的作用是解释细胞视角与克隆视角，SPRING 按 barcode 着色也能显示克隆分布，故不把这组图当作方法优越性证据。Figure 4E 已加 RNA 均值＋方差，使用相同 30 个早期 PC、相同分折和 ridge 调参网格：中性粒细胞中位折 R² 为 0.110，单核细胞为 0.058；kernel 分别为 0.199、0.163。原五种方法的全部 28,380 条预测保持一致。单核细胞 MAE 仍未优于 RNA 均值；不能由此声称总体优越，或把增益单独归因于 kernel。
