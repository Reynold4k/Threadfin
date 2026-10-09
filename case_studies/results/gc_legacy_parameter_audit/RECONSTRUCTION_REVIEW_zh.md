# GSE246382 旧 notebook clone embedding 计算复现审阅

已从原始表达矩阵重建旧算法，并保留 **164 组真实计算结果**。Figure 2B 使用新计算坐标，没有复用报告截图或手工调整点位。图中为 **377 个同鼠 V–D–J 受体组、762 个细胞**；颜色为 Leiden `clone_cluster`，点面积与实际捕获细胞数成正比。它重建了可辨识的连续 GC/PC 状态布局，但不声称恢复缺失原始对象中的坐标或未来命运。

## 对照范围

- 14 组旧/新 RNA 表示 × 三种分组 × ≥1/≥3 过滤基线，以及不限定 productive 的 IGH 诊断。
- 96 组扫描：三种分组 × UMAP neighbours 10/20/40/80 × min_dist 0.1/0.4/0.65/0.85 × seeds 123/7；其中 3 组与基线重合。
- 7 组明确关闭 HVG mask 的全基因 PCA 对照。
- 50 组新增随机种子：10 个候选配置再加入 0/1/42/99/2024；每个候选合计七个种子。
- 全部参数、坐标、版本、输入散列、job ID，以及 k=5/10/15 的图连接性、k=10 邻域保真度和七种子的 ARI/Procrustes 结果均已保存。

## 主要比较

| 输入/参数 | 点数 | 细胞 | Leiden 组数 | k15 图分量 | 局部 trustworthiness |
|---|---:|---:|---:|---:|---:|
| 旧 RNA + 同鼠序列，≥1 | 487 | 762 | 8 | 2 | 0.9959 |
| 旧 RNA + 同鼠序列，≥3 | 49 | 260 | 3 | 1 | 0.9519 |
| 新 RNA + 同鼠序列，≥3 | 49 | 260 | 3 | 2 | 0.9496 |
| 旧 RNA + 同鼠 VDJ，≥1 | 377 | 762 | 6 | 1 | 0.9937 |
| 旧 RNA + pooled VDJ，≥1 | 217 | 762 | 6 | 2 | 0.9906 |
| 主图：旧 RNA + 同鼠 VDJ，k40/d0.65 | 377 | 762 | 6 | 1 | 0.9939 |
| 严格家族连续布局：k80/d0.85 | 487 | 762 | 6 | 1 | 0.9957 |

连接分量由二维 Scanpy 邻居图定义；它不证明生物学连续性。Trustworthiness 衡量相对输入质心的局部邻域保留，不是外部生物学准确率。

主图沿用已选定的同鼠 VDJ、k=40、min_dist=0.65、seed=123；Leiden 图 k=15、resolution=0.3、seed=0。七种子均为一个 k15 分量、六个分区，ARI 中位数 0.877、最低 0.701；局部保真度最低 0.9938，Procrustes disparity 中位数 0.0198。**它不是扫描中种子最稳定的配置**：同鼠 VDJ 历史 k=20/min_dist=0.4 的 ARI 中位数为 0.989。选择是连续布局与局部保真度的探索性折中，六类不是成功标准。

严格同鼠序列家族也能形成连续布局：保留全部 487 个家族时，k=80/min_dist=0.85 的七种子均连接，ARI 中位数 0.863，局部保真度最低 0.9953。保留 ≥3 细胞则只剩 49 家族/260 细胞，因而不能靠单独调 Leiden resolution 恢复原图的信息密度。

## 输入来源与不可消除的限制

1. 旧代码保留全 AnnData，但 Scanpy 1.10.1 的 PCA **自动使用标记的 5,500 HVG**。这与“全基因 PCA”不同；显式 `mask_var=None` 的对照单独存储在 `allgenes_summary.csv`。
2. 原始 `bigplasma.h5ad`、`result_df.csv` 和 IgBLAST 表缺失。替代输入是公开 TRUST4；762 个 productive IGH 匹配细胞，122 个缺失受体细胞不构造 `nan` 假克隆。历史输出的缺失量和最大 VDJ 组大小因此不能精确匹配。
3. 377 同鼠 VDJ 组中 **83 组有多条不同的 junction nucleotide sequences**。分组不能被视为严格单一谱系。487 个同鼠 V/J/junction-sequence 家族为更严格对照。217 个跨鼠 pooled VDJ 组中 61 个跨多个鼠，仅作历史诊断。
4. UMAP-learn 0.5.5 对应报告；其余为明确记录的 2024 时期依赖组合，不能声称完整恢复作者当年环境。RNA marker 复用了表达数据，二维分支不是未来命运证据。
5. GC selection/output 解释仅适用于本例 NP-OVA model-antigen GC；未延伸到 non-GC。历史 Top2a 原图保留在 S8A，来源与当前计算完全分开。

## 输出与复现

- [Figure 2](../../../paper/figure_plan/Figure_2.png) · [选定计算预览](../../../paper/figure_plan/review/GSE246382_selected_clone_embedding.png)
- [输入流程基线](../../../paper/figure_plan/review/GSE246382_baseline_summary.png)
- [同鼠序列家族参数扫描](../../../paper/figure_plan/review/GSE246382_sweep_legacy_donor_sequence_min1.png) · [同鼠 VDJ 扫描](../../../paper/figure_plan/review/GSE246382_sweep_legacy_donor_vdj_min1.png) · [pooled VDJ 诊断](../../../paper/figure_plan/review/GSE246382_sweep_legacy_pooled_vdj_min1.png)
- [全基因 PCA 对照](../../../paper/figure_plan/review/GSE246382_allgenes_summary.png) · [七种子布局](../../../paper/figure_plan/review/GSE246382_selected_seed_comparison.png)
- [全部数值审计](all_runs_audit.csv) · [种子稳定性](seed_stability.csv) · [选定参数](selected.json) · [独立复算核验](independent_verification.json)

计算脚本：`case_studies/reproduce_gc_legacy_embedding.py`，阶段为 `prepare`、`baseline`、`sweep`、`all-genes`、`stability`、`audit`、`verify`、`panels`。独立核验直接执行 pandas 质心 → scipy Euclidean 距离行 → UMAP → Scanpy/Leiden，不调用 Threadfin reclustering 封装。

成功作业：准备/基线 32345445；三组扫描 32345446–448；全基因和七种子扩展 32345955；新 wheel 兼容性 32345957；独立公式重算 32370571。日志位于 `../internal_validation/cluster/logs/`。失败的 32345119 已由类型安全的 donor/VDJ 连接修复，缺失受体保持缺失。

独立复算已通过：坐标最大绝对差 4.77×10⁻⁷（CSV 序列化精度），Leiden 分区 ARI=1。受体组细胞数、门控比例和 marker 均值也逐组与底层细胞表核对一致。初次独立检查 32370281 使用 float32 均值而与包的 float64 输入约定不同；修正精度后 32370571 通过，未修改入图坐标。

当前定量注释以 `selected_cluster_annotations.csv` 为准，并已与底层细胞表核验；较早的 `cluster_annotation_summary.csv` 使用不同汇总，不用于本次图注或正文。
