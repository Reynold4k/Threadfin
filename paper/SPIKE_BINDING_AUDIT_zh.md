# Kim 2022 Spike 标签与旧曲线审计

## 结论

旧图可以恢复为「作者标注的 Spike 阳性克隆在 Threadfin 表达 programme 中的富集」，但不能恢复为「Spike+/− 亲和力曲线」。旧实现位于 `paper/figure_plan/legacy_panels.py:1063`：它读取 `programme_associations.csv` 的 `spike_binding/S+` 行，横轴明确是 *odds of being a spike-binding clone (within donors)*。它不是亲和力、Kd、EC50 或 SHM 曲线。

Kim et al. 数据的重链表（Zenodo 5895181；论文 DOI `10.1038/s41586-022-04527-1`）提供了：

- `s_pos_clone`：作者克隆级的布尔 Spike 状态；
- `expressed_id` 和 `elisa`：部分表达重组 mAb 的定性 ELISA 状态；
- `nuc_RS_19_312` / `nuc_RS_freq_19_312`：重链序列的 SHM 计数/频率。

其中没有连续的结合亲和力变量（Kd、Ka、IC50/EC50 或滴度）。因此 SHM 只能称为突变历史/负荷，不能称为定量亲和力；二元 Spike 或 ELISA 状态只能称为结合/特异性标签，不能称为 affinity。

`s_pos_clone=FALSE` 表示**未被作者识别为 Spike-positive**，不意味着每个家族都经过单抗结合实验且结果为阴性。图中比较组因此写为“not identified S+”，而非统一实验验证的 Spike-negative。部分 clone 的 ELISA 记录不能外推到所有 FALSE 家族。原文另有表达单抗的亲和力实验，但这不等于本次 GEX/BCR 表中每个家族具有定量 affinity；未完成可审计对应关联时，不能将原文实验结果作为 Threadfin 预测。

## 克隆标签和 Threadfin 家族的关系

现有 `ln_vaccine` 运行把原始 `author_clone_id` 保留到 BCR 表中，但 `run_case_study.py` 用 `tf.define_clones` 从 IGH V、J 和 junction 序列在 donor 内重新推断 `clone_id`；表达矩阵没有参与此步骤。故 programme 与 Spike 标签的关联不是由表达参与定义家族导致的循环推断。

交叉表显示 93,627 个 Threadfin IGH 家族均只落入一个作者 clone；92,761 个作者 clone 中有 623 个被 Threadfin 分成两个或更多 IGH 家族（最多 20 个）。所以作者的 clone 标签会传播到这些 Threadfin 子家族，不能把每个子家族当作独立的外部实验重复，也不能声称 Threadfin 新发现了这些 Spike 标签。

## 标签缺失与负标签

原始重链表在每 cell 留一条记录后有 200,901 行；本次文件的 `s_pos_clone` 全部为 TRUE 或 FALSE（见 `input_coverage.csv`）。GEX 中另有 40,393 个细胞没有配对到任何重链记录，因此也没有进入 Threadfin BCR family 分析。现有 loader 的表达式

```python
"S+" if str(v).upper() == "TRUE" else "S-"
```

会在一般情形把缺失值默认为 `S-`。本次实际进入 GEX/BCR 联合运行的 153,049 个细胞中，所有行的 `s_pos_clone` 恰为 TRUE 或 FALSE，故这一次不存在该误标；40,393 个未配对 GEX 细胞也不曾被赋予克隆标签。主流程现已改为显式三态：TRUE→S+、FALSE→S−、其余→NA，并有缺失值回归检查。本次数据无缺失调用，因此修复不改变当前图表和统计结果。

## 可复现审计产物

运行：

```bash
/data/scratch/projects/punim1236/python_envs/threadfin_v4/bin/python case_studies/spike_binding_audit.py
```

输出在 `case_studies/results/spike_binding_audit/`：

- `threadfin_family_crosswalk.csv` 与 `author_clone_crosswalk.csv`：两个克隆体系的全量映射和标签；
- `family_timepoint_source.csv`：每个 Threadfin 家族、donor、时间点一行；
- `donor_timepoint_binding_shm.csv`：以 donor 内家族为单位的 SHM 汇总与覆盖；
- `donor_programme_binding_coverage.csv`：每 donor-programme 的 S+、S−、未知数量及 S+ 比例；
- `binding_shm_donor_summary.{png,pdf}`：仅使用 donor/family 汇总的描述图。

图中不把 cell 当重复，不把缺失当 S−，并把 SHM 与结合标签分开。它可作为补充性来源/质控图；旧的 programme-enrichment 图只有在标题和图注清楚写明「作者 clone 标签在 Threadfin programme 中的 donor 内富集」时才应恢复。
