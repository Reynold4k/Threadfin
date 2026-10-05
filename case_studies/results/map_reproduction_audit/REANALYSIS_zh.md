# Figure 3 重跑与旧图复现核查

**重跑没有改变原来的 clonal UMAP。** 早期 801 个和晚期 1,183 个家族的 IDs、cell-to-family membership、cell/clone UMAP 坐标全部相同；两个队列的最大坐标差均为 0。

| 设置 | 旧 Figure 3G | 初次改版 |
|---|---|---|
| 数据 | early `malaria` | late `malaria_late` |
| 颜色 | 家族内 PB cell fraction | 家族内 GC cell fraction |
| 点面积 | `1 + 5 × sqrt(n_cells)` | 固定 7 |
| 长宽比例 | 原版 auto | equal |

因此，早期 PB 偏向的“线性外观”与晚期 GC 空间不是同一张图。初次改版误把不同队列的视图用于直接比较，这是呈现上的错误。当前 Figure 3B 恢复早期 PB 图，3C 并排显示晚期 GC 图；数据和颜色均明确标记。没有重新调 UMAP 以制造旧形状。

重跑新增的是生物学检验：修正 mouse gene-symbol module；明确终末鼠/治疗设计；按同鼠 family 列出 GC、PB、memory-like 共现；添加固定 denominator、mouse 和 isotype 条件置换；增加 exact IGH/H+L identity 检查。这些检验不重新定义家族或地图坐标。

二维形状不是 lineage direction；一条视觉轴只能说明 profile 的布局。是否同一祖先关系依赖同鼠受体 membership，是否超出采样预期依赖条件控制。详见 [数值审计](coordinate_comparison.json)、[复现图](../../../paper/figure_plan/review/Figure_3_map_reproduction.png) 和 [状态共享源表](../clone_state_sharing/)。
