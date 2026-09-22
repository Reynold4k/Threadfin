# NEXT SESSION PLAN — Threadfin 多数据集验证(交接文档)

> 本文件为会话交接用,发表前删除。当前状态截至 2026-09-22 深夜。
> 新 session 的 agent:请先完整阅读本文件,再阅读 `docs/BIOLOGICAL_INTERPRETATION.md` 和
> `benchmarks/biological_validation/` 下的代码。

## 目标回顾(用户的科学要求)

把 Threadfin 从"在一个公共数据集上能跑"提升为"跨多个 paired scRNA+scBCR 数据集、
生物学上可解释且有说服力的方法"。核心问题:Threadfin 相比常规 clonotype network
提供了什么新生物学信息。要求:多数据集验证、basis 稳健性(X_pca vs X_umap)、
避免循环验证(held-out)、零模型、与常规克隆网络(scirpy)的正面对比图。
**不要把任何东西硬拗成阳性结果;阴性结果要如实解释原因。**

## 已完成

1. **包 v1.0.0 已发布**(commit 8e9a185,已 push):重构、12 tests、CI、README。
2. **数据集调研完成**(4 个 explore subagent 的报告结论):
   - GSE175522+GSE175523 流感疫苗(6 donor × pre/d7,年轻 vs 老年)— GEO 直接下载 ✅
   - GSE317492 EBV 扁桃体类器官(d0/4/7/14/21,d14/d21 分 GFP±)— GEO 直接下载 ✅
   - King 2021 扁桃体(E-MTAB-9005 GEX + E-MTAB-9003 VDJ,6 donor "Total" 样本)✅
   - GSE195673 SARS-CoV-2 疫苗淋巴结 GC(Kim/Zhou 2022 Nature)— GEO 的 GEX 是 aggr
     矩阵难配对;改用 Zenodo 5895181 的 h5ad(5.7GB,作者整合好)+ bcr tsv ✅
   - GSE171964 不是 Turner 2021(是 Scott 的 CITE-seq),不要再用。
   - Stephenson 2021(stephenson2021_5k.h5mu)保持为 Dataset 1。
3. **验证框架已写好** `benchmarks/biological_validation/`:
   - `validation.py`:compute_qc / standard_preprocess / run_parameter_grid /
     basis_robustness / null_permutation_purity(Null1,供体内置换)/ 
     null_random_communities(Null3)/ heldout_split_validation(克隆内 50/50 分裂,
     两半作伪克隆重聚类,共同聚类率 vs 期望)
   - `run_validation.py`:CLI 入口,产出 qc/runs/robustness/nulls/heldout/figures/
     summary.json/report.md
   - `loaders.py`:5 个 loader,统一契约 (adata + 按 barcode 索引的 bcr 表)
   - `configs/*.json`:5 个数据集配置(含人 B 细胞签名基因集)
4. **Stephenson 验证首跑发现并修复**:
   - null 置换的 pandas 索引 bug(已修)
   - **重要发现:PCA vs UMAP basis 的克隆级 ARI ≈ 0**(不是 bug,已用 crosstab 验证)。
     两种 basis 给出不同的细分群落,但都抓住浆母细胞主结构。这是第 6 节"不要只用 UMAP"
     的实证素材——README/论文里要如实写:细粒度边界随 basis 变,主张用 PCA 做定量、
     UMAP 做展示,且要求结论在两个 basis 下都成立。
   - runs.json 显示 UMAP-basis 在该数据集 NMI 更高(状态来自同一表达矩阵,有循环性,
     只能作参考)。
5. **docs/BIOLOGICAL_INTERPRETATION.md 初稿已写**(结论性数字待最终运行后核对)。
6. **scirpy 0.22.5 已装入 venv**(`/data/scratch/projects/punim1236/threadfin_data/venv`),
   用于常规克隆网络对比图。

## 数据状态(/data/scratch/projects/punim1236/threadfin_data/)

- `download_all.sh` 幂等下载脚本([ -s 存在即跳过,gzip magic 校验,失败重试])。
  session 中断就重跑:`bash /data/scratch/projects/punim1236/threadfin_data/download_all.sh`
- 中断前状态:EBV 2.1G ✅;flu ~492M(12 GEX tar + 12 AIRR 基本齐);
  tonsil ~207M(6 GEX + 6 VDJ + CellTypeMetaData.txt);
  gse195673_ln_vaccine ~469M(bcr_heavy/light/meta 齐,5.7GB h5ad 可能没下完,重跑续传)。
- 已完成下载日志在 session 任务里,中断后不可见;以文件存在为准。
- flu 样本编号 `<donor>_<tp>`:donor 120648/120667/141393=年轻,141394/141409/141415=老年;
  tp 0=pre, 7=d7。GEX GSM5340834-845 ↔ BCR AIRR GSM5340846-857(按样本编号配对)。
- tonsil 只用 BCP*_Total_5GEX(BCP002 是 3' 已排除;MBC/IgMneg 亚群是重复抽样,不用)。

## 下一步(按顺序)

1. **确认下载完整**:重跑 download_all.sh 直到日志全 OK 无 FAIL;`du -sh` 各目录核对大小
   (EBV ~2.1G;flu ~560M;tonsil ~230M;ln_vaccine ~6G)。
2. **逐个冒烟测试 loader**(修好再跑全量):
   ```bash
   cd /data/scratch/projects/punim1236/Threadfin/benchmarks/biological_validation
   V=/data/scratch/projects/punim1236/threadfin_data/venv/bin/python
   PYTHONPATH=. $V -c "
   import json; from loaders import LOADERS
   cfg=json.load(open('configs/flu_gse175522.json'))
   a,b=LOADERS[cfg['loader']](cfg); print(a.shape, b.shape, b.clone_id.nunique())"
   ```
   已知风险点:
   - flu tar 内文件在根目录且无 .gz 后缀(_read_mtx_dir 已兼容,待实证)
   - tonsil CellTypeMetaData.txt 的 barcode 格式 vs GEX barcode(可能需加 donor 前缀映射)
   - ln_vaccine:bcr cell_id 形如 `368-01a_s5@BARCODE-1`,h5ad obs_names 格式未知,
     需要先打印两边各 10 个名字再写映射(load_ln_vaccine_gse195673 里现在是占位实现)
   - EBV loader 用 symlink 拼 mtx 目录,注意 symlink 在 scratch 文件系统是否被允许
3. **跑 5 个数据集全量验证**(每个后台或 sbatch;EBV/LN 数据大,用 sbatch):
   ```bash
   PYTHONPATH=. $V run_validation.py configs/<name>.json
   ```
   预期 EBV(数万细胞)和 LN(可能十几万 B 细胞)要 30-60 分钟;flu/tonsil/stephenson 快。
4. **scirpy 对比图**(`compare_scirpy.py` 还没写):对 stephenson + tonsil(或 flu)做
   Panel A(细胞 UMAP/state)、B(常规 clonotype network:scirpy ir.pp.ir_dist +
   ir.tl.define_clonotypes + ir.pl.clonotype_network)、C(Threadfin clone map)、
   D(细胞按 clone_cluster 着色)、E(签名热图)。要点:不是"打败"scirpy,
   而是展示两者回答不同问题。
5. **填 `benchmarks/datasets_manifest.tsv`**(列:dataset/paper/accession/organism/tissue/
   condition/donors/timepoints/GEX source/BCR source/URL/download date/size/md5/notes)。
   md5 用 `md5sum` 对已下载文件算。
6. **定稿 docs/BIOLOGICAL_INTERPRETATION.md**:把 5 个数据集的真实数字填进去,
   明确哪些 statement 被支持/不支持(包括阴性结果)。
7. **更新 README**:多数据集验证小节 + robustness 结论 + 新图。
8. **commit + push**(push 已验证可用:credential.helper=cache 里有有效凭据)。

## 重要技术备忘

- venv:`/data/scratch/projects/punim1236/threadfin_data/venv`(system-site-packages,
  已装 muon/awkward/scirpy;threadfin 以 pip -e 装入)。
- NCBI FTP 限流:并行 >3 连接会 503;下载必须串行(download_all.sh 已处理)。
- scanpy 1.11 移除了 neighbors 的 metric="precomputed" 支持 → 包里用 sklearn+leidenalg
  自建 kNN+Leiden(core.py `_leiden_on_distances`);igraph 1.0 的 to_undirected() 原地
  操作返回 None(已兼容)。
- 用户 AGENTS.md:大数据/大规模绘图 → sbatch;h5ad 读取顺序整读;print flush 每个阶段。
- 用户的科学红线:不要循环验证当强证据;不要假装阳性;null 模型要说明各自 preserve 什么。
- 当前会话权限:Never Ask 模式;用户希望少问多做,直接执行。
