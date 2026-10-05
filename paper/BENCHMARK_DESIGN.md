# Native-method benchmark protocol

## Question and scope

This benchmark asks a narrow, externally checkable question: can an unsupervised family representation support a label-held-out readout of a **measured GC gate fraction**? It does not test ancestry reconstruction, cell lineage direction, future fate, affinity, or functional protection.

The cohorts are the NP-OVA (`mouse_np`) and RBD-vaccine (`mouse_rbd`) reporter experiments. Endpoints are the family fraction of mCherry-low cells (at least six divisions), RBD-probe-positive cells, or dark-zone-sorted cells. A missing reporter/sort call stays missing: it is never converted to a negative label. The `mouse_np` endpoint is division-gate fraction; RBD endpoints are reported only after all native representations have completed and passed the same audit.

## Fixed common input and families

All methods receive the same label-blind input: productive heavy-chain records and an RNA expression representation prepared by library-size normalisation, log1p transformation and 100-PC PCA after removal of IGH, IGK and IGL genes. Reporter/sort columns are absent from every method input.

Families are the already saved Threadfin donor-private IGH families in the case-study tables. They are not re-called per competitor. Each method must return a complete representation for every cell in a family before that family can enter its method-specific representation; performance uses the intersection across **all seven non-null representations**. Scoring stops if any is missing: RNA centroid, donor-context-centred RNA centroid, Threadfin mean profile, Threadfin kernel profile, Benisse, BiGCN, and clone2vec.

The training-target-mean predictor is included as a zero-feature readout baseline, rather than an eighth representation. Every run writes common-cell and common-family SHA-256 identifiers, coverage, dimensions, software versions, native repository commits, runtime status, and representation audit metadata.

## Representations and adapters

The two RNA baselines separate context handling from profile construction. `RNA_centroid` is the raw mean of the 100-PC cell representation within each family. `RNA_context_centroid` subtracts the donor mean from each cell before the same family mean. Threadfin mean and kernel profiles use `context_key=donor`; thus their comparison with the context-centred centroid assesses profile shrinkage/representation rather than merely donor centring. The kernel profile uses the package's specified random-Fourier/kernel representation and smoothing.

Benisse is executed with the checked-out official encoder and R stage. Its native squared latent-distance output is retained as a complete, cell-weighted family kernel. Classical MDS with up to 30 positive components is saved only to inspect that output; it is not used for performance. BiGCN uses its official pipeline and saved node embedding, then pools node representations to fixed families with cell weights. clone2vec is run with its native 30-dimensional output and recorded parameters. No competitor is replaced with a Threadfin representation or all-pairs substitute.

## Common readout

All non-Benisse representations become a linear family kernel by dot product; Benisse supplies its complete native family kernel. For each outer held-out mouse, kernel centring, trace scaling, target centring and fitting occur using the outer training mice only. Kernel-ridge alpha is selected from `0.1, 1, 10, 100, 1000` by nested whole-mouse training folds. The resulting held-out prediction is evaluated with MAE and R2. The centring and alpha selection never use held-out gate labels.

Representations themselves are fit transductively on all common cells without gate labels. Therefore this is a **label-held-out transductive representation readout**, not an inductive test of fitting a representation on unseen mice.

The primary analysis uses families with at least two captured cells and at least two measured target cells. The pre-specified sensitivity analysis uses the analogous threshold of five. Each score row records the selected-family and donor counts, training/test sizes, target prevalence and variance, chosen alpha, and predictions outside the unit interval. Those predictions are not interpreted as calibrated probabilities.

## Reporting rules and artifacts

Coverage is reported separately from target availability: method coverage uses expanded families, while target and selected-family denominators are written for every endpoint and size threshold. Results are shown by held-out mouse and summarised with the median and spread across mice; families are not pooled as independent test folds. A single official BiGCN run is reported as a single native run: its upstream code does not expose a fixed seed, although its saved output makes the downstream readout reproducible. No result is described as state of the art, as a general ranking of B-cell methods, or as evidence of future fate.

Artifacts are under `case_studies/results/native_benchmark/`: `<dataset>/run.json` contains native source commits/status; `representations/<dataset>/representation_audit.json` records representation/adapters; `scores/<dataset>/coverage.csv`, `targets.csv`, `heldout_scores.csv` and `heldout_predictions.csv` hold denominators and per-fold results; and `scores/<dataset>/benchmark_manifest.json` records identifier hashes, versions, run metadata and scoring contract when the current scorer is run.

## Execute and continue

The native source checkouts are kept outside version control in
`../internal_validation/competitors/sources/{Benisse,BiGCN}`. Use the commits
recorded in `model_runs/` / the model manifest, install the dependencies in
the official repositories, and set `THREADFIN_DATA` to the public-data root.
The adapters write native working files into ignored dataset folders; the
committed outputs contain family representations, fold source tables and audits.

```bash
python case_studies/native_benchmark.py mouse_np --stage prepare
python case_studies/native_benchmark.py mouse_rbd --stage prepare
sbatch --array=1-2 --export=ALL,STAGE=encode case_studies/native_benchmark.sbatch
# Once the encoders succeed, submit both complete native models.
sbatch --array=1-2 --export=ALL,STAGE=benisse-r case_studies/native_benchmark.sbatch
sbatch --array=1-2 --export=ALL,STAGE=bigcn case_studies/native_benchmark.sbatch
sbatch --array=1-2 case_studies/score_native_benchmark.sbatch
# Add afterok dependencies on all upstream model jobs before finalization.
python case_studies/finalize_native_benchmark.py
```

HPC templates carry this workspace's allocation paths; edit them for a different
cluster. Check existing jobs before submitting. In this review iteration, the
RBD models are jobs `32294672_2` (Benisse) and `32288767_2` (BiGCN);
`32296689` is queued with `afterok` dependencies to finalize scores and all
figures automatically. The dependency job generates files; the agent subsequently checks and pushes the completed figures under the existing authorization.
`pipeline_status.json` distinguishes the current NP-only completed comparison
from the final two-dataset result. The biological figures can be regenerated
independently with `make_biology_figures.py --biological-only`.

分选 library 的边界：当前 NP 的 HI/Lo 和 RBD 的 A–D physical library 前缀与 division gate 对应；整鼠留出仍共享这些 library。它检验捕获 gate fraction 的读出，不能单独排除分选/测序 library 的技术影响，也不是独立 library 的泛化或总体分化概率验证。`source_library_gate_counts.csv` 保存该对应关系，标签仍未作为模型特征输入。
