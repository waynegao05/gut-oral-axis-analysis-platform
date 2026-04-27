# Gut-Oral Axis Analysis Platform
# 肠口轴分析平台

## Usage First | 用法优先

### 1. Run the web app | 启动网页端

This is the fastest way to use the current platform.

这是当前平台最快的使用方式。

```cmd
cd /d <repo-root>
python -m pip install -r requirements.txt
set GOA_PORT=8765
python enhanced_app.py
```

Open:

打开浏览器访问：

```text
http://127.0.0.1:8765
```

If `8765` is blocked on your machine, switch to another port:

如果你的机器上 `8765` 被占用或被拦截，可以切换端口：

```cmd
set GOA_PORT=8000
python enhanced_app.py
```

The web app now uses the actual research model backend by default, not the legacy hand-crafted prototype scorer.

当前网页端默认调用真实研究主线模型，不再使用早期手工权重原型打分器。

---

### 2. Run the current research mainline | 运行当前研究主线

Install the research dependencies:

安装研究环境依赖：

```cmd
cd /d <repo-root>
python -m pip install -r requirements-research.txt
```

Recommended CUDA environment note:

推荐的 CUDA 环境说明：

- Python: `python` available in the active environment
- PyTorch: current project environment expects CUDA runners
- Optional PyG acceleration package:

```cmd
python -m pip install --no-cache-dir --force-reinstall torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

Single training run:

单次训练：

```cmd
cd /d <repo-root>
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m research.train_v2 --config research_config_v2.yaml --split-seed 42 --device cuda
```

Repeated runs across fixed seeds:

固定种子重复实验：

```cmd
cd /d <repo-root>
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m research.repeat_runs_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda --output-root outputs\current_mainline_v2\cox_fixed_split_repeat
```

Graph perturbation test:

图结构扰动实验：

```cmd
cd /d <repo-root>
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m research.graph_structure_tests_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda --output-root outputs\current_mainline_v2\cox_fixed_split_graph_structure_tests
```

5-seed ensemble evaluation:

5-seed 集成评估：

```cmd
cd /d <repo-root>
python -m research.ensemble_v2 --config research_config_v2.yaml --checkpoint-glob "outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt" --split test --device cuda --split-seed 42 --output outputs/current_mainline_v2/cox_fixed_split_repeat/ensemble_test_summary.json
```

Fixed-split benchmark summary:

固定划分 benchmark 汇总：

```cmd
cd /d <repo-root>
python -m research.fixed_split_benchmark --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda --output-root outputs\current_mainline_v2\cox_fixed_split_benchmark --summarize-existing --mainline-summary-path outputs\current_mainline_v2\cox_fixed_split_repeat\research_repeat_runs_summary.json --tabular-summary-path outputs\current_mainline_v2\fixed_split_benchmark\tabular_baseline\baseline_compare_summary.json --graph-summary-path outputs\current_mainline_v2\fixed_split_benchmark\graph_baseline\graph_specific_baselines_summary.json --ensemble-summary-path outputs\current_mainline_v2\cox_fixed_split_repeat\ensemble_test_summary.json --mainline-model-name gnn_cox_mainline
```

---

### 3. Web payload example | 网页输入示例

The canonical payload accepted by `/analyze` is:

网页 `/analyze` 接口接受的标准输入格式如下：

```json
{
  "microbes": {
    "Fusobacterium": 0.18,
    "Porphyromonas": 0.14,
    "Prevotella": 0.10,
    "Streptococcus": 0.22,
    "Lactobacillus": 0.08
  },
  "clinical": {
    "age": 57,
    "bmi": 24.8,
    "smoking": 0,
    "family_history": 1
  },
  "metabolites": {
    "bile_acids": 0.74,
    "scfa": 0.35,
    "tryptophan_metabolism": 0.68
  }
}
```

The current research-backed web inference supports the oral taxa used by the mainline training tables:

当前研究模型驱动的网页推理主要支持主线训练表中使用的口腔菌：

- `Fusobacterium`
- `Porphyromonas`
- `Prevotella`
- `Streptococcus`
- `Lactobacillus`

Unsupported taxa will not crash the app, but they are ignored by the research GNN backend.

未支持的菌不会导致网页报错，但不会进入研究 GNN 后端的真实推理。

---

## What This Platform Is | 这个平台是什么

This repository is a research-oriented oral-gut axis analysis platform built around a conservative graph neural network survival workflow.

本仓库是一个面向研究的肠口轴分析平台，其核心是保守、可复现、以图神经网络生存分析为主线的工作流。

Its current purpose is not clinical deployment and not metric chasing through uncontrolled model complexity. The platform is intended to support:

当前目标不是临床部署，也不是通过失控的模型堆叠去追逐单次高分。平台主要服务于以下目标：

- oral microbiome graph modeling
- multi-modal survival risk prediction
- graph contribution verification
- baseline comparison against simpler tabular models
- reproducible fixed-split experiments
- web-based structured risk demonstration
- future manuscript, patent, and roadshow support

对应中文：

- 口腔微生物图建模
- 多模态生存风险预测
- 图结构贡献验证
- 与简单表型基线的公平比较
- 可复现的固定划分实验
- 基于网页的结构化风险展示
- 为后续论文、专利和路演提供可解释基础

---

## Current Mainline | 当前主线

The default research mainline is:

当前默认研究主线是：

- `GNN + Cox survival modeling`
- config: `research_config_v2.yaml`
- task: right-censored survival risk prediction
- labels: `time`, `event`
- primary metric: `c-index`

This is the paper-facing mainline. Older prototype branches and discarded heavy experiments are preserved only as history or auxiliary references.

这是当前面向论文和正式实验的主线。早期原型分支与已废弃的重型路线只作为历史或辅助参考保留。

The repository explicitly does **not** treat the following as the default path:

仓库当前明确 **不** 将下列路线作为默认主线：

- contrastive-heavy training
- deep attention stacks as the default
- over-constrained graph-only supervision
- uncontrolled multi-task stacking
- architecture novelty before baseline validation

---

## Current Evidence Snapshot | 当前证据快照

Under the fixed-split `topology_v6` workflow currently committed in this repository:

在当前仓库提交的 `topology_v6` 固定划分工作流下：

- `gnn_5seed_ensemble`: `0.74097`
- `all_tabular_mlp_cox`: `0.73955`
- `gnn_cox_mainline` mean: `0.73877 ± 0.00290`
- `graph_summary_mlp_cox`: `0.72708`

The graph perturbation results on the Cox mainline show:

基于 Cox 主线的图扰动结果显示：

- `original`: `0.73783 ± 0.00454`
- `shuffle_weights`: `0.71463 ± 0.00193`
- `shuffle_edges`: `0.71330 ± 0.00291`
- `shuffle_edges_and_weights`: `0.71240 ± 0.00236`

Interpretation:

解释：

- the conservative `GNN + Cox` mainline is competitive with the strongest tabular baseline
- the 5-seed ensemble is currently the best inference configuration in this repository
- graph perturbation consistently degrades performance, so the model is using graph structure rather than merely carrying a graph-shaped shell

对应中文：

- 保守版 `GNN + Cox` 主线已经基本追平最强表型基线
- 5-seed ensemble 是当前仓库内最佳推理配置
- 图扰动会稳定带来性能下降，说明模型确实在利用图结构，而不是“披着图外壳的表型模型”

Important caveat:

重要前提：

The current mainline uses `topology_v6`, which should be treated as synthetic / expanded research data with noise. These metrics are suitable for method development, not for direct clinical claims.

当前主线使用的是 `topology_v6`，应视为带噪声的合成/扩增研究数据。当前指标适用于方法开发，不足以直接支持临床结论。

---

## Scientific Task Definition | 科学任务定义

The platform is built around **right-censored survival risk prediction**, not plain classification.

本平台围绕 **右删失生存风险预测** 构建，而不是普通二分类。

Canonical target semantics:

标准标签语义：

- `time`: observed follow-up duration
- `event = 1`: event observed at `time`
- `event = 0`: right-censored sample

The shared evaluation axis is:

统一评价轴是：

- concordance index (`c-index`)

This keeps the GNN mainline, linear Cox baselines, graph-summary baselines, and tabular MLP survival baselines on the same task definition.

这样可以保证 GNN 主线、线性 Cox 基线、graph-summary 基线和 tabular MLP 生存基线处于同一任务定义下进行比较。

For more detail, see:

详见：

- `research/TASK_DEFINITION.md`

---

## Platform Mechanism | 平台机制

### A. Data modalities | 数据模态

The current mainline fuses three information sources:

当前主线融合三类信息：

1. oral microbiome graph / node-level features
2. clinical covariates
3. metabolite features

即：

1. 口腔微生物图及节点特征
2. 临床协变量
3. 代谢特征

### B. Graph branch | 图分支

The graph branch models oral taxa as nodes with node features such as abundance and function score, and edges representing structured relationships in the microbiome graph.

图分支将口腔菌建模为节点，节点特征包括丰度和功能分数，边表示微生物图中的结构关系。

The current research backbone is implemented in:

当前研究主干实现位置：

- `research/model_v2.py`

It is intentionally conservative:

它被刻意控制在保守复杂度范围内：

- moderate graph depth
- no default attention explosion
- no default contrastive branch in optimization
- weak auxiliary structure supervision only
- Cox head as default survival objective

即：

- 中等深度图网络
- 不默认堆叠复杂 attention
- 不把 contrastive 分支作为默认优化目标
- 仅保留温和的结构辅助监督
- 默认使用 Cox 生存头

### C. Survival head | 生存头

The default head is Cox-style survival risk modeling:

默认生存头是 Cox 风格风险建模：

- model output: risk score
- supervision: censored survival objective
- main metric: c-index

This keeps the project aligned with survival semantics and avoids collapsing the task into an easier but scientifically weaker pseudo-classification setup.

这保证项目始终保持生存分析语义，避免把任务退化为更容易但科学上更弱的伪分类问题。

### D. Web inference mechanism | 网页推理机制

The web app is no longer a toy demo layer. It now performs:

网页端已不再是单纯玩具原型。它现在执行的是：

1. payload validation / raw-to-canonical normalization
2. structured input construction
3. microbe graph construction
4. loading the research model bridge
5. scoring with the current research checkpoints
6. percentile-style risk reporting
7. rule-based recommendation generation on top of model output

核心链路位置：

- `enhanced_app.py`
- `src/pipeline.py`
- `src/research_model_bridge.py`

The current default web backend is:

当前网页默认后端是：

- `research_gnn_cox_ensemble`

This backend loads the current research checkpoints and performs prediction-level ensemble scoring when multiple seed models are available.

当存在多个 seed checkpoint 时，这个后端会自动加载并执行预测级 ensemble 评分。

---

## Repository Structure | 仓库结构

### Top level | 顶层结构

```text
gut-oral-axis-analysis-platform/
├── archive/                       historical branches, failed routes, old outputs
├── config/                        runtime settings for the web app
├── data/                          sample tables and rule files
├── outputs/                       experiment outputs and summaries
├── research/                      paper-facing research pipeline
├── src/                           web / prototype / bridge modules
├── static/                        front-end JavaScript assets
├── templates/                     Flask HTML templates
├── tests/                         lightweight test coverage
├── enhanced_app.py                current Flask entrypoint
├── research_config_v2.yaml        default GNN + Cox mainline config
├── CURRENT_MAINLINE.md            concise mainline reference
└── README.md                      this document
```

### `research/` | 研究代码层

Key current files:

当前关键文件：

- `research/model_v2.py`: current conservative GNN + Cox backbone
- `research/train_v2.py`: single-run training
- `research/repeat_runs_v2.py`: repeated-run evaluation
- `research/graph_structure_tests_v2.py`: graph perturbation tests
- `research/ensemble_v2.py`: checkpoint ensemble evaluation
- `research/fixed_split_benchmark.py`: unified benchmark summary builder
- `research/baseline_compare.py`: tabular survival baselines
- `research/graph_specific_baselines.py`: graph-summary baselines
- `research/graph_preprocess_sweep.py`: graph preprocessing sweep
- `research/TASK_DEFINITION.md`: task semantics

### `src/` | 网页与桥接层

Key current files:

当前关键文件：

- `src/pipeline.py`: high-level inference pipeline
- `src/research_model_bridge.py`: loads research checkpoints for web inference
- `src/preprocess.py`: payload structuring
- `src/graph_builder.py`: graph construction and topology features
- `src/recommendation.py`: recommendation generation
- `src/report.py`: structured report assembly
- `src/clinical_standardizer.py`: raw clinical JSON normalization

### `outputs/` | 输出目录

Important output groups:

当前重要输出组：

- `outputs/current_mainline_v2/cox_fixed_split_repeat/`
- `outputs/current_mainline_v2/cox_fixed_split_graph_structure_tests/`
- `outputs/current_mainline_v2/cox_fixed_split_benchmark/`
- `outputs/current_mainline_v2/fixed_split_benchmark/`

These folders store the benchmark evidence currently reflected in the repository.

这些目录保存了当前仓库中已经沉淀下来的主线证据。

---

## Recommended Workflow | 推荐工作流

### For web demonstration | 用于网页演示

Use:

使用：

- `enhanced_app.py`

This path is appropriate when you need:

适用场景：

- manual form input
- JSON upload / JSON paste
- quick risk demonstration
- structured report export

### For formal experiments | 用于正式实验

Use the `research/` pipeline with `research_config_v2.yaml`.

使用 `research/` 管线，并以 `research_config_v2.yaml` 为默认主线。

Recommended order:

建议顺序：

1. `research.train_v2`
2. `research.repeat_runs_v2`
3. `research.graph_structure_tests_v2`
4. `research.ensemble_v2`
5. `research.fixed_split_benchmark`

### For baseline-first audit | 用于保守 benchmark 审核

Run:

运行：

- `research.baseline_compare`
- `research.graph_specific_baselines`
- `research.fixed_split_benchmark`

This is the correct path when you need a defensible claim about whether the GNN is truly adding value over simpler alternatives.

当你需要对“GNN 是否真正优于更简单基线”做出可辩护结论时，这条路径才是正确路径。

---

## Input and Output Semantics | 输入输出语义

### Research input tables | 研究输入表

The formal research workflow expects aligned CSV tables indexed by `sample_id`.

正式研究流程要求按 `sample_id` 对齐的多张 CSV 表。

Required tables:

必需表：

- graph table
- clinical table
- metabolite table
- label table

Typical label columns:

典型标签列：

```text
sample_id,time,event
```

### Web output | 网页输出

The web app returns:

网页端输出：

- `risk_score`
- `risk_level`
- `risk_percentile`
- `raw_model_risk`
- `backend`
- recommendations
- structured report JSON

The most important practical point is that the displayed risk is currently percentile-style cohort-relative risk derived from the research model reference cohort.

最重要的实际含义是：网页显示的风险值本质上是基于研究参考队列计算出来的百分位风险，而不是直接可解释为临床绝对发生率。

---

## Reproducibility Rules | 可复现性约束

The repository now follows a conservative experiment discipline:

仓库当前遵循保守实验纪律：

- fixed seeds: `7, 21, 42, 123, 2026`
- fixed split benchmark support
- unified task semantics across baselines and GNN
- explicit graph perturbation tests
- preserved ensemble summary
- config-driven training

When reporting results, always state:

汇报结果时应始终说明：

- dataset version
- split seed
- model family
- whether the score is single-run, repeated-run mean, or ensemble
- whether the data are synthetic / expanded

---

## What This Platform Can and Cannot Claim | 平台当前能支持与不能支持的主张

### Supported claims | 当前可支持的主张

- the project has a stable `GNN + Cox` mainline
- the mainline is competitive with strong tabular survival baselines on the current synthetic benchmark
- the graph branch contributes meaningful information under perturbation tests
- the web demo is now connected to the research model rather than a toy scorer

### Unsupported claims | 当前不能支持的主张

- direct clinical deployment
- prescribing decisions without clinician oversight
- external generalization to real cohorts not represented in this repository
- strong translational claims from `topology_v6` synthetic data alone

---

## Key Files to Read Next | 建议继续阅读的关键文件

If you are new to this repository, read in this order:

如果你刚接手仓库，建议按这个顺序阅读：

1. `README.md`
2. `CURRENT_MAINLINE.md`
3. `research/TASK_DEFINITION.md`
4. `research_config_v2.yaml`
5. `research/train_v2.py`
6. `research/model_v2.py`
7. `src/research_model_bridge.py`

---

## Final Notes | 最后说明

This repository intentionally keeps historical branches, abandoned routes, and archived outputs because the project has gone through multiple rounds of trial and rollback. Those files are useful as history, but they are not the mainline.

本仓库保留了历史分支、废弃路线和归档输出，因为项目经历过多轮试错与回退。这些内容对追溯历史有价值，但它们不是当前主线。

If you only need the current defensible path, use this rule:

如果你只关心当前最可辩护的路径，可以记住这条规则：

> Web entry: `enhanced_app.py`  
> Research mainline: `research_config_v2.yaml` + `research/*_v2.py`  
> Default modeling claim: conservative `GNN + Cox`  
> Best inference setup: `5-seed ensemble`

Clinical disclaimer:

临床免责声明：

This repository is a research and demonstration platform. It is not a validated clinical decision system. Any medically meaningful conclusion requires real cohort design, external validation, and clinician review.

本仓库是研究与演示平台，不是经过验证的临床决策系统。任何具有医学意义的结论都必须依赖真实队列设计、外部验证与临床审核。

## Foud Support | 基金支持

This research was supported by the Undergraduate Talented Innovation Education Program of Shenyang Pharmaceutical University, with the project number XH2025-06. We would like to express our gratitude.

本研究受到沈阳药科大学本科生拔尖创新人才培养计划支持，项目编号XH2025-06，特此感谢。
