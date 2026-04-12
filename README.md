# Gut–Oral Axis Analysis Platform
# 肠口轴智能分析平台

## Overview | 项目概述

This repository is a multi-layer project built around the gut–oral axis hypothesis. It currently contains four parallel tracks:

1. a legacy runnable prototype for rapid demonstration
2. a modular Flask API and browser demo
3. a research training pipeline based on GAT plus Cox survival modeling
4. a clinical-style raw JSON workflow for standardized intake, risk analysis, and research-oriented pharmacological assistance

本仓库围绕“肠口轴”分析思路构建，当前同时承载四条并行路线：

1. 用于快速演示的早期原型
2. 模块化 Flask API 与浏览器演示界面
3. 基于 GAT 与 Cox 生存分析的科研训练流程
4. 面向原始临床风格 JSON 输入的标准化、风险分析与研究型药学辅助工作流

The repository is suitable for method prototyping, research experiments, ablation studies, manuscript preparation, and structured output demonstrations. It does not represent a validated clinical deployment system.

本仓库适用于方法原型验证、科研实验、消融研究、论文准备与结构化输出展示。当前版本仍属于科研与演示框架，尚未形成经过严格临床验证的正式部署系统。

---

## Project Positioning | 项目定位

### English

This project has two different execution philosophies that must be distinguished clearly.

The first is the prototype branch. It is lightweight, deterministic, fast to run, and suitable for demos, route testing, and UI/API verification. Its risk scoring and recommendation logic rely on manually defined default weights and a small illustrative rule base.

The second is the research branch. It uses graph neural networks, multimodal fusion, and Cox-style survival learning. This branch is intended for training, repeated experiments, baseline comparison, graph usefulness verification, and manuscript-level analysis.

### 中文

这个项目内部存在两套明显不同的执行逻辑，使用时必须先区分清楚。

第一套是原型分支。它轻量、可快速运行，适合做演示、接口测试和界面验证。其风险评分与推荐逻辑主要依赖手工设定权重和示例性规则库。

第二套是科研分支。它使用图神经网络、多模态融合与 Cox 风险建模，主要服务于模型训练、重复实验、基线比较、图结构有效性验证以及论文级分析。

---

## Recommended Entry Points | 推荐入口

| Use case | Entry file | Description |
|---|---|---|
| Quick one-file prototype | `app.py` | Legacy single-file demo with inline HTML |
| Modular web demo | `modular_app.py` | Minimal modular Flask app |
| Enhanced web/API demo | `enhanced_app.py` | Adds validation, logging, and report export |
| Command-line prototype run | `cli_analysis.py` | Runs the prototype pipeline without Flask |
| Clinical-style JSON workflow | `clinical_workflow.py` | Standardizes raw case JSON and outputs a structured report |
| Formal research training | `research/train.py` | Trains the GAT-Cox model |
| Formal research inference | `research/predict.py` | Runs prediction from a trained checkpoint |
| Formal research explainability | `research/explain.py` | Exports interpretation results |

| 用途 | 入口文件 | 说明 |
|---|---|---|
| 单文件快速原型 | `app.py` | 早期单文件演示版，自带内嵌 HTML |
| 模块化网页演示 | `modular_app.py` | 最小化模块版 Flask 应用 |
| 增强版网页/API 演示 | `enhanced_app.py` | 增加校验、日志与结果导出 |
| 命令行原型运行 | `cli_analysis.py` | 不启动 Flask，直接跑原型分析 |
| 临床风格 JSON 工作流 | `clinical_workflow.py` | 标准化原始病例 JSON 并输出结构化报告 |
| 正式科研训练 | `research/train.py` | 训练 GAT-Cox 模型 |
| 正式科研推理 | `research/predict.py` | 使用已训练模型进行预测 |
| 正式科研可解释性分析 | `research/explain.py` | 导出解释性结果 |

---

## Repository Structure | 仓库结构

```text
gut-oral-axis-analysis-platform/
├── app.py
├── modular_app.py
├── enhanced_app.py
├── cli_analysis.py
├── clinical_workflow.py
├── requirements.txt
├── requirements-research.txt
├── api_test.http
├── example_raw_clinical_input.json
├── research_config.yaml
├── research_config_small_sample.yaml
├── research_config_edge_aware.yaml
├── research_config_structure_aware.yaml
├── research_config_structure_aware_v2.yaml
├── research_config_stable_eval.yaml
├── research_config_balanced_eval.yaml
├── research_config_balanced_eval_v2.yaml
├── research_config_balanced_scaled_small.yaml
├── research_config_balanced_scaled_mid.yaml
├── README.md
├── README_CLINICAL_AND_RESEARCH.md
├── USAGE.md
├── TESTING.md
├── DEV_NOTES.md
├── ARCHITECTURE.md
├── RESEARCH_USAGE.md
├── QUICKSTART_WINDOWS_V3.md
├── BASELINE_AND_ABLATION_PLAN.md
├── GRAPH_USEFULNESS_EXPERIMENTS.md
├── CLINICAL_WORKFLOW_USAGE.md
├── config/
│   └── settings.py
├── src/
│   ├── preprocess.py
│   ├── graph_builder.py
│   ├── gnn_encoder.py
│   ├── risk_model.py
│   ├── recommendation.py
│   ├── report.py
│   ├── pipeline.py
│   ├── validators.py
│   ├── export_utils.py
│   ├── logging_utils.py
│   ├── clinical_standardizer.py
│   ├── pharmacy_advice.py
│   └── clinical_report_builder.py
├── research/
│   ├── data.py
│   ├── data_stable.py
│   ├── data_balanced.py
│   ├── data_balanced_scaled.py
│   ├── model.py
│   ├── model_edge_aware.py
│   ├── model_structure_aware.py
│   ├── model_structure_aware_v2.py
│   ├── train.py
│   ├── train_stable.py
│   ├── train_balanced.py
│   ├── train_balanced_v2.py
│   ├── train_balanced_scaled.py
│   ├── train_edge_aware.py
│   ├── train_structure_aware.py
│   ├── train_structure_aware_v2.py
│   ├── predict.py
│   ├── predict_stable.py
│   ├── explain.py
│   ├── explain_stable.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils.py
│   ├── trainer.py
│   ├── preprocess_tables.py
│   ├── baseline_compare.py
│   ├── graph_specific_baselines.py
│   ├── graph_structure_tests.py
│   ├── graph_structure_tests_edge_aware.py
│   ├── graph_structure_tests_structure_aware.py
│   ├── graph_structure_tests_structure_aware_v2.py
│   ├── repeat_runs.py
│   ├── repeat_runs_cli.py
│   ├── repeat_runs_balanced_scaled_small.py
│   ├── repeat_runs_balanced_scaled_mid.py
│   ├── repeat_runs_balanced_v2.py
│   ├── repeat_runs_edge_aware.py
│   ├── repeat_runs_structure_aware.py
│   └── repeat_runs_structure_aware_v2.py
├── data/
│   ├── microbe_drug_rules.json
│   └── research/
│       ├── sample_graph_table.csv
│       ├── sample_clinical_table.csv
│       ├── sample_metabolite_table.csv
│       └── sample_label_table.csv
├── templates/
│   └── index.html
└── tests/
    └── test_pipeline.py
```

---

## File-by-File Responsibilities | 文件逐项作用说明

## 1. Root entry files | 根目录入口文件

| File | Role in English | 中文作用 |
|---|---|---|
| `app.py` | Legacy single-file prototype. Contains preprocessing, graph construction, scoring, recommendation logic, Flask routes, and inline HTML in one file. Useful for the fastest possible end-to-end demo. | 早期单文件原型。预处理、建图、评分、推荐、Flask 路由和内嵌页面全部写在一个文件里，适合最快速演示。 |
| `modular_app.py` | Minimal modular Flask entrypoint. Delegates work to `src/pipeline.py` and uses `templates/index.html`. | 最小模块化 Flask 入口，调用 `src/pipeline.py`，配合 `templates/index.html` 使用。 |
| `enhanced_app.py` | Enhanced Flask entrypoint with payload validation, logging, report export, and cleaner API response structure. | 增强版 Flask 入口，增加输入校验、日志、结果导出与更清晰的接口返回格式。 |
| `cli_analysis.py` | Command-line prototype runner using a built-in default payload. Good for quick local verification without browser or API tools. | 命令行原型执行脚本，内置默认输入，适合本地快速验证，无需浏览器。 |
| `clinical_workflow.py` | Clinical-style workflow orchestrator. Reads raw JSON, standardizes it, runs the prototype pipeline, adds pharmacological assistance, and writes final reports. | 临床风格工作流主控脚本。读取原始 JSON，完成标准化、分析、药学辅助生成与结构化报告输出。 |

---

## 2. Dependency and request files | 依赖与接口测试文件

| File | Role in English | 中文作用 |
|---|---|---|
| `requirements.txt` | Dependencies for the lightweight prototype and Flask demo. | 轻量原型与 Flask 演示所需依赖。 |
| `requirements-research.txt` | Dependencies for the research branch, including PyTorch, PyG, and YAML support. | 科研训练分支所需依赖，包含 PyTorch、PyG 与 YAML 支持。 |
| `api_test.http` | HTTP request template for testing the `/analyze` API endpoint locally. | 本地测试 `/analyze` 接口的 HTTP 请求模板。 |
| `example_raw_clinical_input.json` | Example raw clinical-style input for `clinical_workflow.py`. | `clinical_workflow.py` 使用的原始临床风格输入示例。 |

---

## 3. Root documentation files | 根目录说明文档

| File | Role in English | 中文作用 |
|---|---|---|
| `README.md` | Main repository readme. Should serve as the canonical overview. | 仓库主说明文件，应作为统一入口。 |
| `README_CLINICAL_AND_RESEARCH.md` | Expanded mixed readme covering both research and clinical workflow branches. | 扩展版说明，同时覆盖科研训练与临床工作流。 |
| `USAGE.md` | Short guide for running the modular Flask prototype. | 模块化原型的简要运行说明。 |
| `TESTING.md` | Basic testing instructions for the prototype branch. | 原型分支的基础测试说明。 |
| `DEV_NOTES.md` | Development notes summarizing repository status and future upgrades. | 开发备忘录，记录仓库现状与下一步升级方向。 |
| `ARCHITECTURE.md` | High-level architecture summary aligned with the intended patent chain. | 与预期专利技术链一致的高层架构说明。 |
| `RESEARCH_USAGE.md` | Short operating guide for the formal research branch. | 正式科研分支的简要操作说明。 |
| `QUICKSTART_WINDOWS_V3.md` | Windows-oriented quickstart for module-based research execution. | 面向 Windows 环境的科研流程快速启动说明。 |
| `BASELINE_AND_ABLATION_PLAN.md` | Experiment design note for baseline comparison and ablation studies. | 基线比较与消融实验设计说明。 |
| `GRAPH_USEFULNESS_EXPERIMENTS.md` | Guide for testing whether graph structure contributes meaningful value. | 用于验证图结构是否真正带来增益的实验说明。 |
| `CLINICAL_WORKFLOW_USAGE.md` | Specific usage guide for the raw-clinical-input workflow. | 原始临床输入工作流的专门使用说明。 |

---

## 4. Research configuration files | 科研配置文件

| File | Role in English | 中文作用 |
|---|---|---|
| `research_config.yaml` | Default research training config. | 默认科研训练配置。 |
| `research_config_small_sample.yaml` | Conservative config for small-sample settings. | 小样本场景下更保守的配置。 |
| `research_config_edge_aware.yaml` | Config for the edge-aware model branch. | 面向 edge-aware 模型分支的配置。 |
| `research_config_structure_aware.yaml` | Config for the first structure-aware model branch. | 第一版 structure-aware 模型配置。 |
| `research_config_structure_aware_v2.yaml` | Config for the upgraded structure-aware v2 branch. | structure-aware v2 分支配置。 |
| `research_config_stable_eval.yaml` | Config focused on more stable evaluation behavior. | 面向稳定评估版本的配置。 |
| `research_config_balanced_eval.yaml` | Config for balanced-evaluation experiments. | 平衡评估实验配置。 |
| `research_config_balanced_eval_v2.yaml` | Updated config for balanced-evaluation v2. | 平衡评估 v2 配置。 |
| `research_config_balanced_scaled_small.yaml` | Config for balanced plus scaled small-condition experiments. | 平衡且缩放的小规模实验配置。 |
| `research_config_balanced_scaled_mid.yaml` | Config for balanced plus scaled medium-condition experiments. | 平衡且缩放的中等规模实验配置。 |

---

## 5. `config/` directory | `config/` 目录

| File | Role in English | 中文作用 |
|---|---|---|
| `config/settings.py` | Central settings for the prototype branch, including app name, host, port, default weights, and risk thresholds. | 原型分支的集中配置文件，包含应用名、主机、端口、默认权重与风险阈值。 |

---

## 6. `src/` core prototype modules | `src/` 原型核心模块

| File | Role in English | 中文作用 |
|---|---|---|
| `src/preprocess.py` | Converts raw payload dictionaries into normalized structured input, especially microbiome normalization. | 将输入字典整理为标准化结构，重点处理菌群丰度归一化。 |
| `src/graph_builder.py` | Builds microbial interaction graphs and extracts topology summary features such as density and clustering. | 构建微生物互作图，并提取密度、平均度、聚类系数等拓扑特征。 |
| `src/gnn_encoder.py` | Lightweight GNN-style encoder for prototype inference. Produces graph-derived summary signals. | 轻量级 GNN 风格编码器，用于原型推理，输出图相关综合信号。 |
| `src/risk_model.py` | Cox-style prototype risk scorer combining graph, clinical, and metabolite features using default weights. | Cox 风格原型风险评分器，融合图特征、临床特征和代谢特征。 |
| `src/recommendation.py` | Loads the prototype rule base and generates microbiome-driven recommendations. | 加载原型规则库并生成基于菌群特征的推荐结果。 |
| `src/report.py` | Converts intermediate outputs into the standard prototype report schema. | 将中间结果整理为原型标准报告结构。 |
| `src/pipeline.py` | Main prototype pipeline orchestrator. It links preprocessing, graph building, encoding, scoring, recommendation, and report assembly. | 原型主流程编排模块，串联预处理、建图、编码、评分、推荐与报告构建。 |
| `src/validators.py` | Validates input payload structure for the API branch. | 用于 API 分支的输入结构校验。 |
| `src/export_utils.py` | Saves generated reports into timestamped JSON files. | 将生成报告导出为带时间戳的 JSON 文件。 |
| `src/logging_utils.py` | Creates standard loggers for enhanced runtime output. | 提供增强版运行日志工具。 |
| `src/clinical_standardizer.py` | Standardizes raw clinical-style JSON into model-ready `microbes`, `clinical`, `metabolites`, and `metadata`. | 将原始临床风格 JSON 转为模型可读的 `microbes`、`clinical`、`metabolites` 与 `metadata`。 |
| `src/pharmacy_advice.py` | Generates research-oriented pharmacological assistance based on model output and metadata. | 基于模型结果与元数据生成研究型药学辅助建议。 |
| `src/clinical_report_builder.py` | Builds the final structured clinical report from standardized input and model results. | 将标准化输入、模型结果与药学辅助整合为最终临床风格结构化报告。 |

---

## 7. `research/` formal research modules | `research/` 正式科研模块

### 7.1 Data and utility layer | 数据与工具层

| File | Role in English | 中文作用 |
|---|---|---|
| `research/data.py` | Builds PyG datasets from CSV tables for the default research branch. | 从 CSV 构建默认科研分支的 PyG 数据集。 |
| `research/data_stable.py` | Dataset builder for the stable-evaluation branch. | 稳定评估分支的数据构建器。 |
| `research/data_balanced.py` | Dataset builder for balanced-data experiments. | 平衡数据实验的数据构建器。 |
| `research/data_balanced_scaled.py` | Dataset builder for balanced and scaled settings. | 平衡并缩放场景下的数据构建器。 |
| `research/preprocess_tables.py` | Preprocesses raw CSV tables before training. | 训练前对原始 CSV 表进行预处理。 |
| `research/losses.py` | Implements Cox partial likelihood loss. | 实现 Cox 偏似然损失。 |
| `research/metrics.py` | Implements evaluation metrics such as concordance index. | 实现 C-index 等评价指标。 |
| `research/utils.py` | Shared utility functions for the research branch. | 科研分支通用工具函数。 |
| `research/trainer.py` | Shared training helper logic for research experiments. | 科研实验的通用训练辅助模块。 |

### 7.2 Model definitions | 模型定义层

| File | Role in English | 中文作用 |
|---|---|---|
| `research/model.py` | Default GAT-Cox multimodal model. | 默认 GAT-Cox 多模态模型。 |
| `research/model_edge_aware.py` | Edge-aware model variant emphasizing edge information more explicitly. | edge-aware 模型变体，更强调边信息。 |
| `research/model_structure_aware.py` | First structure-aware model variant. | 第一版 structure-aware 模型。 |
| `research/model_structure_aware_v2.py` | Updated structure-aware model variant. | 更新后的 structure-aware v2 模型。 |

### 7.3 Training, prediction, explanation | 训练、预测与解释层

| File | Role in English | 中文作用 |
|---|---|---|
| `research/train.py` | Default training script for the formal GAT-Cox pipeline. | 默认正式训练脚本。 |
| `research/predict.py` | Default prediction script using trained checkpoints. | 默认预测脚本。 |
| `research/explain.py` | Default explainability script for trained models. | 默认解释性分析脚本。 |
| `research/train_stable.py` | Training script for the stable-evaluation branch. | 稳定评估分支训练脚本。 |
| `research/predict_stable.py` | Prediction script for the stable-evaluation branch. | 稳定评估分支预测脚本。 |
| `research/explain_stable.py` | Explanation script for the stable-evaluation branch. | 稳定评估分支解释脚本。 |
| `research/train_balanced.py` | Training script for balanced-data experiments. | 平衡数据实验训练脚本。 |
| `research/train_balanced_v2.py` | Updated balanced training script. | 更新版平衡训练脚本。 |
| `research/train_balanced_scaled.py` | Training script for balanced and scaled experiments. | 平衡并缩放实验训练脚本。 |
| `research/train_edge_aware.py` | Training script for the edge-aware model. | edge-aware 模型训练脚本。 |
| `research/train_structure_aware.py` | Training script for the first structure-aware model. | 第一版 structure-aware 模型训练脚本。 |
| `research/train_structure_aware_v2.py` | Training script for structure-aware v2. | structure-aware v2 训练脚本。 |

### 7.4 Repeated experiments | 重复实验脚本

| File | Role in English | 中文作用 |
|---|---|---|
| `research/repeat_runs.py` | Repeats default training under multiple random seeds. | 在多个随机种子下重复默认训练。 |
| `research/repeat_runs_cli.py` | Command-line wrapper for repeated runs. | 重复实验的命令行封装脚本。 |
| `research/repeat_runs_balanced_scaled_small.py` | Repeated runs for balanced-scaled small settings. | 平衡缩放小规模场景的重复实验脚本。 |
| `research/repeat_runs_balanced_scaled_mid.py` | Repeated runs for balanced-scaled medium settings. | 平衡缩放中等规模场景的重复实验脚本。 |
| `research/repeat_runs_balanced_v2.py` | Repeated runs for balanced v2 experiments. | 平衡 v2 分支重复实验脚本。 |
| `research/repeat_runs_edge_aware.py` | Repeated runs for the edge-aware branch. | edge-aware 分支重复实验脚本。 |
| `research/repeat_runs_structure_aware.py` | Repeated runs for structure-aware experiments. | structure-aware 分支重复实验脚本。 |
| `research/repeat_runs_structure_aware_v2.py` | Repeated runs for structure-aware v2. | structure-aware v2 分支重复实验脚本。 |

### 7.5 Baselines and graph-value verification | 基线与图价值验证

| File | Role in English | 中文作用 |
|---|---|---|
| `research/baseline_compare.py` | Compares graph models against linear and MLP Cox-style tabular baselines. | 将图模型与线性、MLP Cox 风格表型基线进行比较。 |
| `research/graph_specific_baselines.py` | Tests graph-summary baselines to evaluate whether explicit GNN modeling is necessary. | 测试图摘要基线，评估显式 GNN 是否必要。 |
| `research/graph_structure_tests.py` | Perturbs graph structure to test whether topology matters in the default branch. | 通过扰动图结构检验默认分支对拓扑的依赖程度。 |
| `research/graph_structure_tests_edge_aware.py` | Graph structure perturbation tests for the edge-aware branch. | edge-aware 分支的图结构扰动测试。 |
| `research/graph_structure_tests_structure_aware.py` | Graph structure perturbation tests for structure-aware v1. | structure-aware v1 的图结构扰动测试。 |
| `research/graph_structure_tests_structure_aware_v2.py` | Graph structure perturbation tests for structure-aware v2. | structure-aware v2 的图结构扰动测试。 |

---

## 8. `data/` directory | `data/` 目录

| File | Role in English | 中文作用 |
|---|---|---|
| `data/microbe_drug_rules.json` | Prototype rule base mapping microbial markers to research-oriented suggestions. | 原型规则库，将菌群标志物映射到研究型建议。 |
| `data/research/sample_graph_table.csv` | Sample graph table for the research branch. | 科研分支示例图结构表。 |
| `data/research/sample_clinical_table.csv` | Sample clinical covariate table for the research branch. | 科研分支示例临床协变量表。 |
| `data/research/sample_metabolite_table.csv` | Sample metabolite feature table for the research branch. | 科研分支示例代谢特征表。 |
| `data/research/sample_label_table.csv` | Sample survival label table for the research branch. | 科研分支示例生存标签表。 |

---

## 9. `templates/` and `tests/` | `templates/` 与 `tests/`

| File | Role in English | 中文作用 |
|---|---|---|
| `templates/index.html` | Browser UI for the modular Flask prototype. | 模块化 Flask 原型使用的浏览器前端页面。 |
| `tests/test_pipeline.py` | Basic tests covering payload validation and pipeline output schema. | 基础测试脚本，覆盖输入校验与输出结构。 |

---

## Execution Paths | 运行路径

## A. Prototype web demo | 原型网页演示

```bash
pip install -r requirements.txt
python enhanced_app.py
```

Open:

```text
http://127.0.0.1:5000
```

This path is appropriate for UI demonstration, API verification, and quick result export.

这一路径适合界面演示、接口验证和快速导出结果。

---

## B. Prototype CLI run | 原型命令行运行

```bash
pip install -r requirements.txt
python cli_analysis.py
```

This runs the prototype pipeline directly and writes a timestamped JSON report.

该方式会直接执行原型分析流程，并输出带时间戳的 JSON 报告。

---

## C. Clinical raw JSON workflow | 临床原始 JSON 工作流

```bash
pip install -r requirements.txt
python clinical_workflow.py \
  --input example_raw_clinical_input.json \
  --output outputs/clinical_report.json \
  --standardized_output outputs/standardized_input.json
```

This path is appropriate when the input is still a clinical-style case JSON rather than model-ready CSV tables.

当输入仍是临床风格病例 JSON，尚未整理为科研训练所需的 CSV 表时，建议走这条路径。

---

## D. Formal research training | 正式科研训练

```bash
pip install -r requirements-research.txt
python -m research.train --config research_config.yaml
```

Output directory by default:

```text
outputs/research/
```

默认输出目录为：

```text
outputs/research/
```

---

## E. Formal research prediction | 正式科研预测

```bash
python -m research.predict \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --split test
```

---

## F. Formal research explainability | 正式科研解释性分析

```bash
python -m research.explain \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --output outputs/research/explainability.json
```

---

## G. Baseline comparison | 基线比较

```bash
python -m research.baseline_compare \
  --config research_config_small_sample.yaml \
  --seeds 7 21 42 123 2026
```

---

## H. Repeat runs across seeds | 多随机种子重复实验

```bash
python -m research.repeat_runs_cli \
  --config research_config_small_sample.yaml \
  --seeds 7 21 42 123 2026
```

---

## Input Formats | 输入格式

## 1. Prototype / API input | 原型 / API 输入

```json
{
  "microbes": {
    "Fusobacterium": 0.18,
    "Porphyromonas": 0.15,
    "Prevotella": 0.10
  },
  "clinical": {
    "age": 52,
    "bmi": 24.5,
    "smoking": 1,
    "family_history": 1
  },
  "metabolites": {
    "bile_acids": 0.8,
    "scfa": 0.3,
    "tryptophan_metabolism": 0.7
  }
}
```

## 2. Clinical raw JSON input | 临床原始 JSON 输入

```json
{
  "sample_id": "CASE_001",
  "demographics": {
    "age": 57,
    "sex": "female",
    "bmi": 24.8
  },
  "history": {
    "smoking": "no",
    "family_history_colorectal_or_ibd": "yes",
    "recent_antibiotics": "no",
    "recent_probiotics": "yes"
  },
  "clinical_context": {
    "chief_complaint": "intermittent abdominal discomfort and altered bowel habit",
    "suspected_condition": "colonic disease risk screening"
  },
  "oral_microbiome": {
    "taxa": [
      {"taxon": "Fusobacterium", "abundance": 0.18},
      {"taxon": "Porphyromonas", "abundance": 0.14}
    ]
  },
  "metabolites": {
    "bile_acids": 0.74,
    "scfa": 0.35,
    "tryptophan_metabolism": 0.68
  }
}
```

## 3. Formal research CSV inputs | 正式科研 CSV 输入

The research branch expects four aligned tables.

科研分支需要四张按 `sample_id` 对齐的表。

### Graph table | 图结构表

```csv
sample_id,node_name,src,dst,abundance,function_score,edge_weight
```

### Clinical table | 临床表

```csv
sample_id,age,bmi,smoking,family_history
```

### Metabolite table | 代谢表

```csv
sample_id,bile_acids,scfa,tryptophan_metabolism
```

### Label table | 标签表

```csv
sample_id,time,event
```

---

## Output Schema | 输出结构

## Prototype report | 原型报告

```json
{
  "top_microbes": [],
  "gnn_features": {},
  "risk_result": {
    "risk_score": 0.0,
    "risk_level": "low"
  },
  "recommendations": []
}
```

## Clinical structured report | 临床结构化报告

```json
{
  "patient_summary": {},
  "risk_assessment": {},
  "microbiome_findings": {},
  "pharmacological_assistance": [],
  "disclaimer": ""
}
```

---

## Current Logic Boundaries | 当前逻辑边界

### English

The repository contains both a prototype reasoning chain and a formal trainable research chain. They should not be conflated.

The prototype branch uses hand-crafted defaults, lightweight graph encoding, and an illustrative rule base. It is useful for demonstration and software flow validation.

The research branch contains the trainable model family and the experimental scripts that support repeated runs, baseline comparison, graph perturbation analysis, and branch-wise ablation.

Neither branch should be interpreted as a direct prescribing system. Any medically relevant conclusion still requires formal cohort design, external validation, and clinician review.

### 中文

仓库内部同时存在原型推理链与正式可训练科研链，使用时不能混为一谈。

原型分支依赖手工默认参数、轻量图编码与示例规则库，适合做流程展示和软件验证。

科研分支才承载真正可训练的模型家族，以及重复实验、基线比较、图扰动验证和分支消融等论文级脚本。

无论哪一条路径，都不能直接作为处方系统使用。任何医学相关结论都仍需依赖正式队列设计、外部验证与临床审阅。

---

## Suggested Cleanup Direction | 建议后续整理方向

### English

Because the repository has grown by iteration, several generations of files coexist. A cleaner long-term organization would separate the project into:

- `prototype/` for demo code
- `clinical_workflow/` for raw clinical intake and structured reporting
- `research/` for formal models and experiments
- `docs/` for auxiliary markdown documents

This would reduce confusion around multiple entry files and multiple config branches.

### 中文

由于仓库是迭代式扩展出来的，当前保留了多代文件。更清晰的长期整理方式可以考虑拆成：

- `prototype/` 保存演示代码
- `clinical_workflow/` 保存原始临床输入与结构化报告流程
- `research/` 保存正式模型与实验脚本
- `docs/` 保存辅助说明文档

这样可以显著降低多入口文件、多配置分支带来的理解负担。

---

## Minimal Quick Start | 最小快速开始

### Prototype demo | 原型演示

```bash
pip install -r requirements.txt
python enhanced_app.py
```

### Research training | 科研训练

```bash
pip install -r requirements-research.txt
python -m research.train --config research_config.yaml
```

### Clinical workflow | 临床工作流

```bash
pip install -r requirements.txt
python clinical_workflow.py \
  --input example_raw_clinical_input.json \
  --output outputs/clinical_report.json \
  --standardized_output outputs/standardized_input.json
```

---

## Final Note | 最后说明

This repository already has a usable internal logic. The main issue is not lack of functionality. The main issue is coexistence of multiple development stages in one top-level directory. This README is written to make that complexity explicit and navigable.

这个仓库当前已经具备可运行、可实验、可展示的内部逻辑。真正需要解决的核心问题在于，多代开发阶段的文件同时堆叠在同一层目录中。这个 README 的目标，就是把这种复杂性直接展开，让使用路径和文件职责都可追踪、可理解。
