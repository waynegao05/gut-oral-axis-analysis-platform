# Gut–Oral Axis Analysis Platform
# 肠口轴智能分析平台

## Overview | 项目概述

This repository is a research-oriented and prototype-oriented project built around the gut–oral axis hypothesis. It integrates oral microbiome signals, graph-based representation, clinical variables, metabolite features, and structured reporting workflows for disease risk analysis and pharmacological assistance research.

The repository currently includes four parallel tracks:

1. a legacy runnable prototype for rapid demonstration
2. a modular Flask API and browser demo
3. a formal research training pipeline based on graph neural networks and Cox-style survival modeling
4. a clinical-style raw JSON workflow for standardized intake, risk analysis, and research-oriented pharmacological assistance

This project is suitable for method prototyping, research experiments, ablation studies, model comparison, manuscript preparation, and structured output demonstration. It is not a validated clinical deployment system.

本仓库围绕“肠口轴”分析思路构建，兼具科研实验与原型演示属性，整合了口腔微生物信号、图结构表征、临床变量、代谢特征与结构化报告流程，用于疾病风险分析与研究型药学辅助探索。

当前仓库主要包含四条并行主线：

1. 用于快速演示的早期可运行原型
2. 模块化 Flask API 与浏览器演示界面
3. 基于图神经网络与 Cox 风格生存建模的正式科研训练流程
4. 面向原始临床风格 JSON 输入的标准化、风险分析与研究型药学辅助工作流

本项目适用于方法原型验证、科研实验、消融研究、模型比较、论文准备与结构化输出展示。当前版本仍不属于经过严格验证的临床部署系统。

---

## Quick Start | 快速开始

### 1. Prototype demo | 原型演示

Install lightweight dependencies and run the enhanced web demo:

    pip install -r requirements.txt
    python enhanced_app.py

Open the browser at:

    http://127.0.0.1:5000

安装轻量依赖后，运行增强版网页演示：

    pip install -r requirements.txt
    python enhanced_app.py

浏览器访问：

    http://127.0.0.1:5000

### 2. Research training | 科研训练

Install research dependencies and run the default training pipeline:

    pip install -r requirements-research.txt
    python -m research.train --config research_config.yaml

安装科研依赖后，运行默认训练流程：

    pip install -r requirements-research.txt
    python -m research.train --config research_config.yaml

### 3. Clinical workflow | 临床工作流

Run the raw-clinical-input workflow:

    pip install -r requirements.txt
    python clinical_workflow.py \
      --input example_raw_clinical_input.json \
      --output outputs/clinical_report.json \
      --standardized_output outputs/standardized_input.json

运行原始临床输入工作流：

    pip install -r requirements.txt
    python clinical_workflow.py \
      --input example_raw_clinical_input.json \
      --output outputs/clinical_report.json \
      --standardized_output outputs/standardized_input.json

---

## Main Tracks | 仓库主线

### 1. Prototype and demo layer | 原型与演示层

This layer is designed for fast verification, visual demonstration, and API testing.

Key entry files:

- `app.py`: legacy single-file prototype
- `modular_app.py`: minimal modular Flask demo
- `enhanced_app.py`: enhanced Flask demo with validation, logging, and report export
- `cli_analysis.py`: command-line prototype runner

这一层主要用于快速验证、可视化演示与接口测试。

关键入口文件：

- `app.py`：早期单文件原型
- `modular_app.py`：最小化模块版 Flask 演示
- `enhanced_app.py`：带校验、日志与结果导出的增强版 Flask 演示
- `cli_analysis.py`：命令行原型运行脚本

### 2. Core prototype logic | 原型核心逻辑层

This layer lives in `src/` and contains the reusable prototype pipeline:

- preprocessing and normalization
- microbial graph construction
- lightweight GNN-style encoding
- prototype Cox-style risk scoring
- rule-based recommendation generation
- report assembly

这一层位于 `src/` 目录中，封装了原型流程的核心逻辑，包括：

- 数据预处理与归一化
- 微生物图构建
- 轻量级 GNN 风格编码
- 原型 Cox 风格风险评分
- 基于规则的建议生成
- 报告结构化输出

### 3. Clinical-style raw input workflow | 临床风格原始输入工作流

This track is used when the input is still a case-style JSON rather than model-ready tables. It standardizes raw clinical information, runs the prototype pipeline, generates research-oriented pharmacological assistance, and builds a structured report.

Key files:

- `clinical_workflow.py`
- `src/clinical_standardizer.py`
- `src/pharmacy_advice.py`
- `src/clinical_report_builder.py`
- `example_raw_clinical_input.json`

这一条主线适用于输入仍是病例式 JSON，而非模型可直接训练的表格数据。流程会先完成原始临床信息标准化，再调用原型分析链，生成研究型药学辅助建议，并输出结构化报告。

关键文件：

- `clinical_workflow.py`
- `src/clinical_standardizer.py`
- `src/pharmacy_advice.py`
- `src/clinical_report_builder.py`
- `example_raw_clinical_input.json`

### 4. Formal research and experiment layer | 正式科研与实验层

This track lives mainly in `research/`, together with the `research_config*.yaml` files and `data/research/` sample tables. It is used for:

- model training
- model prediction
- explainability analysis
- repeated runs across random seeds
- baseline comparison
- graph usefulness verification
- branch-wise experiment variants such as edge-aware or structure-aware versions

这一条主线主要位于 `research/` 目录，同时配合 `research_config*.yaml` 配置文件与 `data/research/` 示例数据表使用，主要服务于：

- 模型训练
- 模型预测
- 可解释性分析
- 多随机种子重复实验
- 基线比较
- 图结构有效性验证
- edge-aware、structure-aware 等分支实验

---

## Repository Structure | 仓库结构

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

---

## Core Directories | 核心目录说明

### Root directory | 根目录

The root directory contains the main entry scripts, dependency files, example input, and research configuration files. In practice, most users only need to begin from one of these files:

- `enhanced_app.py`
- `clinical_workflow.py`
- `research/train.py`

根目录放置了主要入口脚本、依赖文件、示例输入与科研配置文件。实际使用时，多数用户只需要从以下文件开始：

- `enhanced_app.py`
- `clinical_workflow.py`
- `research/train.py`

### `src/`

This directory contains the modular prototype pipeline. It is the software backbone of the demo and raw clinical workflow branch.

该目录承载模块化原型流程，是演示链与原始临床工作流的核心软件骨架。

### `research/`

This directory contains the formal research codebase, including data builders, model definitions, training scripts, evaluation scripts, repeated-run scripts, and graph-value verification experiments.

该目录承载正式科研代码，包括数据构建、模型定义、训练脚本、评估脚本、多轮重复实验脚本，以及图结构价值验证实验。

### `data/`

This directory contains:

- `microbe_drug_rules.json` for prototype recommendation rules
- `data/research/` sample CSV tables for research experiments

该目录主要保存：

- `microbe_drug_rules.json`，用于原型推荐规则
- `data/research/`，用于科研实验的示例 CSV 数据表

### `templates/`

This directory contains the HTML page used by the modular Flask demo.

该目录保存模块化 Flask 演示所使用的前端页面。

### `tests/`

This directory currently contains basic tests for payload validation and output schema checking.

该目录目前主要包含输入校验与输出结构检查的基础测试。

---

## Input Formats | 输入格式

## 1. Prototype / API input | 原型 / API 输入

This format is used by the prototype pipeline and Flask API.

该格式用于原型分析链与 Flask API。

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

## 2. Clinical raw JSON input | 临床原始 JSON 输入

This format is used by `clinical_workflow.py`.

该格式用于 `clinical_workflow.py`。

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

## 3. Formal research CSV inputs | 正式科研 CSV 输入

The research branch expects four aligned tables indexed by `sample_id`.

科研分支需要四张通过 `sample_id` 对齐的表。

### Graph table | 图结构表

    sample_id,node_name,src,dst,abundance,function_score,edge_weight

### Clinical table | 临床表

    sample_id,age,bmi,smoking,family_history

### Metabolite table | 代谢表

    sample_id,bile_acids,scfa,tryptophan_metabolism

### Label table | 标签表

    sample_id,time,event

---

## Output Formats | 输出格式

## 1. Prototype report | 原型报告

The prototype branch returns a compact analysis report.

原型分支输出紧凑型分析报告。

    {
      "top_microbes": [],
      "gnn_features": {},
      "risk_result": {
        "risk_score": 0.0,
        "risk_level": "low"
      },
      "recommendations": []
    }

## 2. Clinical structured report | 临床结构化报告

The clinical workflow returns a more structured result.

临床工作流会输出更完整的结构化结果。

    {
      "patient_summary": {},
      "risk_assessment": {},
      "microbiome_findings": {},
      "pharmacological_assistance": [],
      "disclaimer": ""
    }

---

## Research Experiments | 科研实验说明

The formal research branch contains several levels of experimentation.

科研分支包含多层次实验内容。

### Default training | 默认训练

Use the default config:

    python -m research.train --config research_config.yaml

使用默认配置训练：

    python -m research.train --config research_config.yaml

### Small-sample training | 小样本训练

Use the more conservative small-sample config:

    python -m research.train --config research_config_small_sample.yaml

使用更保守的小样本配置：

    python -m research.train --config research_config_small_sample.yaml

### Baseline comparison | 基线比较

To compare the graph model against simpler tabular alternatives:

    python -m research.baseline_compare \
      --config research_config_small_sample.yaml \
      --seeds 7 21 42 123 2026

用于比较图模型与更简单的表型基线：

    python -m research.baseline_compare \
      --config research_config_small_sample.yaml \
      --seeds 7 21 42 123 2026

### Repeated runs | 多次重复实验

To test robustness across random seeds:

    python -m research.repeat_runs_cli \
      --config research_config_small_sample.yaml \
      --seeds 7 21 42 123 2026

用于测试不同随机种子下的稳定性：

    python -m research.repeat_runs_cli \
      --config research_config_small_sample.yaml \
      --seeds 7 21 42 123 2026

### Graph usefulness verification | 图结构有效性验证

To test whether graph topology contributes meaningful value, check and run:

- `research/graph_specific_baselines.py`
- `research/graph_structure_tests.py`
- `research/graph_structure_tests_edge_aware.py`
- `research/graph_structure_tests_structure_aware.py`
- `research/graph_structure_tests_structure_aware_v2.py`

若要验证图结构是否真正带来有效增益，可重点查看并运行：

- `research/graph_specific_baselines.py`
- `research/graph_structure_tests.py`
- `research/graph_structure_tests_edge_aware.py`
- `research/graph_structure_tests_structure_aware.py`
- `research/graph_structure_tests_structure_aware_v2.py`

---

## Important Notes | 注意事项

### English

This repository contains both a lightweight prototype reasoning chain and a formal trainable research chain. They should not be conflated.

The prototype branch is mainly for demonstration, API testing, and workflow validation. Its scores and recommendations rely on default weights and a small illustrative rule base.

The research branch is the correct place for model training, repeated experiments, baseline comparison, and manuscript-level evaluation.

Neither branch should be interpreted as a direct prescribing engine or a validated clinical decision system. Any medically relevant conclusion still requires proper cohort design, external validation, and clinician review.

### 中文

本仓库同时包含轻量原型推理链与正式可训练科研链，使用时不能混为一谈。

原型分支主要服务于演示、接口测试与流程验证，其分数与推荐依赖默认权重和示例性规则库。

科研分支才是进行模型训练、重复实验、基线比较与论文级评估的正确位置。

无论哪一条主线，都不能直接视为处方引擎或经过验证的临床决策系统。任何医学相关结论仍需依赖规范队列设计、外部验证与临床审阅。

---

## Suggested Long-Term Cleanup | 后续整理建议

Because the repository has evolved iteratively, multiple generations of scripts and documents coexist in the same top-level structure. A cleaner long-term organization would separate the project into clearer layers such as:

- `prototype/`
- `clinical_workflow/`
- `research/`
- `docs/`

This would make the repository easier to navigate and reduce confusion caused by multiple entry files and multiple config branches.

由于仓库是逐步迭代扩展形成的，目前多代脚本与说明文档仍并存于同一顶层结构中。更清晰的长期整理方案，可以考虑将项目拆分为更明确的层级，例如：

- `prototype/`
- `clinical_workflow/`
- `research/`
- `docs/`

这样会显著降低多入口文件与多配置分支带来的理解成本，也更利于后续维护。

---

## Final Note | 最后说明

This repository already has a workable internal logic and a meaningful experimental structure. At the current stage, the main challenge is that the project can be understood quickly and used correctly across its different branches. This README is intended to make that structure explicit, concise, and directly navigable.

这个仓库当前已经具备较完整的内部逻辑与实验结构。现阶段真正的难点已经推进到项目不同分支能否被快速理解、正确使用。这个 README 的目标，就是把整体结构压缩为一条清晰主线，让读者能够直接定位到自己需要的入口和模块。
