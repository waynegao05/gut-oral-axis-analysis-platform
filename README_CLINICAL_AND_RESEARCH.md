# Gut–Oral Axis AI Platform

A research-oriented framework for modeling disease risk based on the gut–oral axis. The pipeline integrates microbial interaction networks, clinical variables, and metabolomic features using graph neural networks and survival analysis. It is designed for academic research, model validation, manuscript preparation, and structured pharmacological assistance prototyping rather than direct clinical deployment.

## Installation

Install dependencies before running the project:

```bash
pip install -r requirements-research.txt
```

## Data Preparation

The formal research model requires four input tables in CSV format.

Graph table describes microbial interactions. Each sample corresponds to a graph.

```csv
sample_id,node_name,src,dst,abundance,function_score,edge_weight
```

Clinical table contains basic covariates aligned by `sample_id`.

```csv
sample_id,age,bmi,smoking,family_history
```

Metabolite table provides additional biochemical features.

```csv
sample_id,bile_acids,scfa,tryptophan_metabolism
```

Label table is used for survival modeling.

```csv
sample_id,time,event
```

Time represents follow-up duration. Event is binary, where `1` indicates occurrence and `0` indicates censoring.

## Preprocessing

Before training, normalize all input tables:

```bash
python research/preprocess_tables.py \
  --clinical_csv your_clinical.csv \
  --metabolite_csv your_metabolite.csv \
  --graph_csv your_graph.csv \
  --output_dir outputs/preprocessed
```

## Training

Train the formal research model with:

```bash
python research/train.py --config research_config.yaml
```

The training pipeline includes graph representation learning, multi-modal feature fusion, Cox partial likelihood optimization, and early stopping. The best model is automatically saved.

Outputs are written to:

```text
outputs/research/
├── best_model.pt
├── history.json
└── test_metrics.json
```

## Inference

After training, generate predictions with:

```bash
python research/predict.py \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --split test
```

The output contains a risk score for each sample.

## Explainability

To obtain model interpretability:

```bash
python research/explain.py \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --output outputs/research/explainability.json
```

This produces node-level importance scores and corresponding risk contributions.

## Model Description

The formal research model uses graph attention networks to encode microbial interaction structures. Node embeddings are aggregated through global pooling and combined with clinical and metabolomic features. The fused representation is passed through a feed-forward network, and a scalar risk score is generated. Training is based on Cox proportional hazards loss.

## Evaluation

Performance is evaluated using the concordance index, which measures the consistency between predicted risk ranking and observed survival outcomes.

## Clinical Workflow

In addition to the research training pipeline, the repository also includes a raw-clinical-input workflow intended for structured case intake and research-oriented pharmacological assistance output.

This workflow is designed for clinical-style JSON input that has not yet been converted into model-ready feature tables. It supports direct transformation from raw case data into standardized model input, risk analysis, microbiome-based interpretation, pharmacological assistance suggestions, and a structured final report.

### Workflow Files

The clinical workflow is implemented through the following files:

- `src/clinical_standardizer.py`
- `src/pharmacy_advice.py`
- `src/clinical_report_builder.py`
- `clinical_workflow.py`
- `example_raw_clinical_input.json`
- `CLINICAL_WORKFLOW_USAGE.md`

### Raw Input Example

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
      {"taxon": "Porphyromonas", "abundance": 0.14},
      {"taxon": "Prevotella", "abundance": 0.10},
      {"taxon": "Streptococcus", "abundance": 0.08},
      {"taxon": "Lactobacillus", "abundance": 0.03}
    ]
  },
  "metabolites": {
    "bile_acids": 0.74,
    "scfa": 0.35,
    "tryptophan_metabolism": 0.68
  }
}
```

### What the Workflow Does

The workflow handles four tasks in sequence.

First, it reads raw clinical-style input, including demographic information, medical history, oral microbiome data, metabolite-related indicators, and basic clinical context.

Second, it standardizes the raw input into a model-ready structure. This step converts heterogeneous clinical fields into normalized dictionaries for microbial features, clinical covariates, metabolite features, and metadata.

Third, it sends the standardized input into the existing analysis pipeline and generates a risk assessment result based on the current modeling logic.

Fourth, it builds a research-oriented clinical report together with pharmacological assistance suggestions. The output is intended to support structured interpretation rather than replace physician diagnosis or prescribing decisions.

### How to Run the Clinical Workflow

```bash
python clinical_workflow.py \
  --input example_raw_clinical_input.json \
  --output outputs/clinical_report.json \
  --standardized_output outputs/standardized_input.json
```

### Clinical Workflow Outputs

Two output files are generated.

- `outputs/standardized_input.json` stores the standardized model-ready payload.
- `outputs/clinical_report.json` stores the final structured report.

### Report Structure

The final report contains four main sections.

- `patient_summary` records case context such as sample ID, sex, chief complaint, and suspected condition.
- `risk_assessment` stores model-derived risk score and risk level.
- `microbiome_findings` summarizes key microbial findings and rule-triggered observations.
- `pharmacological_assistance` contains research-oriented suggestions derived from risk level, exposure history, and microbiome signals.

### Important Note for Clinical Workflow

The pharmacological assistance generated by this repository is intended as research-oriented decision support. It should be interpreted alongside physician judgment, laboratory testing, imaging, endoscopy, pathology, and formal gastrointestinal evaluation when necessary. It is not a prescribing engine.

## Notes

This repository is a research framework. Meaningful performance requires real cohort data, proper experimental design, calibrated preprocessing, and external validation before any scientific or clinical interpretation.

---

# 中文说明

本项目构建了一套基于肠口轴的疾病风险预测研究框架，融合微生物互作网络、临床特征与代谢组数据，通过图神经网络与生存分析方法进行建模。该系统面向科研使用，适用于模型实验、论文撰写、方法验证以及研究型药学辅助建议生成，不面向临床直接处方应用。

## 安装

```bash
pip install -r requirements-research.txt
```

## 数据准备

正式研究模型需要四类 CSV 输入数据。

微生物网络数据用于构建图结构，每个样本对应一张图。

```csv
sample_id,node_name,src,dst,abundance,function_score,edge_weight
```

临床数据用于提供个体特征，并通过 `sample_id` 对齐。

```csv
sample_id,age,bmi,smoking,family_history
```

代谢组数据作为补充模态。

```csv
sample_id,bile_acids,scfa,tryptophan_metabolism
```

标签数据用于生存分析。

```csv
sample_id,time,event
```

其中 `time` 为随访时间，`event` 为结局变量，`1` 表示事件发生，`0` 表示删失。

## 数据预处理

```bash
python research/preprocess_tables.py \
  --clinical_csv your_clinical.csv \
  --metabolite_csv your_metabolite.csv \
  --graph_csv your_graph.csv \
  --output_dir outputs/preprocessed
```

## 模型训练

```bash
python research/train.py --config research_config.yaml
```

训练过程包括图表示学习、多模态融合、Cox 损失优化以及早停机制，并自动保存最优模型。

输出文件位于：

```text
outputs/research/
├── best_model.pt
├── history.json
└── test_metrics.json
```

## 模型推理

```bash
python research/predict.py \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --split test
```

输出为每个样本对应的风险评分。

## 可解释性分析

```bash
python research/explain.py \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --output outputs/research/explainability.json
```

可获得节点重要性以及风险贡献信息。

## 模型说明

正式研究模型使用图注意力网络对微生物互作进行编码，通过全局池化得到图表示，并与临床和代谢组特征进行融合。最终通过全连接网络输出风险值，并使用 Cox 比例风险损失进行优化。

## 评价指标

采用 C-index 评估模型性能，用于衡量预测风险排序与真实生存时间的一致性。

## 临床工作流

除研究训练流程外，仓库还提供了一条面向原始临床输入的工作流，用于将临床风格原始数据直接转化为标准化模型输入，并进一步输出风险评估、微生物发现、药学辅助建议与结构化报告。

### 涉及文件

临床工作流由以下文件实现：

- `src/clinical_standardizer.py`
- `src/pharmacy_advice.py`
- `src/clinical_report_builder.py`
- `clinical_workflow.py`
- `example_raw_clinical_input.json`
- `CLINICAL_WORKFLOW_USAGE.md`

### 原始输入示例

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
      {"taxon": "Porphyromonas", "abundance": 0.14},
      {"taxon": "Prevotella", "abundance": 0.10},
      {"taxon": "Streptococcus", "abundance": 0.08},
      {"taxon": "Lactobacillus", "abundance": 0.03}
    ]
  },
  "metabolites": {
    "bile_acids": 0.74,
    "scfa": 0.35,
    "tryptophan_metabolism": 0.68
  }
}
```

### 工作流功能

该流程按顺序完成四个任务。

第一，读取原始临床风格输入，包括人口学信息、病史、口腔菌群数据、代谢物指标以及基础临床背景。

第二，将原始输入标准化为模型可识别的结构。该步骤把异质化临床字段整理为微生物特征、临床协变量、代谢组特征和元数据四个部分。

第三，将标准化后的输入送入现有分析流程，生成风险评估结果。

第四，基于风险结果、微生物异常和暴露历史，生成研究型结构化报告与药学辅助建议。

### 运行方式

```bash
python clinical_workflow.py \
  --input example_raw_clinical_input.json \
  --output outputs/clinical_report.json \
  --standardized_output outputs/standardized_input.json
```

### 输出内容

会生成两个输出文件。

- `outputs/standardized_input.json` 保存标准化后的模型输入。
- `outputs/clinical_report.json` 保存最终结构化报告。

### 报告结构

最终报告包含四个主要部分。

- `patient_summary` 用于记录样本编号、性别、主诉和疑似疾病背景。
- `risk_assessment` 用于记录模型生成的风险评分与风险等级。
- `microbiome_findings` 用于总结关键微生物特征及规则触发情况。
- `pharmacological_assistance` 用于输出基于风险分层、暴露历史和菌群异常的研究型药学辅助建议。

### 临床工作流说明

该流程输出的是研究型药学辅助建议，不替代医师诊断、确证性检查或正式处方决策。所有结果都应结合临床判断、实验室检查、影像、肠镜、病理等信息共同解释。

## 注意事项

该项目为科研框架。模型效果依赖真实数据质量、实验设计、前处理一致性以及外部验证结果。用于论文发表或临床讨论前，必须进行严格验证与审慎解释。
