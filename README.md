# Gut–Oral Axis AI Platform

A research-oriented framework for modeling disease risk based on the gut–oral axis. The pipeline integrates microbial interaction networks, clinical variables, and metabolomic features using graph neural networks and survival analysis. It is designed for academic research, model validation, and manuscript preparation rather than clinical deployment.

## Installation

Install dependencies before running the project:

```bash 
pip install -r requirements-research.txt`

## Data Preparation

The model requires four input tables in CSV format.

Graph table describes microbial interactions. Each sample corresponds to a graph.

sample_id,node_name,src,dst,abundance,function_score,edge_weight

Clinical table contains basic covariates aligned by sample_id.

sample_id,age,bmi,smoking,family_history

Metabolite table provides additional biochemical features.

sample_id,bile_acids,scfa,tryptophan_metabolism

Label table is used for survival modeling.

sample_id,time,event

Time represents follow-up duration. Event is binary, where 1 indicates occurrence and 0 indicates censoring.

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

Train the model with:

```bash 
python research/train.py --config research_config.yaml`

The training pipeline includes graph representation learning, multi-modal feature fusion, Cox partial likelihood optimization, and early stopping. The best model is automatically saved.

Outputs are written to:

outputs/research/
best_model.pt  
history.json  
test_metrics.json  

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

The model uses graph attention networks to encode microbial interaction structures. Node embeddings are aggregated through global pooling and combined with clinical and metabolomic features. The fused representation is passed through a feed-forward network, and a scalar risk score is generated. Training is based on Cox proportional hazards loss.

## Evaluation

Performance is evaluated using the concordance index, which measures the consistency between predicted risk ranking and observed survival outcomes.

## Notes

This repository is a research framework. Meaningful performance requires real cohort data and proper experimental design. External validation is necessary before any scientific or clinical interpretation.

---

# 中文说明

本项目构建了一套基于肠口轴的疾病风险预测研究框架，融合微生物互作网络、临床特征与代谢组数据，通过图神经网络与生存分析方法进行建模。该系统面向科研使用，适用于模型实验、论文撰写与方法验证，不面向临床直接应用。

## 安装

```bash 
pip install -r requirements-research.txt`

## 数据准备

需要提供四类CSV数据。

微生物网络数据用于构建图结构，每个样本对应一张图。

sample_id,node_name,src,dst,abundance,function_score,edge_weight

临床数据用于提供个体特征。

sample_id,age,bmi,smoking,family_history

代谢组数据作为补充模态。

sample_id,bile_acids,scfa,tryptophan_metabolism

标签数据用于生存分析。

sample_id,time,event

其中time为随访时间，event为结局变量，1表示事件发生，0表示删失。

## 数据预处理

```bash
python research/preprocess_tables.py \
  --clinical_csv your_clinical.csv \
  --metabolite_csv your_metabolite.csv \
  --graph_csv your_graph.csv \
  --output_dir outputs/preprocessed
```

该步骤完成标准化处理与数据格式统一。

## 模型训练

```bash 
python research/train.py --config research_config.yaml`

训练过程包括图表示学习、多模态融合、Cox损失优化以及早停机制，并自动保存最优模型。

输出文件位于：

outputs/research/
best_model.pt  
history.json  
test_metrics.json  

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

模型使用图注意力网络对微生物互作进行编码，通过全局池化得到图表示，并与临床和代谢组特征进行融合。最终通过全连接网络输出风险值，并使用Cox部分似然进行优化。

## 评价指标

采用C-index评估模型性能，用于衡量预测风险排序与真实生存时间的一致性。

## 注意事项

该项目为科研框架，模型效果依赖真实数据与实验设计。用于论文发表前需完成外部验证与统计分析。