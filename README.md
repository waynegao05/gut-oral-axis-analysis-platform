# Gut-Oral Axis Analysis Platform | 肠口轴分析平台

面向右删失生存风险预测的研究与网页演示平台。当前正式网页后端将结构感知 GNN 与时间拓扑 AFT 专家进行跨划分共识融合。

## Current Release | 当前发布

| Item | Current value |
|---|---|
| Task | right-censored survival risk prediction (`time`, `event`) |
| Default backend | `temporal_topology_aft_cross_split_consensus` |
| Release | `temporal_topology_aft_cross_split_consensus_v1` |
| Split branches | `42`, `43` |
| Model seeds | `7`, `21`, `42`, `123`, `2026` |
| Members | 6 GNN models + 10 AFT models |
| Consensus | validation-selected shared `alpha = 0.63` |
| Primary metric | C-index |
| Dataset | `topology_v6` synthetic/noisy augmented research data |

网页端会对两个 split 的共识风险再取平均，并输出队列相对风险百分位、split disagreement 与可靠性提示。

## Quick Start | 快速启动

### 1. Install dependencies

```powershell
cd <repo-root>
python -m pip install -r requirements.txt
```

如使用 CUDA，可将 `GOA_TEMPORAL_DEVICE` 设为 `cuda` 或 `auto`。CPU 是默认部署设备。

### 2. Run the web app

```powershell
$env:GOA_PORT = "8765"
$env:GOA_MODEL_BACKEND = "temporal_topology"
$env:GOA_TEMPORAL_DEVICE = "cpu"
python enhanced_app.py
```

访问 `http://127.0.0.1:8765`。


## Web Input | 网页输入

`/analyze` 接收标准化 JSON：

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

输入校验会直接提示非法值：

- 菌群丰度和代谢物必须是有限数值且位于 `[0, 1]`
- `age` 必须位于 `[1, 120]`
- `bmi` 必须位于 `[5, 100]`
- `smoking` 与 `family_history` 只能为 `0` 或 `1`
- `NaN`、`Infinity`、负丰度和非数字字符串会被拒绝

## Model Architecture | 模型结构

当前推理链由五个步骤组成：

1. **Canonical validation**：校验并规范化菌群、临床变量和代谢物。
2. **Topology inference**：每个 split 使用仅在其训练集 `2160` 个样本上拟合的标准化 Ridge 模型，从 12 个网页可用输入推断 5 个 function scores 和 10 条具名边权。
3. **Structure-aware GNN**：每个 split 使用 3 个已选 GNN 风险成员，保留节点与边身份信息。
4. **Temporal-topology AFT expert**：每个 split 使用 5 个 XGBoost AFT 种子模型，直接学习右删失生存时间结构。
5. **Cross-split consensus**：先将标准化 GNN 风险与 AFT 风险按 `0.37 / 0.63` 融合，再平均 split 42 与 43 的结果。

网页输入无法直接测得真实边权或功能分数，因此网页端使用的是 **inferred topology**，不是实验室实测拓扑。响应中会明确返回：

- `topology_source = inferred_from_web_inputs`
- `topology_inference_method = split_train_only_standardized_ridge`
- 推断得到的 function scores 与 edge weights
- 默认填充值、训练范围外输入与 split disagreement

GNN 结构归一化原本会依赖同一批次中的其他样本。网页部署改用每个 split 固定的中位数校准 anchor，使单个受试者的分数不受并发请求内容影响。正式离线复跑仍使用保存的 8 样本评估上下文，两种上下文不能混为同一验证协议。

## Formal Evidence | 正式证据

权重仅由 validation 选择，test 标签不参与融合权重选择。

| Split | Reference test C-index | Selected test C-index | Delta | Calibrated Cox loss delta |
|---:|---:|---:|---:|---:|
| 42 | 0.743263 | 0.760866 | +0.017603 | -0.014535 |
| 43 | 0.737404 | 0.753247 | +0.015843 | -0.011890 |
| Mean | 0.740333 | **0.757056** | **+0.016723** | **-0.013212** |

两个 split 的 C-index 与校准 Cox loss 均改善。保存的 cumulative/dynamic IPCW ROC 诊断为：

| Horizon | Split 42 AUC | Split 43 AUC |
|---:|---:|---:|
| 36 | 0.867920 | 0.842278 |
| 60 | 0.832749 | 0.815883 |
| 84 | 0.834438 | 0.826573 |
| Mean | **0.845035** | **0.828245** |

### Historical exploration potential | 历史探索潜力

早期实验曾出现 `0.8967` 的探索高分。该数值保留为方法潜力与上限线索，但它来自旧探索协议，暴露复跑后降至约 `0.6905`，未通过当前固定划分、无 test 选权和跨 split 共识标准。因此：

- 可以把 `0.8967` 写作历史探索潜力；
- 不能把它写作当前正式测试成绩；
- 当前可复现的正式主结果是两 split 平均 `0.757056`。

### Topology reconstruction boundary | 拓扑推断边界

网页拓扑 Ridge 在各自 held-out 数据上的重建表现：

| Split | Function MAE | Function R2 | Edge MAE | Edge R2 |
|---:|---:|---:|---:|---:|
| 42 | 0.127786 | 0.339600 | 0.154402 | 0.312421 |
| 43 | 0.128280 | 0.337292 | 0.152709 | 0.325750 |

这些指标衡量“由网页字段重建研究表拓扑”的能力，不代表真实生物学边权测量精度。

## Research Workflow | 研究流程

基础 GNN + Cox 主干仍保留为 reference：

```powershell
python -m research.train_v2 --config research_config_v2.yaml --split-seed 42 --device cuda
python -m research.repeat_runs_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda
python -m research.graph_structure_tests_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda
```

时间拓扑 AFT 独立实验入口：

```powershell
python -m experiments.temporal_independent_v3.seed_sweep `
  --config research_config_v2.yaml `
  --mainline-predictions <split-specific-mainline-predictions.npz> `
  --split-seed 42 `
  --seeds 7 21 42 123 2026
```

正式共识汇总：

```text
outputs/current_mainline_v2/temporal_independent_v3/cross_split_consensus/cross_split_consensus_summary.json
```

详细实验说明见 `experiments/temporal_independent_v3/README.md`。

## Repository Structure | 仓库结构

```text
archive/                              分类保存旧后端、旧配置、旧模型和旧文档
config/                               网页运行配置与发布指标
ctm_fusion_experiment/                历史 CTM 实验依赖，不是当前网页主线
experiments/temporal_independent_v3/  当前时间拓扑 AFT 实验与共识工具
research/                             GNN、Cox、基线与正式研究流水线
src/                                  网页预处理、推理桥接、报告与建议
static/                               前端静态资源
templates/                            Flask 页面
tests/                                单元与接口测试
enhanced_app.py                       当前网页入口
research_config_v2.yaml               GNN reference 配置
CURRENT_MAINLINE.md                   当前主线速查
```

`outputs/` 保存本地模型和实验结果，但不提交到 GitHub。归档分类和兼容状态见 `archive/README.md`。

## Output Semantics | 输出语义

网页 `risk_score` 与 `risk_percentile` 是相对于 `topology_v6` 参考队列的百分位，不是个体绝对发病概率。主要字段包括：

- `risk_score`, `risk_level`, `risk_percentile`, `raw_model_risk`
- `split_consensus_risks`, `split_disagreement`, `prediction_reliability`
- `backend`, `model_release`, `ensemble_size`
- inferred topology、输入范围提示、建议与结构化报告

完整响应示例见 `API_RESPONSE_EXAMPLE.md`。

## Reproducibility Rules | 可复现性约束

- 固定模型种子：`7, 21, 42, 123, 2026`
- 固定 split：`42`, `43`
- 仅使用 validation 选择共识权重
- test 标签不参与模型或融合选择
- 同时报告 C-index、Cox loss 与时间依赖 AUC
- 网页 inferred-topology 结果与研究表 measured-topology 结果分开陈述
- 报告时必须注明 `topology_v6` 是 synthetic/noisy augmented 数据

## Scope and Limitations | 适用边界

当前证据支持：

- 时间拓扑 AFT 专家为 GNN reference 提供了独立信息；
- 在两个预先定义 split 上均提高 C-index 并降低校准 Cox loss；
- 图结构扰动会损害旧 GNN reference，说明图分支不是纯装饰；
- 网页已连接当前研究模型，而不是手工规则打分器。

当前证据不支持：

- 将 `0.757056` 直接称为网页部署模型的临床 C-index；
- 将 inferred edge weights 或 function scores 称为实测生物标志物；
- 从 synthetic/noisy augmented 数据推导临床诊断、处方或外部泛化结论；
- 在缺少真实独立队列和外部验证时宣称可用于临床决策。

## Funding Support | 基金支持

This research was supported by the Undergraduate Talented Innovation Education Program of Shenyang Pharmaceutical University, project XH2025-06.

本研究受沈阳药科大学本科生拔尖创新人才培养计划支持，项目编号 XH2025-06。

## Clinical Disclaimer | 临床声明

This repository is a research and demonstration platform, not a validated clinical decision system. Medically meaningful conclusions require real-cohort design, external validation, and clinician review.

本仓库仅用于研究与演示，不是经过临床验证的决策系统。任何医学结论都需要真实队列、外部验证和临床审核。
