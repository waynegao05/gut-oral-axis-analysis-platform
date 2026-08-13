# Gut-Oral Axis Analysis Platform | 肠口轴分析平台

面向右删失生存风险预测的研究与网页演示平台。当前正式网页后端使用
AC-ICAM 真实 PFS/OS 随访数据训练的 V8 PFS 模型；旧的时间拓扑模型继续保留为本地回滚和研究比较路径。

## Current Release | 当前发布

| Item | Current value |
|---|---|
| Platform release | `CTM2.0` |
| Task | right-censored survival risk prediction (`time`, `event`) |
| Default backend | `ac_icam_real_outcome_clinical_pfs` |
| Model release | `ac_icam_real_outcome_pfs_v8` |
| Primary endpoint | PFS |
| Cohort | 246 AC-ICAM patients, 71 PFS events |
| Model seeds | `7`, `21`, `42`, `123`, `2026` |
| Members | 5 full-cohort ridge-Cox deployment members |
| Default variant | clinical core |
| Optional variant | clinical + measured tumor-RNA ICR |
| Pharmacy layer | `pharmacy_assistance_v3` |
| Primary metric | C-index |
| Formal PFS C-index | 0.7756 core; 0.7845 with measured ICR |
| Formal AUC | core: 0.8185 at 36 months; 0.8013 at 60 months |
| Dataset | measured AC-ICAM clinical outcomes and paired tumor/normal 16S |

网页端输出 AC-ICAM OOF 参考分布中的相对 PFS 风险百分位、36/60 月 PFS 概率和可靠性提示。
微生物和辅助治疗字段不改变正式 V8 PFS 分数，因为五种子评估没有证明它们能提高 PFS。

## Quick Start | 快速启动

### 1. Install dependencies

```powershell
cd <repo-root>
python -m pip install -r requirements.txt
```

V8 网页推理只需要 CPU。

### 2. Run the web app

```powershell
$env:GOA_PORT = "8765"
$env:GOA_MODEL_BACKEND = "ac_icam_v8"
python enhanced_app.py
```

访问 `http://127.0.0.1:8765`。


## npm Client Package | npm 客户端包

The typed API client is published through GitHub Packages as
`@waynegao05/gut-oral-axis-client`. It contains request helpers and TypeScript
types only; datasets, generated outputs, plotting files, and trained weights are
not included.

```ini
@waynegao05:registry=https://npm.pkg.github.com
```

```bash
npm install @waynegao05/gut-oral-axis-client
```

GitHub Packages requires an authenticated token with `read:packages` for
installation. Package source and usage examples are in
[`packages/client`](packages/client).

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
    "sex": "Female",
    "stage": 3,
    "path_t": 3,
    "path_n": 1,
    "path_m": 0,
    "tumor_location": "Colon Sigmoideum",
    "tumor_morphology": "Adenocarcinoma",
    "bmi": 24.8,
    "smoking": 0,
    "family_history": 1
  },
  "metabolites": {
    "bile_acids": 0.74,
    "scfa": 0.35,
    "tryptophan_metabolism": 0.68
  },
  "metadata": {
    "current_medications": [],
    "drug_allergies": [],
    "recent_antibiotics": 0,
    "recent_probiotics": 0,
    "renal_impairment": 0,
    "hepatic_impairment": 0,
    "pregnancy": 0,
    "suspected_condition": "colorectal_cancer_followup"
  }
}
```

输入校验会直接提示非法值：

- 菌群信息不改变 V8 PFS；没有完整病理资料时，五项核心菌群全部填写才会生成独立的研究风险参考分位
- 网页只强制要求 `age` 和 `sex`；`age` 必须位于 `[18, 75]`
- AJCC 分期、病理 T/N/M、肿瘤部位和形态学均可留空，适用于没有癌症诊断或病理资料的人群
- 只有上述肿瘤病理字段全部提供时才计算 V8 PFS；缺失项不会被填成“正常”或默认分期
- `icr_score` 只接受肿瘤 RNA 实测值；病理资料完整时，提供该值会自动使用 ICR 扩展模型
- `bmi` 必须位于 `[5, 100]`
- `smoking` 与 `family_history` 只能为 `0` 或 `1`
- 药学背景的状态字段只能为 `0` 或 `1`，用药和过敏史必须为字符串列表
- 未知药学背景应省略；明确无用药或无过敏时提交空列表
- `NaN`、`Infinity`、负丰度和非数字字符串会被拒绝

## Model Architecture | 模型结构

当前网页推理链：

1. **基础输入契约**：年龄与性别必填，年龄限制为 18–75 岁；其余资料按实际情况选填。
2. **PFS 可计算性判断**：只有 AJCC 分期、病理 T/N/M、肿瘤部位和形态学全部提供时才进入 V8；否则明确返回“本次不计算 PFS”。
3. **无病理研究指数**：病理资料不完整且五项核心菌群齐全时，独立调用保留的时间-拓扑模型，输出 `0-100` 的研究参考分位及刻度图；缺失菌不按 0 处理。
4. **模型选择**：病理完整时默认使用临床核心模型；只有提供肿瘤 RNA 实测 `icr_score` 时才切换到 ICR 扩展模型。
5. **五成员 Cox 推理**：每个成员在完整 AC-ICAM PFS 队列上拟合，正则化强度由对应种子的交叉验证选择。
6. **OOF 风险校准**：将部署风险映射到严格五种子 OOF 参考分布，形成队列相对风险百分位。
7. **PFS 概率**：使用每个成员的 Breslow 基线累积风险估计 36/60 月 PFS 概率。
8. **药学辅助**：菌群与用药背景进入独立药学模块，不改变 V8 PFS 风险分数；没有病理资料时该模块仍可运行。

本地部署工件为 `config/releases/ac_icam_real_outcome_pfs_v8.json`。该文件包含训练后的模型系数，按发布策略不会上传 GitHub；可在具备获准数据的本地环境中通过以下命令重新生成：

```powershell
python -m experiments.ac_icam_real_outcome_v8.deployment
```

V8 PFS 模型用于已经确诊的 AJCC I-IV 期结直肠癌患者，不是一般人群筛查模型。36/60 月概率是队列模型估计，不是个体预后保证。无完整病理资料时返回的 `general_risk_result` 来自 synthetic/noisy augmented `topology_v6` 研究参考队列，只表示相对位置，不是患癌概率、筛查结果或诊断。

## Pharmacy Assistance v3 | 药学辅助决策层

正式网页和 `clinical_workflow.py` 现在共用 `src/pharmacy_engine.py`，不再各自维护薄弱的硬编码建议。该层提供：

- `standard / limited / withheld` 三级质量门控；
- 缺失菌不按零处理，完整五菌面板才允许菌群阈值复核；
- 当前用药、药物过敏、近期抗生素/益生菌、肝肾功能和妊娠背景校验；
- 46 个首批药物的 RxNorm 标准化与产品特异性 openFDA / DailyMed 标签证据；
- 2012 ONC 最小高危相互作用集中的 14 组可执行筛查，并显式保留 1 组 QT 规则缺口；
- 精确成分过敏命中，以及 AGA 特定适应证下的菌株级益生菌候选；
- 面向网页首屏的通俗行动摘要，并把建议分成“现在先做什么”和“后续核对”；
- 每条建议明确列出下一步准备材料、联系对象和不能自行执行的用药动作；
- 每条卡片的触发值、阈值、理由、证据等级和来源链接；
- 知识库版本、复核日期和 SHA-256 摘要；
- 明确禁止根据输出自动启停药、换药或调剂量；菌株候选也必须由临床人员核对适应证、产品、剂量和疗程。

`interaction_screening_performed` 只表示是否完成最小高危子集的成对筛查；`comprehensive_interaction_screening_performed` 仍为 `false`。说明书中的用法用量可以作为证据展示，但 `patient_specific_dose_selected` 与 `treatment_duration_selected` 固定为 `false`。完整字段契约和边界见 `PHARMACY_ASSISTANCE.md`。

网页默认只展示通俗行动和原因；规则编号、证据等级、RxCUI、SPL SET ID、模型指标与完整 JSON 收入折叠的研究/审计详情，不占据主要阅读区域。

药学背景可以省略，但只有七项均明确提供且模型可靠性通过时，状态才可能为 `standard`；缺失信息会降级为 `limited`，旧后端等无法提供可验证可靠性状态的场景会暂缓菌群阈值卡片。

## Formal Evidence | 正式证据

正式结果来自种子 `7, 21, 42, 123, 2026` 的重复外层五折评估。预处理、特征变换和正则化选择都在训练折内完成。

| V8 model | PFS C-index | Bootstrap 95% CI | AUC 36 | AUC 60 |
|---|---:|---:|---:|---:|
| Clinical core | **0.7756** | 0.7214-0.8251 | 0.8185 | 0.8013 |
| Clinical + measured ICR | **0.7845** | 0.7328-0.8323 | **0.8294** | **0.8133** |
| Clinical + microbiome safe blend | 0.7740 | 0.7197-0.8237 | 0.8177 | 0.7984 |

因此网页默认采用临床核心模型，而不是分数更低的微生物融合模型。ICR 扩展成绩只适用于具有肿瘤 RNA 实测 ICR 的输入。

### Archived temporal-topology evidence | 旧后端比较

旧 `temporal_topology_aft_cross_split_consensus_v1` 的两 split 平均 held-out C-index 为 `0.757056`。
它使用 synthetic/noisy augmented `topology_v6`，与 V8 的真实 AC-ICAM PFS 队列不是同一数据和终点协议，不能直接解释为同一测试集上的替换增益。

### Historical exploration potential | 历史探索潜力

早期实验曾出现 `0.8967` 的探索高分。该数值保留为方法潜力与上限线索，但它来自旧探索协议，暴露复跑后降至约 `0.6905`，未通过当前固定划分、无 test 选权和跨 split 共识标准。因此：

- 可以把 `0.8967` 写作历史探索潜力；
- 不能把它写作当前正式测试成绩；
- 当前 V8 可复现的正式 PFS 主结果是临床核心 `0.7756`，实测 ICR 扩展 `0.7845`。

### Archived topology reconstruction boundary | 旧拓扑推断边界

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
experiments/ac_icam_real_outcome_v8/  当前真实结局评估与部署工件生成
experiments/temporal_independent_v3/  当前时间拓扑 AFT 实验与共识工具
research/                             GNN、Cox、基线与正式研究流水线
data/pharmacy_rules_v3.json           当前版本化药学规则与证据登记表
data/pharmacy_knowledge/              RxNorm/FDA 标签、有限 DDI 与益生菌指南数据
research/build_drug_knowledge_v1.py   官方药品数据重建工具
src/                                  网页预处理、推理桥接、报告与药学辅助引擎
static/                               前端静态资源
templates/                            Flask 页面
tests/                                单元与接口测试
enhanced_app.py                       当前网页入口
research_config_v2.yaml               GNN reference 配置
CURRENT_MAINLINE.md                   当前主线速查
```

`outputs/` 保存本地模型和实验结果，但不提交到 GitHub。归档分类和兼容状态见 `archive/README.md`。

## Output Semantics | 输出语义

网页 `risk_score` 与 `risk_percentile` 是相对于 AC-ICAM 五种子 OOF PFS 风险分布的百分位，不是一般人群发病概率。主要字段包括：

- `risk_score`, `risk_level`, `risk_percentile`, `raw_model_risk`
- `pfs_probability.36`, `pfs_probability.60`, `prediction_reliability`
- `prediction_available`, `not_available_reason`, `missing_oncology_fields`
- `backend`, `model_release`, `model_variant`, `ensemble_size`
- 输入范围提示、模型适用人群、建议与结构化报告
- `pharmacy_assessment`：质量状态、用药背景、建议摘要、证据和禁止操作

完整响应示例见 `API_RESPONSE_EXAMPLE.md`。

## Reproducibility Rules | 可复现性约束

- 固定模型种子：`7, 21, 42, 123, 2026`
- 外层五折评估与内层正则化选择严格分开
- 所有预处理和模型选择只在相应训练折中拟合
- 同时报告 C-index、bootstrap 95% CI 与 36/60 月时间依赖 AUC
- 部署风险百分位使用五种子 OOF 预测作为参考分布
- ICR 扩展与常规临床核心成绩分开陈述
- V8 与 synthetic/noisy augmented V7/旧 topology_v6 结果分开陈述

## Scope and Limitations | 适用边界

当前证据支持：

- AC-ICAM 全分期临床核心模型在重复 OOF 评估中达到 PFS C-index `0.7756`；
- 实测肿瘤 RNA ICR 扩展模型达到 `0.7845`；
- 网页加载的是序列化 Cox 模型、OOF 校准和 Breslow 基线风险，不是手工规则打分；
- 微生物和治疗敏感性分支没有提高 PFS，因此未进入正式风险分数。

当前证据不支持：

- 把内部重复交叉验证称为外部临床验证；
- 把 36/60 月模型概率解释为个体预后保证；
- 把 ICR 扩展成绩用于没有肿瘤 RNA 实测 ICR 的输入；
- 把 V8 用作未确诊人群的癌症筛查或诊断；
- 将药学辅助卡片解释为已完成药物相互作用、剂量或禁忌证审核；
- 在缺少真实独立队列和外部验证时宣称可用于临床决策。

## License | 许可证

本项目采用 MIT License。具体条款请参阅 [LICENSE](LICENSE) 文件。

## Funding Support | 基金支持

本研究受沈阳药科大学本科生拔尖创新人才培养计划支持，项目编号 XH2025-06。

## Clinical Disclaimer | 临床声明

本仓库仅用于研究与演示，不是经过临床验证的决策系统。任何医学结论都需要真实队列、外部验证和临床审核。
