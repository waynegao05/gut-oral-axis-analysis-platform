# Pharmacy Assistance v3 | 药学辅助决策层

`pharmacy_assistance_v3` 是当前网页与临床工作流共用的研究型药学辅助层。它把模型风险、输入完整性、菌群校准值、用药背景、RxNorm 药名标准化、FDA 产品说明书和有限高危相互作用规则组织成可审计的复核结果。它不诊断疾病，不把说明书文字转换成患者处方，也不授权启药、停药、换药或调整剂量。

## Active Path | 当前链路

```text
canonical/raw clinical input
  -> field validation and standardization
  -> temporal-topology risk model
  -> model reliability and input-quality gate
  -> medication-context review
  -> RxNorm medication normalization
  -> product-specific openFDA / DailyMed label lookup
  -> minimum high-priority interaction and exact-ingredient allergy screening
  -> indication-gated probiotic guideline options
  -> calibrated marker review
  -> evidence-linked recommendation cards
```

统一实现位于：

- `src/pharmacy_engine.py`：质量门控、建议卡片和审计字段
- `src/drug_knowledge.py`：药名标准化、标签证据、有限相互作用和益生菌规则执行
- `data/pharmacy_rules_v3.json`：版本化阈值、规则和证据登记表
- `data/pharmacy_knowledge/`：药物种子、46 个标签快照、15 组最小高危 DDI 与益生菌指南规则
- `research/build_drug_knowledge_v1.py`：从 RxNorm 与 openFDA 官方接口重建标签数据库
- `src/pipeline.py`：正式网页模型与药学层的集成点
- `src/pharmacy_advice.py`、`src/recommendation.py`：旧调用方的兼容适配器

旧的三条原型规则已归档到 `archive/legacy_configs/microbe_drug_rules_prototype.json`，不再由正式流程读取。

## Input Contract | 输入契约

药学层使用模型已有的 `microbes`、`clinical` 字段，并接受可选 `metadata`：

```json
{
  "metadata": {
    "current_medications": ["metformin 500 mg twice daily"],
    "drug_allergies": ["penicillin: rash"],
    "recent_antibiotics": 0,
    "recent_probiotics": 0,
    "renal_impairment": 0,
    "hepatic_impairment": 0,
    "pregnancy": 0,
    "suspected_condition": "gut_risk_screening"
  }
}
```

- `current_medications` 与 `drug_allergies` 必须是字符串列表；明确报告“无”时使用空列表。
- 其余字段只能为 `0` 或 `1`；未知时省略字段，不得用 `0` 代替未知。
- 当前用药应包含处方药、非处方药和补充剂；过敏史建议同时记录反应类型。
- `suspected_condition` 仅用于匹配已登记的指南情境；它不是系统诊断。未由临床确认时使用 `gut_risk_screening`。
- `metadata` 可省略，但七项背景全部明确提供后才可能达到 `standard`；缺项会返回 `limited` 和具体字段名。

原始临床 JSON 也可通过 `medication_context` 和 `history` 提供这些字段。字符串形式的“无”、`none`、`no known allergies` 会被规范为已报告的空列表。

## Quality Gate | 质量门控

药学层不会在输入不可靠时照常输出菌群驱动结论。

| Status | Meaning | Main behavior |
|---|---|---|
| `standard` | 输入完整，模型未触发可靠性告警 | 可生成研究性复核卡片 |
| `limited` | 缺少面板/用药信息、存在默认值、split disagreement 或近期抗生素等 | 明确列出原因，限制解释强度 |
| `withheld` | 输入超出训练范围、没有校准值或后端可靠性状态不可验证 | 暂缓全部菌群阈值卡片 |

重要修正：

- 缺失菌种不再被当作零丰度。
- 菌群阈值只在五菌完整面板且后端提供校准值时计算。
- 阈值基于完整五菌面板的组成比例，不使用模型内部的总量放大与截断值。
- 近期抗生素或任何未确认的药学背景都会在状态和原因中显式呈现。
- 所有卡片均为 `clinician_review_only`，并固定 `allows_medication_change = false`。

## Marker Calibration | 菌群校准

当前三条菌群复核规则使用 `topology_v6` 参考队列的经验分位数：

| Marker | Trigger | Threshold | Interpretation |
|---|---:|---:|---|
| `Fusobacterium` | `> q75` | `0.268132` | 高组成比例复核，不构成抗炎或抗菌适应证 |
| `Porphyromonas` | `> q75` | `0.252460` | 口腔炎症、采样和抗菌药暴露复核 |
| `Lactobacillus` | `< q25` | `0.103096` | 低组成比例复核，不自动推荐益生菌 |

每个样本先按“该菌丰度 / 五个必需菌丰度总和”转换为面板内组成比例，再与同样变换后的 `topology_v6` 参考分位数比较。这样总量缩放不会改变触发结果，也不会使用模型内部放大和截断后的值。

这些是 `topology_v6` synthetic/noisy augmented 研究数据上的内部探索阈值，不是临床诊断界值。每个触发卡片同时保留用户提交丰度、面板组成比例、阈值、分位数、数据集、来源表和证据等级。

## Output Contract | 输出契约

`/analyze` 在顶层和 `report` 内返回同一份 `pharmacy_assessment`。主要字段包括：

- `engine_version`、`knowledge_schema_version`、`knowledge_last_reviewed`
- `knowledge_sha256`：规则文件内容摘要，用于审计版本漂移
- `status`、`status_label`、`quality.status_reasons`
- `risk_context`、`medication_context`
- `drug_knowledge.normalization`：原始药名、匹配状态、标准成分和 RXCUI
- `drug_knowledge.label_lookup`：产品特异性标签身份、章节摘录、SPL SET ID 与 DailyMed 链接
- `drug_knowledge.interaction_screening`：最小高危子集的筛查范围、命中和未覆盖规则
- `drug_knowledge.allergy_screening`：仅限精确成分的过敏命中；不声称完成同类交叉反应审核
- `drug_knowledge.probiotic_decision_support`：适应证限定的菌株候选或“不常规推荐”结论
- `drug_knowledge.candidate_therapy_support`：具体药物、剂量和疗程是否因信息不足被暂缓
- `plain_language_summary`：面向网页首屏的通俗摘要，包含当前结论、优先事项、已核对范围、未核对范围和安全提示
- `summary`：建议数、菌群触发数、优先复核数、药学背景完整性和安全状态
- `recommendations`：按优先级排序的结构化卡片；每张卡同时提供可直接展示的 `action_steps`
- `evidence_sources`：本次实际使用的证据条目
- `prohibited_actions`、`disclaimer`

每张建议卡片至少包含：

```json
{
  "recommendation_id": "fusobacterium_upper_quartile_review",
  "category": "marker_review",
  "title": "Fusobacterium 偏高，建议重点核对肠道症状与筛查记录",
  "suggestion": "记录近期消化道症状、既往肠道检查和家族史，并交给消化专科核对是否需要进一步评估。",
  "action_steps": [
    "记录近期消化道症状、体重变化和症状开始时间。",
    "准备既往肠道检查结果、结直肠癌筛查记录和家族史。",
    "把资料交给消化专科判断是否需要进一步评估。"
  ],
  "rationale": "单项菌群偏高不能证明感染、肿瘤或需要用药。",
  "priority": 0.9,
  "urgency": "priority",
  "urgency_label": "优先处理",
  "evidence_level": "internal_exploratory",
  "evidence_source_ids": ["INTERNAL_TOPOLOGY_V6", "FDA_CDS_2026"],
  "submitted_abundance": 0.18,
  "panel_composition": 0.333333,
  "trigger": {
    "operator": "gt",
    "threshold": 0.268132,
    "value_scale": "five_marker_panel_composition"
  },
  "requires_clinician_review": true,
  "allows_medication_change": false
}
```

根级 `recommendations` 保留为兼容别名，内容来自同一引擎，不再维护第二套规则。

网页按“行动摘要 → 现在先做什么 → 后续核对 → 当前用药资料”的顺序展示。证据等级、规则编号、RxCUI、SPL SET ID、模型指标和完整 JSON 默认折叠，仅在研究或审计时展开。说明书匹配本身不再作为一条建议卡片，避免把“有资料可看”误解成临床行动。

## Evidence Governance | 证据治理

知识库登记 FDA 临床决策支持指导、openFDA 标签、DailyMed、NLM RxNorm、ONC 高优先级 DDI 共识、CDC 抗菌药管理、AGA 益生菌指南、USPSTF 结直肠癌筛查和 WHO 用药核对资料。加载时会校验规则引用的证据 ID、标签记录 SHA-256 与整库摘要；响应只返回本次卡片实际使用的来源。

内部菌群阈值与外部指南严格分层：外部指南用于安全边界和临床复核背景，不被包装成对本项目菌群阈值的临床验证。

标签库当前包含 46 个经过给药途径匹配的成分记录。每条记录保存 RxNorm 编码、openFDA 查询、产品与生产者信息、标签生效日期、SPL SET ID、DailyMed 原始链接和关键标签章节。标签是具体产品的证据快照，不保证覆盖同一成分的全部厂家、剂型或途径。

相互作用层实现 2012 ONC 专家共识最小集合中的 14 组规则；第 15 组高危 QT 延长药物联用因完整动态成员表未获可再分发来源而明确标为未实现。即使返回零命中，也不能解释为“没有相互作用”。

## Current Limitations | 当前尚未具备

- 不是全面、实时、机构认可的药物相互作用数据库；当前仅筛查一个最小高危历史子集。
- 未覆盖药物-疾病、药物-食物、同类交叉过敏、重复用药、QT 风险全表和所有药物-菌群相互作用。
- 说明书章节可以显示，但系统不会根据二元“肾/肝功能异常”选择患者剂量；仍缺少 eGFR、肝功能分级、体重、适应证、实验室值和治疗目标。
- 具体药物候选目前固定为不自动生成；现有风险模型或菌群标志物不能单独建立药物适应证。
- 益生菌只在 AGA 已登记的特定情境显示菌株候选，且不选择产品、剂量或疗程，不允许用其他同属产品替代。
- 没有真实独立临床队列上的前瞻验证、药师一致性研究或结局获益证据。
- 未接入机构处方集、电子病历、实验室系统、实时药物警戒或商业药品知识库。

`interaction_screening_performed = true` 仅表示至少两项用药经过最小高危子集筛查；`comprehensive_interaction_screening_performed` 仍固定为 `false`。两个字段必须一起解释。

## Verification | 验证

```powershell
python -m research.build_drug_knowledge_v1
python -m research.rebuild_pharmacy_calibration_v2
python -m pytest -q tests/test_build_drug_knowledge_v1.py `
  tests/test_drug_knowledge.py `
  tests/test_pharmacy_engine.py `
  tests/test_clinical_standardizer.py `
  tests/test_pipeline.py `
  tests/test_app_validation.py
```

第一条命令联网重建 46 个药物的版本化标签快照，任何精确药名或给药途径匹配失败默认都会阻止写入。第二条命令以只读方式从源表复算菌群阈值。测试覆盖哈希完整性、药名与品牌名标准化、高危 DDI 命中、精确成分过敏、标签剂量非处方边界、益生菌适应证门控、菌群阈值复算、网页响应和临床报告复用。
