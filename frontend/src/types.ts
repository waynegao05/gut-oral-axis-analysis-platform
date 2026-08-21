/**
 * 后端响应契约。
 *
 * 字段依据以下后端实现逐个核对得出，不是猜测：
 *   enhanced_app.py / src/pipeline.py / src/report.py /
 *   src/ac_icam_v8_bridge.py / src/pharmacy_engine.py / src/drug_knowledge.py
 *
 * 约定：
 * - 精确契约（MainAnalysisResponse 及其成员）描述后端在
 *   WEB_MODEL_BACKEND="ac_icam_v8" 下实际产出的结构，用于文档与出入口的类型收敛。
 * - 视图类型（*View）是各渲染函数的入参：字段全部可选，用来保留旧实现对
 *   缺字段的防御性回退（`x || {}`、`?.`），同时仍然拦住键名拼写错误。
 * - 仍标注为 unknown 的字段，来源是 data/ 下的 JSON 知识库而非 Python 代码，
 *   无法从实现推断字面量；前端也不消费它们。
 */

export type NumericMap = Record<string, number>;

/* ------------------------------------------------------------------ *
 * 1. 信封
 * ------------------------------------------------------------------ */

/**
 * Flask 后端成功响应恒为 `{ok: true, ...}`，失败恒为 `{ok: false, errors: [...]}`
 * 并带 HTTP 400/500（enhanced_app.py:145,150,167,172,180,183,186）。
 *
 * status / error_code / message / request_id 由桌面 WebView2 宿主可能补充，
 * Flask 路径下恒为 undefined —— api.ts 的 decodeResponse 已兼容两者。
 */
export interface ApiEnvelope {
  ok?: boolean;
  status?: "ok" | "success" | "error";
  errors?: string[];
  error_code?: string;
  message?: string;
  request_id?: string;
}

/* ------------------------------------------------------------------ *
 * 2. 字面量联合
 * ------------------------------------------------------------------ */

/** enhanced_app.py:44,46 —— 仅这两种。 */
export type SourceFormat = "canonical" | "raw_standardized";

/** ac_icam_v8_bridge.py:624-628（计算成功）/ :208,292,383（不可用）。 */
export type RiskLevel = "low" | "medium" | "high" | "not_available";

/** ac_icam_v8_bridge.py:632-638 / :211 / :295。 */
export type PredictionReliability =
  | "standard"
  | "caution_out_of_training_range"
  | "caution_defaulted_inputs"
  | "caution_split_disagreement"
  | "not_applicable_missing_oncology"
  | "not_applicable_incomplete_microbiome";

/** ac_icam_v8_bridge.py:213 / :354 / :364 / :386。 */
export type NotAvailableReason =
  | "missing_oncology_fields"
  | "incomplete_microbiome_panel"
  | "invalid_microbiome_panel"
  | "out_of_training_range";

/** ac_icam_v8_bridge.py:219,231（PFS）/ :301,392（研究风险指数）。 */
export type Endpoint = "PFS" | "research_risk_index";

/** ac_icam_v8_bridge.py:218,230,300,391,595,657。 */
export type ModelVariant =
  | "clinical_core"
  | "clinical_icr"
  | "not_calculated"
  | "temporal_topology_research_percentile";

/** pharmacy_engine.py:449；标签映射见 :310-314。 */
export type PharmacyStatus = "standard" | "limited" | "withheld";

/**
 * pharmacy_engine.py:496 —— 内置卡片只产出这两个值。
 * 注意：marker 规则卡的 urgency 来自 JSON 知识库，理论上可能是别的字符串，
 * 此时 :497 的 urgency_label 会退化为「后续核对」，前端也会归入 routine 列。
 * 因此 RecommendationCard.urgency 用 string，此联合仅作文档。
 */
export type Urgency = "priority" | "routine";

/* ------------------------------------------------------------------ *
 * 3. 提交载荷
 * ------------------------------------------------------------------ */

/**
 * 表单/JSON 提交给 /standardize 与 /analyze 的标准结构。
 * clinical 允许 null：吸烟、家族史两个下拉在留空时会写入 null
 * （与旧实现一致，见 form/read.ts）。
 */
export interface CanonicalPayload {
  microbes: NumericMap;
  clinical: Record<string, number | string | null>;
  metabolites: NumericMap;
  metadata?: Record<string, unknown>;
}

/**
 * /standardize 与 /analyze 回传的标准化载荷。
 * canonical 分支原样透传提交内容（enhanced_app.py:44）；
 * raw_standardized 分支恒为 clinical_standardizer.py:256-261 的四键结构。
 */
export interface StandardizedPayload {
  microbes?: Record<string, number> | undefined;
  clinical?: Record<string, unknown> | undefined;
  metabolites?: Record<string, unknown> | undefined;
  metadata?: Record<string, unknown> | undefined;
}

/* ------------------------------------------------------------------ *
 * 4. 风险结果
 * ------------------------------------------------------------------ */

export interface PfsProbability {
  "36": number | null;
  "60": number | null;
}

/** V8 PFS 计算成功。ac_icam_v8_bridge.py:647-672，恰好 16 键。 */
export interface V8PfsRiskResult {
  risk_score: number;
  risk_level: "low" | "medium" | "high";
  risk_percentile: number;
  raw_model_risk: number;
  prediction_reliability: PredictionReliability;
  /** = !out_of_training_range，可能为 false。 */
  prediction_available: boolean;
  ensemble_size: number;
  backend: string;
  model_release: string;
  model_variant: "clinical_core" | "clinical_icr";
  endpoint: "PFS";
  pfs_probability: PfsProbability;
  progression_probability: PfsProbability;
  time_horizon_unit: "months";
  intended_use: string;
  research_use_only: true;
}

/** 缺任一肿瘤病理字段。ac_icam_v8_bridge.py:206-225，恰好 18 键。 */
export interface V8MissingOncologyRiskResult {
  risk_score: null;
  risk_level: "not_available";
  risk_percentile: null;
  raw_model_risk: null;
  prediction_reliability: "not_applicable_missing_oncology";
  prediction_available: false;
  not_available_reason: "missing_oncology_fields";
  /** 形如 "clinical.stage"。 */
  missing_oncology_fields: string[];
  ensemble_size: 0;
  backend: string;
  model_release: string;
  model_variant: "not_calculated";
  endpoint: "PFS";
  pfs_probability: PfsProbability;
  progression_probability: PfsProbability;
  time_horizon_unit: "months";
  intended_use: string;
  research_use_only: true;
}

export type RiskResult = V8PfsRiskResult | V8MissingOncologyRiskResult;

/** 菌群面板不全或总和为 0。ac_icam_v8_bridge.py:290-317。 */
export interface GeneralRiskUnavailable {
  risk_score: null;
  risk_level: "not_available";
  risk_percentile: null;
  raw_model_risk: null;
  prediction_reliability: "not_applicable_incomplete_microbiome";
  prediction_available: false;
  not_available_reason: "incomplete_microbiome_panel" | "invalid_microbiome_panel";
  backend: "temporal_topology_aft_cross_split_consensus";
  model_release: string;
  model_variant: "temporal_topology_research_percentile";
  endpoint: "research_risk_index";
  display_name: string;
  risk_kind: "research_cohort_percentile";
  score_unit: "reference_percentile_0_100";
  /** 恒为五项核心菌属。 */
  required_microbes: string[];
  /** 形如 "microbes.Fusobacterium"；invalid 分支为 []。 */
  missing_microbe_fields: string[];
  absolute_cancer_probability: false;
  screening_result: false;
  pfs_calculated: false;
  dataset_version: "topology_v6";
  dataset_is_synthetic_noisy_augmented: true;
  intended_use: string;
  research_use_only: true;
}

/**
 * temporal_topology 真正打分。ac_icam_v8_bridge.py:377-409。
 * 基础结果来自 temporal_topology_bridge.score()，其完整键集不由前端约束，
 * 故保留索引签名。
 */
export interface GeneralRiskComputed {
  risk_score: number | null;
  risk_level: RiskLevel;
  risk_percentile: number | null;
  raw_model_risk: number | null;
  prediction_reliability: string;
  prediction_available: boolean;
  not_available_reason?: "out_of_training_range";
  model_variant: "temporal_topology_research_percentile";
  endpoint: "research_risk_index";
  display_name: string;
  risk_kind: "research_cohort_percentile";
  score_unit: "reference_percentile_0_100";
  required_microbes: string[];
  missing_microbe_fields: string[];
  absolute_cancer_probability: false;
  screening_result: false;
  pfs_calculated: false;
  dataset_version: "topology_v6";
  dataset_is_synthetic_noisy_augmented: true;
  intended_use: string;
  research_use_only: true;
}

/** PFS 计算成功时后端回传 `{}`（report.py:19）。 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface EmptyRiskResult {}

export type GeneralRiskResult =
  | EmptyRiskResult
  | GeneralRiskUnavailable
  | GeneralRiskComputed;

/* ------------------------------------------------------------------ *
 * 5. 建议卡片
 * ------------------------------------------------------------------ */

/** pharmacy_engine._make_card :489-517。 */
export interface RecommendationCard {
  recommendation_id: string;
  category: string;
  title: string;
  suggestion: string;
  rationale: string;
  /** clamp 到 [0,1] 后 round(...,4)。 */
  priority: number;
  /** 内置卡恒为 Urgency；marker 规则卡由知识库决定，故放宽为 string。 */
  urgency: string;
  urgency_label: string;
  action_type: "clinician_review_only";
  marker: string;
  evidence_level: string;
  evidence_source_ids: string[];
  requires_clinician_review: true;
  allows_medication_change: false;
  /** 为空时后端回退为 [suggestion]。 */
  action_steps: string[];

  /** 仅 marker 规则卡。 */
  marker_value?: number;
  panel_composition?: number;
  submitted_abundance?: number;
  trigger?: Record<string, unknown>;

  /** 仅 DDI 告警卡 / 过敏命中卡。 */
  interaction_match?: Record<string, unknown>;
  allergy_match?: Record<string, unknown>;

  /** 仅益生菌卡。 */
  probiotic_candidates?: ProbioticCandidate[];
  independent_of_model_result?: true;
  decision_basis?: string;
}

/** drug_knowledge.py:616-626。candidate_id 来自益生菌知识库，实测为字符串。 */
export interface ProbioticCandidate {
  candidate_id: string;
  strains: string[];
  candidate_type: "guideline_option_for_clinician_review";
  dose_selected: false;
  duration_selected: false;
  product_interchangeable: false;
  requires_clinician_review: true;
  allows_automatic_start_or_stop: false;
}

/** 知识库校验器强制这些键存在（pharmacy_engine.py:200-208）；year 未校验类型。 */
export interface EvidenceSource {
  source_id: string;
  organization: string;
  title: string;
  year: unknown;
  source_type: string;
  url: string;
  scope_note: string;
}

/* ------------------------------------------------------------------ *
 * 6. 用药评估
 * ------------------------------------------------------------------ */

export interface QualityStatusReason {
  code: string;
  message: string;
}

/** pharmacy_engine._quality_context :450-469。 */
export interface PharmacyQuality {
  status: PharmacyStatus;
  status_label: string;
  status_reasons: QualityStatusReason[];
  required_marker_panel: string[];
  observed_markers: string[];
  missing_markers: string[];
  panel_completeness: number;
  calibration_ready: boolean;
  calibration_available: boolean;
  calibration_scale: unknown;
  calibration_normalization: unknown;
  panel_abundance_total: number;
  calibrated_marker_values: Record<string, number>;
  model_reliability: string;
  defaulted_inputs: string[];
  out_of_training_range_inputs: string[];
  out_of_training_range_details: Array<Record<string, unknown>>;
  /** ac_icam_v8 后端从不产出该键 ⇒ 恒为 []。 */
  unsupported_microbes_ignored: string[];
}

/** pharmacy_engine.py:1402-1414。 */
export interface RiskContext {
  risk_level: string;
  risk_percentile: number | null;
  prediction_reliability: string;
  prediction_available?: boolean;
  not_available_reason: string | null;
  /** 实测恒为 null：桥把成员分歧写在 model_features.member_disagreement。 */
  split_disagreement: number | null;
  model_release: string | null;
}

/** pharmacy_engine.py:885-912 + :1336-1355 覆盖。 */
export interface MedicationContext {
  provided_fields: string[];
  missing_fields: string[];
  context_completeness: number;
  medication_list_available: boolean;
  allergy_history_available: boolean;
  current_medications: string[];
  drug_allergies: string[];
  renal_impairment: boolean;
  hepatic_impairment: boolean;
  pregnancy: boolean;
  recent_antibiotics: boolean;
  recent_probiotics: boolean;
  interaction_screening_performed: boolean;
  interaction_screening_scope: string | null;
  comprehensive_interaction_screening_performed: false;
  interaction_screening_note: string | null;
  label_lookup_performed: boolean;
  label_record_count: number;
  drug_name_normalization_coverage: number | null;
  drug_knowledge_available: boolean;
}

/** pharmacy_engine._plain_language_summary :1290-1299。 */
export interface PlainLanguageSummary {
  headline: string;
  urgent_count: number;
  routine_count: number;
  independent_guidance_count: number;
  /** 最多 4 条。 */
  what_to_do_now: string[];
  what_was_checked: string[];
  what_was_not_checked: string[];
  safety_note: string;
}

/** pharmacy_engine.build_pharmacy_assessment.summary :1423-1453。 */
export interface PharmacySummary {
  recommendation_count: number;
  marker_trigger_count: number;
  priority_card_count: number;
  medication_history_complete: boolean;
  medication_context_complete: boolean;
  interaction_screening_performed: boolean;
  interaction_screening_scope: string | null;
  comprehensive_interaction_screening_performed: false;
  high_priority_interaction_match_count: number;
  label_lookup_performed: boolean;
  label_record_count: number;
  medication_candidate_generated: false;
  patient_specific_dose_selected: false;
  treatment_duration_selected: false;
  probiotic_candidate_count: number;
  medication_change_allowed: false;
}

/* --------------------------- 说明书证据 --------------------------- */

/** drug_knowledge.py:333-337 + :357。 */
export interface LabelSection {
  excerpt: string;
  truncated: boolean;
  full_character_count: number;
  label_zh: string;
}

/** drug_knowledge.py:374-383。 */
export interface LabelIdentity {
  generic_names?: string[];
  brand_names?: string[];
  manufacturer_names?: string[];
  product_types?: string[];
  routes?: string[];
  spl_set_id?: string | null;
  spl_version?: string | null;
  effective_time?: string | null;
}

/** drug_knowledge.py:384-389。 */
export interface LabelSource {
  source_id?: "OPENFDA_LABEL_CURRENT";
  openfda_query_url?: string | null;
  dailymed_url?: string | null;
  record_sha256?: string | null;
}

/** drug_knowledge._label_evidence_for_record :368-402。 */
export interface LabelRecord {
  input: string;
  drug_id: string;
  display_name: string;
  review_prompt: string;
  rxcui: string | null;
  label_identity?: LabelIdentity;
  source?: LabelSource;
  /** 只包含正文非空的章节，空章节整个不出现（drug_knowledge.py:355-358）。 */
  sections?: Record<string, LabelSection | undefined>;
  dose_and_course_reference?: Record<string, unknown>;
  product_specific_label?: true;
  allows_medication_change?: false;
}

export interface LabelLookup {
  performed?: boolean;
  source_scope?: string;
  record_count?: number;
  records?: LabelRecord[];
  unmatched_inputs?: string[];
  negative_result_excludes_warning_or_contraindication?: false;
}

export interface NormalizationSummary {
  input_count?: number;
  matched_count?: number;
  coverage?: number | null;
  medications?: Array<Record<string, unknown>>;
  allergies?: Array<Record<string, unknown>>;
}

/**
 * drug_knowledge.screen_high_priority_interactions :503-524；
 * DDI 库不可用时只有 6 个键（:725-732），故此处字段全部可选。
 */
export interface InteractionScreening {
  interaction_screening_performed?: boolean;
  screening_status?: string;
  screening_scope?: string;
  comprehensive_interaction_screening_performed?: false;
  negative_result_excludes_other_interactions?: false;
  input_medication_count?: number;
  resolved_for_subset_count?: number;
  screened_pair_count?: number;
  normalization_coverage?: number | null;
  source_rule_count?: number;
  implemented_rule_count?: number;
  unsupported_rule_ids?: string[];
  match_count?: number;
  matches?: Array<Record<string, unknown>>;
  source?: Record<string, unknown>;
  note?: string;
  /** 仅 DDI 库不可用分支。 */
  status?: string;
  error?: string;
}

/**
 * drug_knowledge.build_drug_knowledge_review。
 * 数据库读取失败时（:677-695）只有 7 个顶层键，其余全部缺失，
 * 因此除 available 外一律可选。
 */
export interface DrugKnowledgeReview {
  available?: boolean;
  status?: string;
  error?: string;
  database?: Record<string, unknown>;
  normalization?: NormalizationSummary;
  label_lookup?: LabelLookup;
  interaction_screening?: InteractionScreening;
  allergy_screening?: Record<string, unknown>;
  probiotic_decision_support?: Record<string, unknown>;
  candidate_therapy_support?: Record<string, unknown>;
  safety_boundary?: Record<string, unknown>;
  evidence_source_ids?: string[];
}

/** pharmacy_engine.build_pharmacy_assessment :1393-1464。 */
export interface PharmacyAssessment {
  engine_version: unknown;
  knowledge_schema_version: unknown;
  knowledge_last_reviewed: unknown;
  knowledge_sha256: string;
  intended_use: unknown;
  status: PharmacyStatus;
  status_label: string;
  quality: PharmacyQuality;
  risk_context: RiskContext;
  medication_context: MedicationContext;
  drug_knowledge: DrugKnowledgeReview;
  plain_language_summary: PlainLanguageSummary;
  summary: PharmacySummary;
  recommendations: RecommendationCard[];
  /** 只含被卡片引用到的条目（pharmacy_engine.py:1375-1382）。 */
  evidence_sources: EvidenceSource[];
  prohibited_actions: string[];
  disclaimer: string;
}

/* ------------------------------------------------------------------ *
 * 7. 顶层响应
 * ------------------------------------------------------------------ */

/** report.py:14,16 —— Python tuple 序列化为二元数组，按丰度降序取前 10。 */
export type TopMicrobe = [name: string, abundance: number];

/** src/report.py:15-22，恰好 6 键。 */
export interface AnalysisReport {
  top_microbes: TopMicrobe[];
  /** pipeline.py:59-61；前端不消费，仅整体透传给桌面保存。 */
  gnn_features: Record<string, unknown>;
  risk_result: RiskResult;
  general_risk_result: GeneralRiskResult;
  recommendations: RecommendationCard[];
  pharmacy_assessment: PharmacyAssessment;
}

/** POST /standardize 成功响应，enhanced_app.py:152-158。 */
export interface StandardizeResponse extends ApiEnvelope {
  source_format: SourceFormat;
  standardized_payload: StandardizedPayload;
}

/** POST /analyze 成功响应，enhanced_app.py:184-197。 */
export interface MainAnalysisResponse extends ApiEnvelope {
  source_format: SourceFormat;
  standardized_payload: StandardizedPayload;
  report: AnalysisReport;
  risk_result: RiskResult;
  general_risk_result: GeneralRiskResult;
  recommendations: RecommendationCard[];
  pharmacy_assessment: PharmacyAssessment;
  top_microbes: TopMicrobe[];
  /** 桌面端会被前端用 SaveReportResult.display_location 覆盖。 */
  saved_to: string;
}

/* ------------------------------------------------------------------ *
 * 8. 渲染视图类型
 *
 * 渲染函数不直接吃精确契约，而是吃这些「字段全可选」的视图：
 * 既保留旧实现对缺字段的容忍，又能拦住键名写错。
 * 精确契约可以直接赋值给对应视图。
 * ------------------------------------------------------------------ */

/** renderRiskBanner / renderRiskScale / renderV8OutcomeSummary 的入参。 */
export interface RiskResultView {
  risk_score?: number | null;
  risk_level?: string;
  risk_percentile?: number | null;
  prediction_reliability?: string;
  prediction_available?: boolean;
  not_available_reason?: string;
  missing_microbe_fields?: string[];
  missing_oncology_fields?: string[];
  model_variant?: string;
  model_release?: string;
  endpoint?: string;
  pfs_probability?: PfsProbability;
  intended_use?: string;
}

/** 各 pharmacy 渲染函数的入参。 */
export interface PharmacyAssessmentView {
  engine_version?: unknown;
  status?: string;
  status_label?: string;
  quality?: Partial<PharmacyQuality>;
  medication_context?: Partial<MedicationContext>;
  drug_knowledge?: DrugKnowledgeReview;
  plain_language_summary?: Partial<PlainLanguageSummary>;
  recommendations?: RecommendationCard[];
  evidence_sources?: EvidenceSource[];
}

/* ------------------------------------------------------------------ *
 * 9. 内部口腔腺瘤筛查（oral-adenoma.ts 使用，结构未变）
 * ------------------------------------------------------------------ */

export interface OralAdenomaSchemaResponse extends ApiEnvelope {
  model_release: string;
  research_only: true;
  input_unit: "percent";
  required_sum_range_percent: [number, number];
  feature_count: number;
  feature_ids: string[];
  taxonomies: string[];
  accepted_sample_types: string[];
  claim_boundary: string;
}

export interface OralAdenomaRequest {
  sample_type: "oral" | "oral_swab" | "buccal_swab" | "saliva";
  oral_abundances: NumericMap;
}

export interface RateMetric {
  value: number;
  numerator: number;
  denominator: number;
  ci95_wilson: [number, number];
}

export interface AucMetric {
  value: number;
  ci95_stratified_bootstrap: [number, number];
  positive_n: number;
  negative_n: number;
}

export interface OralAdenomaResult {
  prediction_available: true;
  endpoint: "oral_microbiome_adenoma_screening_research";
  model_release: string;
  sample_type: string;
  adenoma_probability: number;
  operating_threshold: number;
  screen_positive: boolean;
  result_label: string;
  formal_internal_metrics: {
    adenoma_sensitivity: RateMetric;
    false_positive_rate: RateMetric;
    specificity: RateMetric;
    roc_auc: AucMetric;
  };
  selected_taxonomies: string[];
  research_only: true;
  not_diagnostic: true;
  verified_diminutive_adenoma_le_5mm: false;
  claim_boundary: string;
}

export interface OralAdenomaAnalysisResponse extends ApiEnvelope {
  oral_adenoma_result: OralAdenomaResult;
}
