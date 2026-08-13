export type NumericMap = Record<string, number>;

export interface CanonicalPayload {
  microbes: NumericMap;
  clinical: Record<string, number | string>;
  metabolites: NumericMap;
  metadata?: Record<string, unknown>;
}

export type AnalysisInput = CanonicalPayload | Record<string, unknown>;

export interface SuccessEnvelope {
  ok: true;
}

export interface FailureEnvelope {
  ok: false;
  errors: string[];
}

export interface StandardizeResponse extends SuccessEnvelope {
  source_format: string;
  standardized_payload: CanonicalPayload;
}

export interface MainAnalysisResponse extends SuccessEnvelope {
  source_format: string;
  standardized_payload: CanonicalPayload;
  report: Record<string, unknown>;
  risk_result: Record<string, unknown>;
  general_risk_result: Record<string, unknown>;
  recommendations: Array<Record<string, unknown>>;
  pharmacy_assessment: Record<string, unknown>;
  top_microbes: Array<[string, number]>;
  saved_to: string;
}

export interface OralAdenomaSchemaResponse extends SuccessEnvelope {
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

export type OralSampleType =
  | "oral"
  | "oral_swab"
  | "buccal_swab"
  | "saliva";

export interface OralAdenomaRequest {
  sample_type: OralSampleType;
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

export interface OralAdenomaAnalysisResponse extends SuccessEnvelope {
  oral_adenoma_result: OralAdenomaResult;
}
