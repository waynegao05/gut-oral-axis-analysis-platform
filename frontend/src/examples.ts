import type { CanonicalPayload } from "./types";

/** 页面「载入标准示例」与「重置为标准示例」使用的标准结构样例。 */
export const CANONICAL_EXAMPLE: CanonicalPayload = {
  microbes: {
    Fusobacterium: 0.18,
    Porphyromonas: 0.15,
    Prevotella: 0.1,
    Streptococcus: 0.09,
    Lactobacillus: 0.02,
  },
  clinical: {
    age: 52,
    sex: "Female",
    stage: 3,
    path_t: 3,
    path_n: 1,
    path_m: 0,
    tumor_location: "Colon Sigmoideum",
    tumor_morphology: "Adenocarcinoma",
    bmi: 24.5,
    smoking: 1,
    family_history: 1,
  },
  metabolites: {
    bile_acids: 0.8,
    scfa: 0.3,
    tryptophan_metabolism: 0.7,
  },
  metadata: {
    current_medications: [],
    drug_allergies: [],
    recent_antibiotics: 0,
    recent_probiotics: 0,
    renal_impairment: 0,
    hepatic_impairment: 0,
    pregnancy: 0,
    suspected_condition: "colorectal_cancer_followup",
  },
};

/** 「载入临床示例」使用的原始临床 JSON 样例，需经 /standardize 才能回填表单。 */
export const RAW_CLINICAL_EXAMPLE = {
  sample_id: "DEMO-001",
  demographics: {
    age: 55,
    bmi: 25.3,
    sex: "female",
  },
  history: {
    smoking: "yes",
    family_history_colorectal_or_ibd: "positive",
    recent_antibiotics: "no",
    recent_probiotics: "yes",
  },
  medication_context: {
    current_medications: ["metformin 500 mg twice daily"],
    drug_allergies: ["penicillin: rash"],
    renal_impairment: "no",
    hepatic_impairment: "no",
    pregnancy: "no",
  },
  oncology: {
    stage: 3,
    path_t: 3,
    path_n: 1,
    path_m: 0,
    tumor_location: "Colon Sigmoideum",
    tumor_morphology: "Adenocarcinoma",
  },
  oral_microbiome: {
    taxa: [
      { taxon: "Fusobacterium", abundance: 0.16 },
      { taxon: "Porphyromonas", abundance: 0.13 },
      { taxon: "Prevotella", abundance: 0.08 },
      { taxon: "Streptococcus", abundance: 0.07 },
      { taxon: "Lactobacillus", abundance: 0.03 },
    ],
  },
  metabolites: {
    bile_acids: 0.74,
    scfa: 0.31,
    tryptophan_metabolism: 0.68,
  },
  clinical_context: {
    chief_complaint: "recurrent abdominal discomfort",
    suspected_condition: "colorectal_cancer_followup",
  },
} as const;

/** 表单上固定存在的五项核心菌属，其余菌种走自定义行。 */
export const PRESET_MICROBES = [
  "Fusobacterium",
  "Porphyromonas",
  "Prevotella",
  "Streptococcus",
  "Lactobacillus",
] as const;

/**
 * 元数据里以 0/1 表示的二值字段。
 * DOM id 用短横线、载荷键用下划线，这里把两者显式配对，避免运行时做字符串替换。
 */
export interface BinaryMetadataField {
  /** 去掉 `metadata-` 前缀的 DOM id 片段。 */
  readonly id: string;
  /** CanonicalPayload.metadata 里的键名。 */
  readonly key: string;
}

export const BINARY_METADATA_FIELDS: readonly BinaryMetadataField[] = [
  { id: "recent-antibiotics", key: "recent_antibiotics" },
  { id: "recent-probiotics", key: "recent_probiotics" },
  { id: "renal-impairment", key: "renal_impairment" },
  { id: "hepatic-impairment", key: "hepatic_impairment" },
  { id: "pregnancy", key: "pregnancy" },
];
