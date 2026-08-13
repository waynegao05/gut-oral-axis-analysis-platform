# API Response Example

Representative response from the current AC-ICAM V8 real-outcome PFS backend.
Large model metadata lists are abbreviated here; the endpoint returns the
complete fields.

```json
{
  "ok": true,
  "report": {
    "top_microbes": [
      ["Streptococcus", 0.22],
      ["Fusobacterium", 0.18],
      ["Porphyromonas", 0.14]
    ],
    "gnn_features": {
      "backend": "ac_icam_real_outcome_clinical_pfs",
      "model_release": "ac_icam_real_outcome_pfs_v8",
      "model_variant": "clinical_core",
      "endpoint": "PFS",
      "microbiome_used_for_risk": false,
      "treatment_used_for_risk": false,
      "icr_used_for_risk": false,
      "defaulted_inputs": [],
      "out_of_training_range_inputs": [],
      "formal_metrics": {
        "ensemble_oof_c_index": 0.7756446991404011
      }
    },
    "risk_result": {
      "risk_score": 63.41,
      "risk_level": "medium",
      "risk_percentile": 63.41,
      "raw_model_risk": 0.260397,
      "pfs_probability": {
        "36": 0.723103,
        "60": 0.692087
      },
      "prediction_reliability": "standard",
      "prediction_available": true,
      "ensemble_size": 5,
      "backend": "ac_icam_real_outcome_clinical_pfs",
      "model_release": "ac_icam_real_outcome_pfs_v8",
      "model_variant": "clinical_core",
      "endpoint": "PFS",
      "research_use_only": true
    },
    "recommendations": [
      {
        "recommendation_id": "risk_review_high",
        "category": "risk_follow_up",
        "title": "PFS 模型提示较高相对风险，优先核对随访计划",
        "suggestion": "整理病理、治疗、近期影像和复诊安排，交给肿瘤科核对随访计划。",
        "action_steps": [
          "记录目前症状、开始时间以及近期是否加重。",
          "准备既往检查结果、家族史和完整用药清单。",
          "把这些资料和本结果交给消化专科或临床药师。"
        ],
        "rationale": "该结果表示 AC-ICAM 队列中的相对进展风险位置，不是个体预后保证或治疗指令。",
        "priority": 0.96,
        "urgency": "priority",
        "urgency_label": "优先处理",
        "evidence_level": "model_assisted_review",
        "evidence_source_ids": ["FDA_CDS_2026", "AC_ICAM_V8_PFS"],
        "requires_clinician_review": true,
        "allows_medication_change": false
      }
    ],
    "pharmacy_assessment": {
      "engine_version": "pharmacy_assistance_v3",
      "knowledge_schema_version": "3.0",
      "knowledge_last_reviewed": "2026-07-18",
      "knowledge_sha256": "...",
      "status": "standard",
      "status_label": "信息较完整，可供医生或药师参考",
      "quality": {
        "status_reasons": [],
        "missing_markers": [],
        "panel_completeness": 1.0,
        "calibration_ready": true,
        "model_reliability": "standard"
      },
      "plain_language_summary": {
        "headline": "有 1 项需要优先核对",
        "urgent_count": 1,
        "routine_count": 0,
        "what_to_do_now": [
          "记录目前症状、开始时间以及近期是否加重。"
        ],
        "what_was_checked": [
          "尚未填写当前用药，未进行药名核对"
        ],
        "what_was_not_checked": [
          "未收录在当前知识库中的其他药物相互作用",
          "患者个人应使用的具体药物、剂量、疗程或停换药方案"
        ],
        "safety_note": "这是给医生或药师复核的辅助清单，不是诊断或处方。"
      },
      "summary": {
        "recommendation_count": 1,
        "marker_trigger_count": 0,
        "priority_card_count": 1,
        "medication_history_complete": true,
        "medication_context_complete": true,
        "interaction_screening_performed": false,
        "interaction_screening_scope": "onc_2012_minimum_high_priority_subset",
        "comprehensive_interaction_screening_performed": false,
        "high_priority_interaction_match_count": 0,
        "label_lookup_performed": false,
        "label_record_count": 0,
        "medication_candidate_generated": false,
        "patient_specific_dose_selected": false,
        "treatment_duration_selected": false,
        "probiotic_candidate_count": 0,
        "medication_change_allowed": false
      },
      "recommendations": [
        {
          "recommendation_id": "risk_review_high",
          "requires_clinician_review": true,
          "allows_medication_change": false
        }
      ],
      "drug_knowledge": {
        "available": true,
        "status": "limited_clinical_decision_support",
        "database": {
          "dataset_id": "goa_openfda_label_evidence_v1",
          "record_count": 46,
          "comprehensive_drug_coverage": false
        },
        "normalization": {
          "input_count": 0,
          "matched_count": 0,
          "medications": []
        },
        "label_lookup": {
          "performed": false,
          "record_count": 0,
          "records": []
        },
        "interaction_screening": {
          "interaction_screening_performed": false,
          "screening_scope": "onc_2012_minimum_high_priority_subset",
          "comprehensive_interaction_screening_performed": false,
          "negative_result_excludes_other_interactions": false
        },
        "candidate_therapy_support": {
          "medication_candidates_generated": false,
          "patient_specific_dose_selected": false,
          "treatment_duration_selected": false
        }
      },
      "evidence_sources": [
        {
          "source_id": "FDA_CDS_2026",
          "organization": "U.S. Food and Drug Administration",
          "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software"
        }
      ]
    }
  },
  "pharmacy_assessment": {
    "engine_version": "pharmacy_assistance_v3",
    "status": "standard"
  },
  "saved_to": "outputs/report_YYYYMMDD_HHMMSS.json"
}
```

`risk_score` is a percentile relative to the AC-ICAM five-seed OOF PFS-risk
distribution. It is not a general-population cancer probability.
`pfs_probability` is a Breslow model estimate for an already diagnosed
colorectal-cancer patient and is not an individual prognostic guarantee.
`clinical_icr` appears only when a measured tumor-RNA ICR score was submitted.

For a person without complete colorectal-cancer pathology, age and sex are
still required but the oncology fields may be omitted. The endpoint returns
HTTP `200` with `prediction_available=false`,
`not_available_reason="missing_oncology_fields"`, null 36/60-month PFS values,
and a list of the missing oncology fields. The service does not manufacture
normal staging values.

If all five core microbes are present, the response also includes a separate
`general_risk_result`, for example:

```json
{
  "prediction_available": true,
  "endpoint": "research_risk_index",
  "display_name": "菌群-临床研究风险指数",
  "risk_percentile": 71.94,
  "risk_level": "high",
  "absolute_cancer_probability": false,
  "screening_result": false,
  "pfs_calculated": false,
  "dataset_version": "topology_v6",
  "dataset_is_synthetic_noisy_augmented": true
}
```

This is a research-reference percentile used for visualization and review. It
is not an absolute colorectal-cancer probability, screening result, diagnosis,
or PFS prediction. If the five-microbe panel is incomplete, its value remains
null and `not_available_reason` is `incomplete_microbiome_panel`.

The top-level `pharmacy_assessment` is the same object stored under `report.pharmacy_assessment`; the duplicate path keeps API and saved-report callers compatible. Root-level `recommendations` is also an alias of the assessment cards.

`interaction_screening_performed` refers only to the local minimum high-priority subset and becomes `true` when at least two submitted medications are screened. `comprehensive_interaction_screening_performed` remains `false`; a zero-match result does not exclude other interactions. Label dosage sections are product evidence only, while `patient_specific_dose_selected` and `treatment_duration_selected` remain `false`.

Invalid values return HTTP `400` with field-level errors. Examples include missing age or sex, negative microbial abundance, age outside `18-75`, BMI outside `5-100`, non-binary clinical or medication-context flags, malformed medication/allergy lists, and non-finite numbers.
