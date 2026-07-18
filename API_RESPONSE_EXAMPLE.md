# API Response Example

Representative response from the current temporal-topology AFT consensus backend. Large inferred-topology and artifact lists are abbreviated here; the endpoint returns the complete fields.

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
      "backend": "temporal_topology_aft_cross_split_consensus",
      "model_release": "temporal_topology_aft_cross_split_consensus_v1",
      "num_split_branches": 2,
      "num_gnn_models": 6,
      "num_aft_models": 10,
      "consensus_alpha": 0.63,
      "topology_source": "inferred_from_web_inputs",
      "topology_inference_method": "split_train_only_standardized_ridge",
      "gnn_inference_context": "fixed_median_batch_normalization_anchor",
      "defaulted_inputs": [],
      "out_of_training_range_inputs": [],
      "unsupported_microbes_ignored": []
    },
    "risk_result": {
      "risk_score": 70.06,
      "risk_level": "high",
      "risk_percentile": 70.06,
      "raw_model_risk": 0.579259,
      "split_consensus_risks": {
        "42": 0.433723,
        "43": 0.724796
      },
      "split_disagreement": 0.291073,
      "prediction_reliability": "standard",
      "ensemble_size": 16,
      "backend": "temporal_topology_aft_cross_split_consensus",
      "model_release": "temporal_topology_aft_cross_split_consensus_v1"
    },
    "recommendations": [
      {
        "recommendation_id": "risk_review_high",
        "category": "risk_follow_up",
        "title": "模型提示较高风险，优先安排临床复核",
        "suggestion": "整理症状、既往检查、家族史和完整用药清单，带着本结果咨询消化专科或临床药师。",
        "action_steps": [
          "记录目前症状、开始时间以及近期是否加重。",
          "准备既往检查结果、家族史和完整用药清单。",
          "把这些资料和本结果交给消化专科或临床药师。"
        ],
        "rationale": "该结果表示研究队列中的相对位置，不是诊断或个人绝对发病概率。",
        "priority": 0.96,
        "urgency": "priority",
        "urgency_label": "优先处理",
        "evidence_level": "model_assisted_review",
        "evidence_source_ids": ["FDA_CDS_2026", "INTERNAL_TOPOLOGY_V6"],
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

`risk_score` is a percentile relative to the `topology_v6` reference cohort. It is not an absolute event probability. Function scores and edge weights returned under `gnn_features` are model-inferred values, not direct laboratory measurements.

The top-level `pharmacy_assessment` is the same object stored under `report.pharmacy_assessment`; the duplicate path keeps API and saved-report callers compatible. Root-level `recommendations` is also an alias of the assessment cards.

`interaction_screening_performed` refers only to the local minimum high-priority subset and becomes `true` when at least two submitted medications are screened. `comprehensive_interaction_screening_performed` remains `false`; a zero-match result does not exclude other interactions. Label dosage sections are product evidence only, while `patient_specific_dose_selected` and `treatment_duration_selected` remain `false`.

Invalid values return HTTP `400` with field-level errors. Examples include negative microbial abundance, age outside `1-120`, BMI outside `5-100`, non-binary clinical or medication-context flags, malformed medication/allergy lists, and non-finite numbers.
