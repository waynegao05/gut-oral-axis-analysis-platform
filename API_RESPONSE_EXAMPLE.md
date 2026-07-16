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
    "recommendations": []
  },
  "saved_to": "outputs/report_YYYYMMDD_HHMMSS.json"
}
```

`risk_score` is a percentile relative to the `topology_v6` reference cohort. It is not an absolute event probability. Function scores and edge weights returned under `gnn_features` are model-inferred values, not direct laboratory measurements.

Invalid values return HTTP `400` with field-level errors. Examples include negative microbial abundance, age outside `1-120`, BMI outside `5-100`, non-binary smoking/family-history values, and non-finite numbers.
