# API Response Example

Typical JSON response shape:

```json
{
  "ok": true,
  "report": {
    "top_microbes": [["Fusobacterium", 0.33], ["Porphyromonas", 0.28]],
    "gnn_features": {
      "gnn_signal": 0.17,
      "centrality_signal": 0.66,
      "abundance_signal": 0.2,
      "module_signal": 0.41,
      "density": 1.0
    },
    "risk_result": {
      "risk_score": 1.42,
      "risk_level": "high"
    },
    "recommendations": [
      {
        "marker": "Fusobacterium",
        "abundance": 0.18,
        "priority": 1.1,
        "suggestion": "Consider anti-inflammatory strategy and closer colon surveillance.",
        "rationale": "High Fusobacterium abundance is treated as a high-risk inflammatory marker in the prototype rule base."
      }
    ]
  },
  "saved_to": "outputs/report_YYYYMMDD_HHMMSS.json"
}
```

This shape is intended for prototype demonstration and can later be extended into a more formal clinical report schema.
