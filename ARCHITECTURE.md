# Architecture Overview

## Active Inference Path

```text
JSON or form input
  -> field validation and canonical normalization
  -> split-specific train-only topology inference
  -> 3-member structure-aware GNN reference per split
  -> 5-member temporal-topology AFT expert per split
  -> validation-locked risk fusion (alpha = 0.63)
  -> average split 42 and split 43
  -> cohort-relative percentile, reliability flags, and report
```

The active release is `temporal_topology_aft_cross_split_consensus_v1`.

## Module Ownership

| Layer | Main modules |
|---|---|
| HTTP and forms | `enhanced_app.py` |
| Validation | `src/validators.py`, `src/clinical_standardizer.py` |
| Pipeline dispatch | `src/pipeline.py` |
| Current model bridge | `src/temporal_topology_bridge.py` |
| GNN reference | `research/model_v2.py`, `research/full_risk_head_refiner_v2.py` |
| AFT expert | `experiments/temporal_independent_v3/topology_aft_fusion.py` |
| Consensus selection | `experiments/temporal_independent_v3/cross_split_consensus.py` |
| Report and advice | `src/report.py`, `src/recommendation.py`, `src/pharmacy_advice.py` |

## Deployment Boundary

Research evaluation reads topology from aligned research tables and replays deterministic eight-sample contexts. Web inference cannot observe those topology fields, so it infers five function scores and ten named edge weights from web-available inputs. A fixed calibration anchor removes request-batch dependence from GNN normalization.

The resulting web topology is explicitly labeled as inferred, not measured. Offline held-out C-index must therefore not be presented as a direct clinical validation score for the web endpoint.

## Compatibility

Set `GOA_MODEL_BACKEND=legacy_cox` to load the archived five-seed Cox bridge at `archive/legacy_web_backends/cox_ensemble_v1.py`. The default path never silently falls back to it.
