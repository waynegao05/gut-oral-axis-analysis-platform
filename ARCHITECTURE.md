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
  -> cohort-relative percentile and reliability flags
  -> pharmacy quality gate
  -> RxNorm normalization and product-specific label evidence
  -> limited high-priority DDI and exact-ingredient allergy screening
  -> evidence-linked review cards
  -> structured report
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
| Pharmacy engine | `src/pharmacy_engine.py`, `src/drug_knowledge.py`, `data/pharmacy_rules_v3.json`, `data/pharmacy_knowledge/` |
| Report and compatibility | `src/report.py`, `src/clinical_report_builder.py`, `src/pharmacy_advice.py`, `src/recommendation.py` |

## Deployment Boundary

Research evaluation reads topology from aligned research tables and replays deterministic eight-sample contexts. Web inference cannot observe those topology fields, so it infers five function scores and ten named edge weights from web-available inputs. A fixed calibration anchor removes request-batch dependence from GNN normalization.

The resulting web topology is explicitly labeled as inferred, not measured. Offline held-out C-index must therefore not be presented as a direct clinical validation score for the web endpoint.

## Pharmacy Decision Boundary

`src/pharmacy_engine.py` consumes the submitted microbial panel, calibrated model inputs, model reliability flags, risk context, and optional medication metadata. It emits a versioned `pharmacy_assessment` with three states:

- `standard`: complete inputs and no model reliability alert;
- `limited`: incomplete panel or medication context, defaulted model inputs, split disagreement, unsupported microbes, or recent antibiotic exposure;
- `withheld`: out-of-training-range inputs, unavailable calibrated abundance values, or an unverified backend reliability state.

Marker cards require all five supported microbes. Missing markers are not interpreted as zero, and every card is restricted to clinician review with medication changes disabled. All seven medication-context fields must be explicitly reported for `standard` status. The drug-knowledge layer contains 46 product-specific label snapshots and a historical minimum high-priority DDI subset. It distinguishes limited screening from comprehensive review, never treats a negative result as proof of safety, and never converts label dosage text into a patient-specific dose or duration.

The active engine contract and evidence governance are documented in `PHARMACY_ASSISTANCE.md`.

## Compatibility

Set `GOA_MODEL_BACKEND=legacy_cox` to load the archived five-seed Cox bridge at `archive/legacy_web_backends/cox_ensemble_v1.py`. The default path never silently falls back to it.
