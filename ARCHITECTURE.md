# Architecture Overview

## Active Inference Path

```text
JSON or form input
  -> field validation and canonical normalization
  -> endpoint eligibility routing
     -> complete oncology: AC-ICAM V8 clinical core or measured-ICR model
        -> 5-member ridge-Cox inference
        -> 36/60 month PFS and OOF-reference percentile
     -> incomplete oncology + complete five-microbe panel
        -> temporal-topology research inference
        -> topology_v6 research reference percentile
  -> pharmacy quality gate
  -> RxNorm normalization and product-specific label evidence
  -> limited high-priority DDI and exact-ingredient allergy screening
  -> evidence-linked review cards
  -> structured report
```

The active release is `ac_icam_real_outcome_pfs_v8`.

## Module Ownership

| Layer | Main modules |
|---|---|
| HTTP and forms | `enhanced_app.py` |
| Validation | `src/validators.py`, `src/clinical_standardizer.py` |
| Pipeline dispatch | `src/pipeline.py` |
| Current model bridge | `src/ac_icam_v8_bridge.py` |
| V8 evaluation | `experiments/ac_icam_real_outcome_v8/benchmark.py` |
| Deployment training | `experiments/ac_icam_real_outcome_v8/deployment.py` |
| Local model artifact | `config/releases/ac_icam_real_outcome_pfs_v8.json` (not tracked) |
| Previous temporal backend | `src/temporal_topology_bridge.py` |
| Pharmacy engine | `src/pharmacy_engine.py`, `src/drug_knowledge.py`, `data/pharmacy_rules_v3.json`, `data/pharmacy_knowledge/` |
| Report and compatibility | `src/report.py`, `src/clinical_report_builder.py`, `src/pharmacy_advice.py`, `src/recommendation.py` |

## Optional Internal Oral-Adenoma Path

The oral-only adenoma model is an independent research endpoint and is disabled
by default. Setting `GOA_ENABLE_INTERNAL_ORAL_ADENOMA=1` exposes separate
`/internal/oral-adenoma/schema` and `/internal/oral-adenoma/analyze` endpoints.
It requires all 381 oral-swab or saliva genus percentages and rejects stool,
blood, and tissue inputs. Its result is never combined with the AC-ICAM V8 PFS
score.

The web bridge reads audited numeric weights from the local-only
`config/releases/oral_adenoma_internal_v3.json`; it does not deserialize the
research joblib bundle. TypeScript sources live in `frontend/src/` and compile
to `static/generated/app.js`. The pre-migration browser files are preserved in
`archive/legacy_frontend_vanilla_js_20260814/`.

## Deployment Boundary

The web contract requires age and sex, with age limited to 18-75. AJCC stage,
pathological T/N/M, tumor location, and morphology are optional at submission
so general users can still access the non-PFS modules. The V8 PFS bridge runs
only when that oncology set is complete; otherwise it returns an explicit
not-calculated result and does not impute a normal stage. `icr_score` is
optional and activates the expanded model only when it is a measured
tumor-RNA value and the oncology set is complete. It is never inferred from
routine fields.

Microbiome and adjuvant-treatment inputs are excluded from the deployed PFS
score because the locked repeated evaluation did not show an improvement.
They remain available to independent descriptive and pharmacy modules. For
incomplete oncology input, the complete five-microbe panel can also generate
a separately named `general_risk_result`. This research percentile is never
stored in the V8 PFS `risk_result` and is explicitly marked as non-diagnostic,
non-screening, and not an absolute cancer probability.

The reported C-index and AUC values are internal repeated-cross-validation
estimates on AC-ICAM, not external clinical validation. Breslow 36/60 month
PFS values are model estimates, not guaranteed individual outcomes.

## Pharmacy Decision Boundary

`src/pharmacy_engine.py` consumes the submitted microbial panel, calibrated model inputs, model reliability flags, risk context, and optional medication metadata. It emits a versioned `pharmacy_assessment` with three states:

- `standard`: complete inputs and no model reliability alert;
- `limited`: incomplete panel or medication context, defaulted model inputs, split disagreement, unsupported microbes, or recent antibiotic exposure;
- `withheld`: out-of-training-range inputs, unavailable calibrated abundance values, or an unverified backend reliability state.

Marker cards require all five supported microbes. Missing markers are not interpreted as zero, and every card is restricted to clinician review with medication changes disabled. All seven medication-context fields must be explicitly reported for `standard` status. The drug-knowledge layer contains 46 product-specific label snapshots and a historical minimum high-priority DDI subset. It distinguishes limited screening from comprehensive review, never treats a negative result as proof of safety, and never converts label dosage text into a patient-specific dose or duration.

The active engine contract and evidence governance are documented in `PHARMACY_ASSISTANCE.md`.

## Compatibility

Set `GOA_MODEL_BACKEND=temporal_topology` for the previous temporal-topology
backend, or `GOA_MODEL_BACKEND=legacy_cox` for the archived five-seed Cox
bridge at `archive/legacy_web_backends/cox_ensemble_v1.py`. The default path
never silently falls back to either one.
