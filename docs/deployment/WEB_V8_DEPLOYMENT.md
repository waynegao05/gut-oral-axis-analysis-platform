# AC-ICAM V8 Web Deployment

## Active release

- backend selector: `ac_icam_v8`
- response backend: `ac_icam_real_outcome_clinical_pfs`
- release: `ac_icam_real_outcome_pfs_v8`
- endpoint: PFS
- cohort: 246 AC-ICAM patients, 71 PFS events
- default model: clinical core, five deployment members
- optional model: clinical + measured tumor-RNA ICR

The clinical-core repeated OOF estimate is C-index `0.7756`, AUC36
`0.8185`, and AUC60 `0.8013`. The measured-ICR variant is C-index
`0.7845`, AUC36 `0.8294`, and AUC60 `0.8133`.

These are internal repeated-cross-validation results, not external clinical
validation.

## Build the artifact

Run the formal all-stage PFS benchmark first, then:

```powershell
python -m experiments.ac_icam_real_outcome_v8.deployment
```

The command writes:

```text
config/releases/ac_icam_real_outcome_pfs_v8.json
```

This fitted artifact is intentionally ignored by Git and is not included in
GitHub source archives or Release assets. Keep or rebuild it only in an
approved local environment.

The artifact contains five fitted ridge-Cox members, feature encoders,
member-level Breslow baseline cumulative hazards, OOF risk calibration,
training ranges, and the locked formal metrics.

The web bridge verifies the release SHA-256 before loading. A modified artifact
fails explicitly instead of silently changing predictions.

## Start the web app

```powershell
$env:GOA_MODEL_BACKEND = "ac_icam_v8"
$env:GOA_PORT = "8765"
python enhanced_app.py
```

`ac_icam_v8` is also the default when `GOA_MODEL_BACKEND` is unset.

## Input contract

Required:

- `clinical.age`
- `clinical.sex`

Age must be in the web-supported range `18-75`.

Optional on the form, but all required together to calculate V8 PFS:

- `clinical.stage`
- `clinical.path_t`
- `clinical.path_n`
- `clinical.path_m`
- `clinical.tumor_location`
- `clinical.tumor_morphology`

Other optional inputs:

- `clinical.icr_score`: activates `clinical_icr`; it must be a measured
  tumor-RNA ICR value.
- `microbes`, `metabolites`, treatment and medication context: do not alter
  the V8 PFS score.
- When oncology fields are incomplete, all five core microbes are required
  together to calculate the separate research reference percentile:
  `Fusobacterium`, `Porphyromonas`, `Prevotella`, `Streptococcus`, and
  `Lactobacillus`. Missing microbes are not interpreted as zero.

The endpoint rejects missing age or sex and unsupported values. When oncology
fields are incomplete, the request still succeeds but returns
`prediction_available=false`,
`not_available_reason=missing_oncology_fields`, and null PFS estimates.
No missing pathology value is converted into a normal stage or a web default.
Inputs outside the AC-ICAM training range remain in the audit JSON but are not
displayed as an available risk estimate.

When the five-microbe panel is complete, the same successful response also
contains `general_risk_result`. It uses the retained temporal-topology
research backend and the synthetic/noisy augmented `topology_v6` reference
distribution. It is a `0-100` cohort percentile, not an absolute cancer
probability, screening result, diagnosis, or PFS estimate. Inputs outside that
model's training range are withheld.

## Output contract

Important `risk_result` fields:

- `risk_percentile`: relative position in the five-seed AC-ICAM OOF risk
  distribution.
- `pfs_probability.36` and `.60`: Breslow model estimates.
- `model_variant`: `clinical_core` or `clinical_icr`.
- `prediction_reliability` and `prediction_available`.
- `research_use_only = true`.

Important `general_risk_result` fields for incomplete oncology input:

- `endpoint = research_risk_index`.
- `risk_percentile`: position in the `topology_v6` research reference
  distribution.
- `absolute_cancer_probability = false`.
- `screening_result = false`.
- `dataset_is_synthetic_noisy_augmented = true`.
- `prediction_available`: requires the complete five-microbe panel and
  in-range model inputs.

Microbiome and pharmacy quality status no longer suppress a valid clinical
PFS result. They only control microbiome- or medication-related guidance.

## Rollback

Previous temporal-topology backend:

```powershell
$env:GOA_MODEL_BACKEND = "temporal_topology"
python enhanced_app.py
```

Archived Cox backend:

```powershell
$env:GOA_MODEL_BACKEND = "legacy_cox"
python enhanced_app.py
```

No previous implementation or model artifact is deleted by the V8 release.
