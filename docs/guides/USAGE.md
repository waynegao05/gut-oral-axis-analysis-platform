# Usage

## Web App

```powershell
python -m pip install -r requirements.txt
$env:GOA_MODEL_BACKEND = "ac_icam_v8"
$env:GOA_PORT = "8765"
python enhanced_app.py
```

Open `http://127.0.0.1:8765`.

The default backend loads the local-only
`config/releases/ac_icam_real_outcome_pfs_v8.json`. Model weights are excluded
from GitHub; rebuild the artifact in an approved local environment after a
formal V8 rerun with:

```powershell
python -m experiments.ac_icam_real_outcome_v8.deployment
```

The web form requires only age and sex; age must be between 18 and 75.
AJCC stage, pathological T/N/M, tumor location, and morphology are optional
form fields so people without a cancer diagnosis can still use the
microbiome and pharmacy modules. V8 PFS is calculated only when all of those
oncology fields are present. Missing oncology fields are never interpreted as
normal stage values. Measured tumor-RNA ICR is optional and activates the
expanded model only when the PFS input is otherwise complete. Microbiome
values do not alter PFS scoring.

For a person without complete oncology fields, filling all five core microbes
generates a separate `0-100` research reference percentile and visual scale.
The five microbes are `Fusobacterium`, `Porphyromonas`, `Prevotella`,
`Streptococcus`, and `Lactobacillus`; omitted values are not treated as zero.
This number is based on the synthetic/noisy augmented `topology_v6` research
reference and is not an absolute colorectal-cancer probability, screening
result, diagnosis, or PFS estimate.

The form also accepts optional pharmacy context: current medications, drug allergies, recent antibiotics/probiotics, renal or hepatic impairment, pregnancy, and a clinician-confirmed review context. Leave unknown values blank; enter `无` when an empty medication or allergy list has been confirmed. Medication strings may include brand, strength, and frequency. The system attempts local RxNorm normalization and label lookup but does not trigger automatic prescribing.

## Command-Line Example

```powershell
python cli_analysis.py
```

## Raw Clinical Workflow

```powershell
python clinical_workflow.py `
  --input <raw-input.json> `
  --standardized_output outputs/standardized_input.json `
  --output outputs/clinical_report.json
```

Raw input can provide medication context as follows:

```json
{
  "demographics": {
    "age": 62,
    "sex": "female"
  },
  "oncology": {
    "stage": 3,
    "path_t": 3,
    "path_n": 1,
    "path_m": 0,
    "tumor_location": "Colon Sigmoideum",
    "tumor_morphology": "Adenocarcinoma"
  },
  "history": {
    "recent_antibiotics": "no",
    "recent_probiotics": "yes"
  },
  "medication_context": {
    "current_medications": ["metformin 500 mg twice daily"],
    "drug_allergies": ["penicillin: rash"],
    "renal_impairment": "no",
    "hepatic_impairment": "no",
    "pregnancy": "no"
  },
  "clinical_context": {
    "suspected_condition": "colorectal_cancer_followup"
  }
}
```

The final JSON contains a versioned `pharmacy_assessment` with `drug_knowledge.normalization`, `label_lookup`, limited `interaction_screening`, exact-ingredient `allergy_screening`, and guarded `probiotic_decision_support`. A label dosage section is evidence only; it is not a selected patient dose or duration. See [PHARMACY_ASSISTANCE.md](../clinical/PHARMACY_ASSISTANCE.md) for field semantics and current limitations.

## Refresh Medication Knowledge

```powershell
python -m research.build_drug_knowledge_v1
```

The rebuild uses the reviewed 46-drug seed, RxNorm, and exact openFDA generic-name and route matching. It refuses to replace the database if any requested record fails unless `--allow-partial` is explicitly used for diagnosis.

## Previous Backend Comparison

```powershell
$env:GOA_MODEL_BACKEND = "temporal_topology"
python enhanced_app.py
```

The previous temporal backend requires its local artifacts under
`outputs/current_mainline_v2/`.

## Legacy Cox Comparison

```powershell
$env:GOA_MODEL_BACKEND = "legacy_cox"
python enhanced_app.py
```

This is a rollback/comparison mode, not the current release.

See the [project README](../../README.md) for payload constraints, evidence, and interpretation boundaries. See [API_RESPONSE_EXAMPLE.md](../api/API_RESPONSE_EXAMPLE.md) for the current response schema.
