# Usage

## Web App

```powershell
python -m pip install -r requirements.txt
$env:GOA_MODEL_BACKEND = "temporal_topology"
$env:GOA_TEMPORAL_DEVICE = "cpu"
$env:GOA_PORT = "8765"
python enhanced_app.py
```

Open `http://127.0.0.1:8765`.

The default backend requires untracked local model artifacts under `outputs/current_mainline_v2/`. Missing artifacts cause an explicit error.

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
    "suspected_condition": "gut_risk_screening"
  }
}
```

The final JSON contains a versioned `pharmacy_assessment` with `drug_knowledge.normalization`, `label_lookup`, limited `interaction_screening`, exact-ingredient `allergy_screening`, and guarded `probiotic_decision_support`. A label dosage section is evidence only; it is not a selected patient dose or duration. See `PHARMACY_ASSISTANCE.md` for field semantics and current limitations.

## Refresh Medication Knowledge

```powershell
python -m research.build_drug_knowledge_v1
```

The rebuild uses the reviewed 46-drug seed, RxNorm, and exact openFDA generic-name and route matching. It refuses to replace the database if any requested record fails unless `--allow-partial` is explicitly used for diagnosis.

## Legacy Comparison

```powershell
$env:GOA_MODEL_BACKEND = "legacy_cox"
python enhanced_app.py
```

This is a rollback/comparison mode, not the current release.

See `README.md` for payload constraints, evidence, and interpretation boundaries. See `API_RESPONSE_EXAMPLE.md` for the current response schema.
