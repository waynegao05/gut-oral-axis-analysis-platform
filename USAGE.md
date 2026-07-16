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

## Legacy Comparison

```powershell
$env:GOA_MODEL_BACKEND = "legacy_cox"
python enhanced_app.py
```

This is a rollback/comparison mode, not the current release.

See `README.md` for payload constraints, evidence, and interpretation boundaries. See `API_RESPONSE_EXAMPLE.md` for the current response schema.
