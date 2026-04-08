# Clinical Workflow Usage

This workflow is intended for raw clinical-style input that has not yet been converted into model-ready tensors or feature tables.

## Input

Provide a JSON file following the structure in `example_raw_clinical_input.json`.

## Run

```bash
python clinical_workflow.py \
  --input example_raw_clinical_input.json \
  --output outputs/clinical_report.json \
  --standardized_output outputs/standardized_input.json
```

## What it does

1. Reads raw clinical-style input
2. Standardizes the payload into model-ready structure
3. Runs the existing analysis pipeline
4. Builds pharmacological assistance suggestions
5. Exports a final structured report

## Outputs

- `outputs/standardized_input.json`: standardized model input
- `outputs/clinical_report.json`: final research-oriented report

## Important note

The generated pharmacological assistance is research-oriented decision support and should be interpreted alongside physician judgment, laboratory testing, imaging, and confirmatory gastrointestinal evaluation.
