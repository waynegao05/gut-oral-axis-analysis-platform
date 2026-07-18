# Testing Guide

## Full Suite

The default tests require the current local research artifacts under `outputs/current_mainline_v2/`.

```powershell
python -m pytest -q
```

## Focused Checks

```powershell
python -m pytest -q tests/test_pipeline.py tests/test_app_validation.py
python -m pytest -q tests/test_temporal_topology_bridge.py
python -m pytest -q tests/test_research_data_validation.py
python -m pytest -q tests/test_build_drug_knowledge_v1.py tests/test_drug_knowledge.py tests/test_pharmacy_engine.py tests/test_clinical_standardizer.py
python -m research.build_drug_knowledge_v1
python -m research.rebuild_pharmacy_calibration_v2
```

The temporal bridge tests verify:

- exact replay against saved split-specific consensus risks;
- current release and backend identifiers;
- train-only topology inference;
- fixed calibration-anchor context;
- complete inferred function-score and edge-weight outputs.

The pharmacy tests verify:

- missing microbes are not converted into false low-abundance triggers;
- model reliability and out-of-range inputs activate limited/withheld states;
- medication, allergy, antibiotic, probiotic, and special-population context is preserved;
- generated label records and the full record list pass SHA-256 integrity checks;
- dose-bearing, brand, generic, and Chinese medication inputs normalize to reviewed RXCUIs;
- limited high-priority DDI and exact-ingredient allergy matches are surfaced without authorizing medication changes;
- label dosage text never sets a patient-specific dose or duration;
- probiotic strain candidates require an exact registered guideline context;
- every rule references a registered evidence source and disables medication changes;
- marker thresholds reproduce from the tracked `topology_v6` graph table;
- web, pipeline, and clinical-report callers reuse the same assessment.

## Static Checks

```powershell
python -m compileall -q archive config experiments research src tests
node --check static/app.js
git diff --check
```

Before publishing, also confirm that no `outputs/`, local environment, editor state, or model checkpoint is staged.
