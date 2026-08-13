# Testing Guide

Pytest is configured to collect only `tests/`. This prevents generated output
folders containing vendored package tests from being mistaken for project
tests.

## TypeScript Frontend

```powershell
npm install
npm run typecheck
npm run build
```

## Internal Oral-Adenoma Endpoint

```powershell
python -m pytest -q tests/test_oral_adenoma_bridge.py
```

These checks cover JSON/sklearn numerical parity, complete 381-genus input,
oral-only enforcement, the default-off switch, and explicit internal mode.

## Full Suite

The default V8 tests require the local, Git-ignored deployment artifact at
`config/releases/ac_icam_real_outcome_pfs_v8.json`. Temporal-backend replay
tests additionally require the existing local artifacts under
`outputs/current_mainline_v2/`.

```powershell
python -m pytest -q
```

## Focused Checks

```powershell
python -m pytest -q tests/test_pipeline.py tests/test_app_validation.py
python -m pytest -q tests/test_ac_icam_v8_bridge.py tests/test_ac_icam_real_outcome_v8.py
python -m pytest -q tests/test_temporal_topology_bridge.py
python -m pytest -q tests/test_research_data_validation.py
python -m pytest -q tests/test_public_data_v1.py tests/test_topology_v7.py
python -m pytest -q tests/test_build_drug_knowledge_v1.py tests/test_drug_knowledge.py tests/test_pharmacy_engine.py tests/test_clinical_standardizer.py
python -m research.build_drug_knowledge_v1
python -m research.rebuild_pharmacy_calibration_v2
```

The V8 web tests verify:

- the default core model and optional measured-ICR model select correctly;
- formal C-index metadata is reproduced from the locked five-seed benchmark;
- 36/60 month PFS estimates are finite and temporally ordered;
- missing age or sex is rejected, while incomplete oncology fields return an
  explicit non-PFS result without imputation;
- incomplete oncology input with a complete five-microbe panel returns a
  separate research percentile, while an incomplete panel returns no number;
- the general research percentile is explicitly marked as non-absolute,
  non-screening, synthetic/noisy augmented research output;
- age 18 and 75 are accepted, while 17 and 76 are rejected;
- out-of-training-range inputs suppress the displayed risk;
- microbiome input does not alter the formal V8 PFS score;
- previous temporal and archived Cox paths remain selectable.

The temporal bridge regression tests verify:

- exact replay against saved split-specific consensus risks;
- current release and backend identifiers;
- train-only topology inference;
- fixed calibration-anchor context;
- complete inferred function-score and edge-weight outputs.

The topology v7 tests verify:

- 3,600 unique generated samples and complete graph/tabular/label tables;
- explicit generated-data provenance and a preserved v6 archive;
- controlled survival signal, independent censoring, and preserved generator_v1 rollback data;
- generation-group-disjoint train, validation, and test splits;
- valid finite ranges for all model inputs and generated survival labels.

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
