# Internal oral adenoma model

## Purpose

`oral_adenoma_internal_v3` is an independent research endpoint for adenoma
versus healthy-control screening from a complete oral-swab or saliva genus
profile. It does not alter or combine with the AC-ICAM V8 PFS score.

## Evidence boundary

- Real participants: 92 (34 adenoma, 58 healthy).
- Input: 381 oral-swab genus percentages; no stool, blood, or tissue input.
- Nested repeated OOF sensitivity: 22/34 (64.71%).
- False-positive rate: 3/58 (5.17%).
- ROC AUC: 0.9219.
- The cohort mean lesion size was 0.8 +/- 0.3 cm, but individual lesion sizes
  were unavailable. This is not a verified <=5 mm diminutive-adenoma result.
- Retrospective, single-center, internal research only; not clinical validation.

## Run locally

The ordinary web command keeps this endpoint disabled. To enable it locally:

```powershell
.\scripts\start_internal_oral_adenoma.ps1
```

The script sets `GOA_ENABLE_INTERNAL_ORAL_ADENOMA=1`, builds the TypeScript
frontend, and starts Flask. The page provides a JSON template containing all
381 required genus names.

## Frontend build

```powershell
npm install
npm run typecheck
npm run build
```

The browser loads `static/generated/app.js`. The pre-migration frontend is
preserved under `archive/legacy_frontend_vanilla_js_20260814/`.

## Release format

The web bridge reads audited numeric weights from the local-only, Git-ignored
`config/releases/oral_adenoma_internal_v3.json`. It does not deserialize the
research `joblib` bundle in the application process. The JSON release is
generated with:

```powershell
python -m experiments.oral_adenoma_internal_v3.export_release
```

Neither the JSON weights nor the archived joblib bundle is uploaded to GitHub.
