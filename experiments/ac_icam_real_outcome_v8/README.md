# AC-ICAM real-outcome V8

This directory is an independent benchmark on measured tumor and
adjacent-normal colon microbiome with real PFS/OS follow-up. It does not
replace V7, the current web backend, or any existing training output.

## What V8 contains

- 246 patients with paired tumor and adjacent-normal 16S profiles.
- 517 genus-level abundance features extracted from the published RData.
- Real pathological stage, T/N/M, age, sex, tumor location, morphology,
  adjuvant-treatment history, PFS/OS time, and censoring status.
- The complete published 41-genus MBR panel and the 14-genus MRS-T /
  12-genus MRS-N panels in `published_panels.json`.

The published MBR and 16S-MRS panels are reference-only. Their taxa were
selected using outcomes from the same AC-ICAM patients used here, so
showing them as candidate cross-validation features would overstate
generalization.

## Locked evaluation

- Primary: stage I-IV PFS, 246 patients and 71 events.
- Secondary: stage I-IV OS, 246 patients and 101 events.
- Sensitivity: stage I-III PFS/OS.
- Five repeated outer 5-fold splits: 7, 21, 42, 123, and 2026.
- Three-fold inner selection inside every outer training fold.
- All prevalence filtering, CLR transforms, paired tumor-normal
  relationships, supervised feature selection, regularization, and blend
  weights are fitted using outer-train patients only.
- Metrics: pooled OOF Harrell C-index, patient-bootstrap 95% CI, and
  cumulative/dynamic IPCW AUC at 36 and 60 months.

## Formal results

The values below are five-seed OOF ensemble estimates.

| Scope / endpoint | Model | C-index | 95% CI | AUC 36 | AUC 60 |
|---|---|---:|---:|---:|---:|
| Stage I-IV PFS | Clinical core | 0.7756 | 0.7214-0.8251 | 0.8185 | 0.8013 |
| Stage I-IV PFS | Clinical + measured ICR | **0.7845** | 0.7328-0.8323 | **0.8294** | **0.8133** |
| Stage I-IV PFS | Clinical + microbiome safe blend | 0.7740 | 0.7197-0.8237 | 0.8177 | 0.7984 |
| Stage I-IV OS | Clinical + ICR + microbiome safe blend | 0.6926 | 0.6377-0.7465 | 0.7631 | 0.7014 |
| Stage I-III PFS | Clinical + measured ICR | 0.6777 | 0.5926-0.7535 | 0.6952 | 0.6701 |
| Stage I-III OS | Clinical + ICR + microbiome safe blend | 0.6145 | 0.5435-0.6836 | 0.6509 | 0.6002 |

Interpretation:

- The `0.761` target is exceeded for broad stage I-IV PFS.
- The gain is primarily due to real pathological stage, especially the
  separation of stage IV, not to the microbiome module.
- The internal microbiome relation model does not improve PFS after
  five-seed evaluation. It gives a small OS increment, but OS remains
  below target.
- ICR is measured from tumor RNA and is not available from routine web
  inputs. The deployable clinical-only result is therefore 0.7756, not
  0.7845.
- The same-cohort published MBR reaches 0.7993 on stage I-III OS, but only
  0.5478 in the independent ICAM42 subset. It is a potential upper
  reference, not evidence that V8 generalizes at that level.

V8 is not directly comparable with the synthetic/noisy V7 benchmark.
The tissues and outcome definitions differ.

## Run

Build or validate the processed cohort:

```powershell
python -m experiments.ac_icam_real_outcome_v8.data
```

Run the primary five-seed benchmark:

```powershell
python -m experiments.ac_icam_real_outcome_v8.benchmark `
  --endpoint PFS `
  --scope all_stage `
  --seeds 7 21 42 123 2026 `
  --bootstrap-iterations 1000
```

Run the other locked analyses by changing `--endpoint` to `OS` and/or
`--scope` to `stage_i_iii`.

Outputs are written to:

```text
outputs/ac_icam_real_outcome_v8/<scope>_<endpoint>/
```

No GPU is required.

## Sources

- AC-ICAM article: <https://doi.org/10.1038/s41591-023-02324-5>
- Public microbiome files: <https://doi.org/10.6084/m9.figshare.16944775>
- Clinical data: <https://www.cbioportal.org/study/summary?id=coad_silu_2022>
- Published AC-ICAM code: <https://doi.org/10.5281/zenodo.7766220>
- MRS article: <https://doi.org/10.1038/s41467-025-67047-2>
- MRS code archive: <https://doi.org/10.5281/zenodo.17588313>
