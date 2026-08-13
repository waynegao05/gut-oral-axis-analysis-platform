# Five-genus colorectal neoplasia screening benchmark

This experiment estimates two screening metrics that the PFS mainline cannot
provide: false-positive rate in healthy controls and sensitivity for small
adenomas. It is isolated from all production and survival-model code.

## Cohort and endpoint

- Public cohort: ZellerG_2014 / ERP005534.
- Negative group: 61 healthy controls.
- Positive group: 27 small adenomas, 15 large adenomas, and 53 colorectal cancers.
- Small adenoma means less than 10 mm according to the source cohort definition.
- Features are restricted to the five project genera: Fusobacterium,
  Porphyromonas, Prevotella, Streptococcus, and Lactobacillus.

The exact analysis choices are fixed in `protocol_lock.json`. Thresholds are
selected using only inner out-of-fold predictions. No outer-test label is used
to select a model, hyperparameter, or threshold.

## Run

```powershell
python -m experiments.adenoma_screening_v1.benchmark
```

The cohort-preparation R source is retained locally and intentionally excluded
from GitHub. The benchmark command therefore requires its approved, locally
prepared input table.

Formal outputs are written to `outputs/adenoma_screening_v1`.

## Interpretation boundary

This is a retrospective internal cross-validation result from one public stool
metagenomic cohort. It is not prospective clinical validation, an external
validation cohort, or analytical validation of the colorimetric assay. The
result must not be presented as the current PFS model's screening performance.
