# Oral-only internal adenoma model

This experiment is isolated from the web application and all survival-model
mainlines. It uses only oral-swab microbiome features to distinguish colorectal
adenoma from healthy controls. Stool, fecal, intestinal, blood, serum, plasma,
and tissue inputs are hard failures in both preparation and inference.

## Locked real cohort

- Source: Zhang et al., *Theranostics* 2020, DOI `10.7150/thno.49515`.
- Samples: 34 colorectal adenoma and 58 healthy participants.
- Input: bilateral buccal oral swabs, 381 genus relative abundances.
- Excluded: all 161 CRC samples and every non-oral modality.
- Synthetic samples: none.

The source reports an adenoma mean lesion size of `0.8 +/- 0.3 cm`, but does
not provide individual lesion sizes. The formal endpoint is therefore
**adenoma**, not verified diminutive adenoma (`<=5 mm`).

## Reproduce

```powershell
python -m experiments.oral_adenoma_internal_v3.prepare_data
python -m experiments.oral_adenoma_internal_v3.benchmark
python -m experiments.oral_adenoma_internal_v3.build_manifest
```

To reuse an already downloaded official supplement:

```powershell
python -m experiments.oral_adenoma_internal_v3.prepare_data `
  --archive outputs/oral_adenoma_source_audit/zhang_supp/thnov10p11595s2.zip
```

## Validation

The primary result is five-fold nested OOF repeated over seeds
`7, 21, 42, 123, 2026`. Candidate selection and the low-FPR threshold are fit
only inside each outer training partition. The final decision averages five
cross-fitted log-odds margins for each real participant.

`batch_prefix_leave_one_group_out.csv` is a mandatory robustness diagnostic.
It detects performance loss across source sample-ID prefixes and must not be
hidden when the primary OOF result is presented.

## Deployment boundary

The generated joblib bundle is for internal research only and is not imported
by the web application. Its metrics are retrospective, single-center internal
validation, not prospective, external, analytical-kit, or clinical validation.
