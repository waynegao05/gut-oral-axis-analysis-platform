# V8 decision

## Adopt

- Use `clinical_core` as the current real-outcome PFS benchmark:
  C-index 0.7756, AUC36 0.8185, AUC60 0.8013.
- Keep `clinical_icr` as an expanded measured-data benchmark:
  C-index 0.7845, AUC36 0.8294, AUC60 0.8133.
- Keep paired tumor-normal microbiome features and their stability report
  for biological analysis.

## Do not adopt as the performance mainline

- Do not select `safe_blend_core` for PFS. Its C-index is 0.7740, below
  the clinical-only 0.7756.
- Do not include adjuvant treatment in the primary baseline. It is
  post-surgical and its PFS C-index is lower than the core model.
- Do not use the published MBR or 16S-MRS panels as formal V8 candidate
  features. Both used AC-ICAM outcomes during their original selection.

## Remaining limitation

The broad-stage PFS result exceeds 0.761 because stage IV adds a strong,
real prognostic separation. Within stage I-III, PFS remains 0.6777 even
with ICR. Improving that clinically harder setting requires independent
recurrence information or an external cohort, not more depth on the same
37 PFS events.
