# V10 performance decision

## Decision

Do not promote the AC-ICAM transfer prior to the formal V7 cohort. Keep the
current mainline unchanged.

The historical `0.761` result is not a stable cross-split baseline. The
reconstructed V6 cross-split mean was approximately `0.7571`. On the current
formal V7 cohort, even the generator's deterministic latent-risk ordering has
a macro C-index of `0.7567` and a pooled C-index of `0.7573`. This is not a
mathematical upper bound, but it shows that a stable `>0.761` claim is not
supported by the current outcome signal.

## V10 evidence

- AC-ICAM source cohort: 209 stage I-III patients with complete PFS, 37 events.
- Five-genus source Cox prior: mean repeated-CV C-index `0.5807`, minimum
  `0.5688`; the source gate passed.
- Published 41-genus MBR reference on these patients: C-index `0.7457`.
- Five-genus MBR distillation: Spearman `0.3181`, R2 `0.0769`; the
  distillation gate failed.
- V7 development baseline: C-index `0.742606`.
- Best AC-ICAM transfer method: C-index `0.742414`, delta `-0.000192`.
- The required development delta was `+0.005`; no formal V7 audit was run.

## Interpretation

The public cohort contains real prognostic information, but most of its useful
microbiome signal lies outside the five genera currently observed by V7. A
frozen prior computed from the shared five genera is largely redundant with
the current model and does not overcome independent outcome noise.

Further architecture stacking, CTM residuals, precomputed edge weights, or
test-set selection cannot create the missing information. Such changes may
produce a favorable split by chance but would not support a reproducible
`>0.761` claim.

## Required next benchmark

A legitimate attempt to exceed `0.761` needs a new, independently locked
real-outcome benchmark with additional measured inputs, not generated
surrogates:

1. Expand the measured microbiome panel beyond the current five genera,
   prioritizing the published AC-ICAM MBR/MRS taxa.
2. Include observed stage, treatment, recurrence/progression time, and
   censoring status.
3. Compute topology relationships inside the model from measured abundances.
4. Select the model with nested development folds and evaluate once on an
   untouched external cohort.
5. Report V7 as a synthetic development proxy and the real cohort as the
   clinical performance benchmark; do not mix their C-index claims.
