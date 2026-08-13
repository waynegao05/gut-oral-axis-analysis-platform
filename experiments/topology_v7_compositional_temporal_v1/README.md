# Topology V7 compositional-temporal experiment

This experiment is isolated from the deployed backend, the locked V7 formal
models, and the completed nested-refit experiment.

The first stage measures the existing five-seed V7 audit ensemble with one
consistent survival protocol:

- macro and pooled Harrell C-index
- censoring-adjusted Uno C-index
- cumulative/dynamic IPCW AUC at 36, 60, and 84 time units
- normalized integrated AUC over 24 to 96 time units
- IPCW Brier score and integrated Brier score
- diagnostic distance to the generator's deterministic latent-risk oracle

The seen audit cohort is diagnostic only and cannot select a future model.
Future development and audit cohorts have separate predeclared seeds, and the
audit cohort may be generated only after the candidate is locked.

Every externally added dataset must have an official project page or official
repository accession, direct download URL, access date, access terms, and file
hash. Observed source measurements and model-generated values remain explicitly
separated.

Graph edge weights are computed inside the model from permitted input features.
Precomputed edge weights, outcome labels, generation groups, sample IDs,
provenance, and audit-only latent variables cannot enter the edge calculator.
