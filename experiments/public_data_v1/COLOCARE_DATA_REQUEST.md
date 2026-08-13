# ColoCare Data Request Draft

## Proposed project

External validation of an oral-gut-axis survival modelling workflow in newly
diagnosed stage I-III colorectal cancer, using baseline fecal microbiome,
clinical covariates, and right-censored disease-free survival.

## Minimum requested data

- de-identified participant identifier
- baseline pre-surgery stool sample identifier and collection timing
- processed ASV/OTU abundance table and taxonomy table
- sequencing depth and relevant processing/QC metadata
- age, sex, BMI, smoking status, recruitment site, tumour site, and AJCC stage
- recent antibiotic exposure and neoadjuvant/adjuvant treatment indicators
- DFS follow-up time, DFS event indicator, and exact event definition
- OS follow-up time and OS event indicator when available
- data dictionary, missing-value codes, and cohort inclusion/exclusion flags

Only de-identified or limited data are requested. No direct identifiers or dates
are required; elapsed times are sufficient.

## Analysis safeguards

- patient-level grouped splits and repeated nested cross-validation
- censoring-aware Cox/AFT evaluation with c-index and time-dependent AUC
- preprocessing and graph construction fitted inside each training fold
- no patient-level linkage to unrelated public cohorts
- explicit reporting that the cohort contains fecal, not matched oral, samples

The current request process and administrator address are recorded in
`datasets.json`.
