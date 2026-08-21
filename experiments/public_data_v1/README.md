# Public Data V1

This directory contains the public-cohort audit and the reproducible
`topology_v7_hybrid_generator_v2` hybrid generator. The public source files
remain isolated and unchanged. `config/research/research_config_v2.yaml` targets v7 for new research training,
while the deployed web release remains pinned to archived v6 data until v7 is
fully retrained and validated.

The superseded generator-v1 dataset and its model artifacts are retained under
`archive/datasets/topology_v7_generator_v1` and
`archive/model_releases/topology_v7_generator_v1`. Generator-v2 writes new
training artifacts to its own output directory.

## Finding

No currently registered cohort is a download-ready, patient-matched replacement
for all of the current inputs:

- oral microbiome
- gut microbiome
- clinical covariates
- metabolomics
- exact right-censored `time` and `event`

The main limitation is not file format. It is that the required modalities and
survival outcomes are not publicly available for the same patients in one
cohort.

## Recommended evidence design

1. Preserve `topology_v6` under `archive/datasets/topology_v6` as the released
   model's reproducibility benchmark.
2. Use `PRJNA899104 / GSE217490` for patient-matched oral-gut external
   validation. Its endpoint is adenoma/CRC stage, not survival. The paper
   reports NMR results, but the public files do not include a patient-level
   metabolomics matrix.
3. Request the ColoCare baseline stool, covariates, DFS time, and DFS event data
   for a real right-censored survival benchmark.
4. Use the open tissue microbiome cohorts only for their declared tasks. Tissue
   sequencing must not be presented as a measured oral or fecal microbiome.
5. Never join unrelated real cohorts row by row and describe the result as
   measured multimodal patients.
6. Treat all 3,600 v7 rows as generated development samples, not as additional
   observed patients.
7. Use generation-group-disjoint splits for every v7 evaluation.

The full v7 contract is documented in `TOPOLOGY_V7_DESIGN.md`.

## Audit

```powershell
python -m experiments.public_data_v1
python -m experiments.public_data_v1 --json
```

The registry is in `datasets.json`. Raw and processed public data belong under
`data/public/<dataset-id>/` and are ignored by Git by default.

## Prepare the open oral-gut cohort

The processed GSE217490 feature table can be downloaded and converted without
handling the raw SRA reads:

```powershell
python -m experiments.public_data_v1.prepare_gse217490
```

This creates patient-level paired saliva/stool features and a provenance
manifest under `data/public/russo_crc_oral_gut_2023/`. The endpoint is
adenoma-versus-adenocarcinoma classification. The converter deliberately does
not create `time` or `event` columns.

## Conversion rules

- The patient is the split unit; paired sites and repeated samples must never be
  divided across train, validation, and test sets.
- Use baseline samples for survival prediction unless the task explicitly uses
  time-varying covariates.
- Construct co-occurrence or topology edges using the training fold only, then
  freeze them for validation and test samples.
- Preserve exact source units and censoring definitions in a cohort manifest.
- In external-validation analyses, missing modalities stay missing. The v7
  development generator may model missing modalities only when their synthetic
  provenance remains explicit and no clinical evidence claim is made.
- Recalibrate the web risk bands and pharmacy markers before any public cohort
  can become a deployed backend.
