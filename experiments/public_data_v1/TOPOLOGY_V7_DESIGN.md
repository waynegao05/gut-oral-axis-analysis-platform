# topology_v7 hybrid generated cohort

## Purpose

`topology_v7` is a model-generated development cohort. It combines the joint
oral-gut microbiome distribution from the open Russo CRC cohort with the
existing `topology_v6` priors needed to fill modalities that are not publicly
available at patient level.

The 3,600 rows are not 3,600 observed patients. They are 3,600 unique draws
from a reproducible conditional generator. The original 42 paired public
patients remain untouched under `data/public/russo_crc_oral_gut_2023`.

## Generation contract

| Modality | v7 method | Evidence status |
|---|---|---|
| Paired saliva/stool microbiome | Class-conditional PCA and local Gaussian sampling from 42 public paired patients | Anchored to real measurements; generated rows |
| Five-taxon model panel | Site-prevalence fusion followed by v6 marginal calibration | Generated and calibrated |
| Clinical variables | Random-forest conditional model plus out-of-bag residual sampling from v6 | Synthetic prior |
| Metabolic pathways | Multi-output random forest plus correlated residual sampling from v6 | Synthetic imputation |
| Function scores | Random-forest prediction from abundance, clinical, and metabolite inputs | Model inferred, not measured |
| Edge weights | Graphical-lasso partial associations learned from the public cohort with sample-specific modulation | Model inferred, not measured |
| Survival outcome | Transparent multimodal latent risk, controlled log-normal AFT event time, and feature-independent censoring | Fully generated proxy |

Generator_v2 fixes the residual-dominated generator_v1 outcome. Its fixed
event-noise-to-signal ratio is 1.25, the deterministic latent-risk C-index must
remain within 0.72-0.80, and every generation group must remain at or above
0.68. These are synthetic development constraints, not clinical claims. The
latent risk and hidden event/censor times are stored only in provenance and are
explicitly prohibited as model features.

## Leakage control

Public anchors are assigned to five non-overlapping generation groups. Both
anchors used for any generated sample come from the same group. The central
data splitter detects `generation_group_id` and keeps groups disjoint across
training, validation, and test sets. A random row-level split is prohibited for
this dataset.

## Interpretation boundary

- v7 can be used to develop code, compare architectures, and test whether a
  model recovers the generator's structured signal.
- v7 cannot establish clinical prognosis, treatment utility, real-world
  calibration, or external generalization.
- A higher v7 C-index is not equivalent to a higher C-index in an independent
  observed cohort.
- The 42-patient Russo dataset must continue to be evaluated separately for
  real-data classification and oral-gut mechanism replication.
- A cohort with observed follow-up and event status is still required for a
  genuine survival replacement.

## Rebuild

```powershell
python -m experiments.public_data_v1.build_topology_v7 --samples 3600 --seed 20260720
```

The command copies v6 into `archive/datasets/topology_v6`, preserves an existing
generator_v1 cohort under `archive/datasets/topology_v7_generator_v1`, writes
generator_v2 under `data/research`, and records input/output SHA-256 hashes in
`data/research/topology_v7_manifest.json`.
