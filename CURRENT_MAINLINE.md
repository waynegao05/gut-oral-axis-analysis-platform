# Current Mainline

## Release

- platform release: `CTM2.0`
- deployed model release: `ac_icam_real_outcome_pfs_v8`
- default web backend: `ac_icam_real_outcome_clinical_pfs`
- task: colorectal-cancer progression-free survival risk prediction
- endpoint: real PFS time and censoring status
- primary metric: C-index
- deployment cohort: 246 AC-ICAM patients with 71 PFS events
- default web model: clinical core
- optional web model: clinical + measured tumor-RNA ICR
- pharmacy layer: `pharmacy_assistance_v3`

## Architecture

The release contains five full-cohort ridge-Cox deployment members. Each member
selects its regularization strength under one of the locked seeds
`7, 21, 42, 123, 2026`. Deployment risks are standardized and mapped to the
five-seed OOF reference distribution.

The web form requires age and sex (18-75 years). AJCC stage, pathological
T/N/M, tumor location, and morphology are optional form inputs, but V8 PFS is
calculated only when all are present. Incomplete oncology input produces an
explicit not-calculated PFS result. If the five core microbes are complete,
the response separately exposes a `topology_v6` research reference percentile
for non-PFS visualization; it is not a general-population cancer probability
or screening result. A measured tumor-RNA `icr_score` activates the expanded
model only for an otherwise complete PFS input. Microbiome and treatment
fields do not alter the deployed PFS score.

## Formal Evidence

The formal V8 estimates use repeated outer five-fold evaluation with inner
selection in every outer training fold.

| Model | PFS C-index | Bootstrap 95% CI | AUC36 | AUC60 |
|---|---:|---:|---:|---:|
| Clinical core | **0.7756** | 0.7214-0.8251 | 0.8185 | 0.8013 |
| Clinical + measured ICR | **0.7845** | 0.7328-0.8323 | 0.8294 | 0.8133 |
| Clinical + microbiome safe blend | 0.7740 | 0.7197-0.8237 | 0.8177 | 0.7984 |

These are internal repeated-cross-validation estimates, not external clinical
validation. The ICR result applies only when tumor RNA was actually measured.

The historical exploratory score `0.8967` is retained only as a potential upper-bound observation. It did not survive the current formal rerun protocol and must not be reported as the current result.

## Archived Temporal-Topology Backend

The previous temporal-topology backend remains selectable for research
comparison. Its web form does not directly measure functional scores or
microbial edge weights. For each split, standardized Ridge models infer:

- 5 node function scores;
- 10 named edge weights.

These values are labeled as inferred topology in every response. Fixed median batch-normalization anchors make one patient's GNN score independent of other concurrent web requests.

## Current Code

- current web bridge: `src/ac_icam_v8_bridge.py`
- web dispatch: `src/pipeline.py`
- deployment trainer: `experiments/ac_icam_real_outcome_v8/deployment.py`
- deployment artifact: `config/releases/ac_icam_real_outcome_pfs_v8.json`
- formal V8 benchmark: `experiments/ac_icam_real_outcome_v8/benchmark.py`
- previous web bridge: `src/temporal_topology_bridge.py`
- GNN reference: `research/model_v2.py`
- GNN training: `research/train_v2.py`
- full-risk refiner: `research/full_risk_head_refiner_v2.py`
- AFT fusion: `experiments/temporal_independent_v3/topology_aft_fusion.py`
- five-seed runner: `experiments/temporal_independent_v3/seed_sweep.py`
- split consensus: `experiments/temporal_independent_v3/cross_split_consensus.py`
- pharmacy engine: `src/pharmacy_engine.py`
- drug knowledge runtime: `src/drug_knowledge.py`
- pharmacy knowledge base: `data/pharmacy_rules_v3.json`
- medication label and limited DDI data: `data/pharmacy_knowledge/`
- config: `research_config_v2.yaml`

## topology_v7 Training Candidate

The root research config now points to 3,600 unique model-generated v7 samples
from `topology_v7_hybrid_generator_v2`.
Their paired oral-gut microbiome distribution is learned from 42 real public
patients. Clinical variables, metabolite pathways, function scores, graph
weights, and censored outcomes remain generated or inferred and are explicitly
tracked in `data/research/topology_v7_sample_provenance.csv`.

The superseded generator-v1 dataset is preserved under
`archive/datasets/topology_v7_generator_v1`. Its existing model artifacts are
also preserved under `archive/model_releases/topology_v7_generator_v1`; the
new generator-v2 training output uses a separate directory.

The V7 generator and GNN work remains a separate research candidate. It is not
used by the AC-ICAM V8 web risk score. V7 results must remain labeled as
model-generated development evidence.

## Pharmacy Assistance

The web and raw-clinical workflows share one versioned pharmacy-assistance engine. It gates marker interpretation, normalizes submitted medications with a local RxNorm snapshot, returns product-specific openFDA/DailyMed label evidence, screens 14 implementable rules from a 15-rule minimum high-priority DDI set, and can expose indication-gated probiotic strain options. It does not perform comprehensive interaction screening, patient-specific dose adjustment, prescribing, or clinical diagnosis. See `PHARMACY_ASSISTANCE.md` for the active contract and limitations.

## Required Local Artifacts

The default V8 artifact is required locally but intentionally excluded from
GitHub because it contains fitted model parameters:

```text
config/releases/ac_icam_real_outcome_pfs_v8.json
```

It can be rebuilt in an approved local environment from the processed cohort
and formal OOF results with:

```powershell
python -m experiments.ac_icam_real_outcome_v8.deployment
```

The archived temporal-topology comparison still requires its local
`outputs/current_mainline_v2/` artifacts. There is no silent fallback.

## Backend Selection

Default:

```powershell
$env:GOA_MODEL_BACKEND = "ac_icam_v8"
```

Previous temporal-topology comparison:

```powershell
$env:GOA_MODEL_BACKEND = "temporal_topology"
```

Archived Cox comparison:

```powershell
$env:GOA_MODEL_BACKEND = "legacy_cox"
```

The compatibility implementation is stored at `archive/legacy_web_backends/cox_ensemble_v1.py`.

## Mainline Rule

New candidates should replace this release only when they:

- improve predeclared held-out splits without test-label selection;
- preserve survival and censoring semantics;
- report both discrimination and loss/calibration behavior;
- separate measured research topology from web-inferred topology;
- pass reproducibility, artifact replay, and deployment determinism checks.
