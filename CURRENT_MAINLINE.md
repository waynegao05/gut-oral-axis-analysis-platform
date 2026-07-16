# Current Mainline

## Release

- release: `temporal_topology_aft_cross_split_consensus_v1`
- default web backend: `temporal_topology_aft_cross_split_consensus`
- task: right-censored survival risk prediction
- labels: `time`, `event`
- primary metric: C-index
- dataset: `topology_v6` synthetic/noisy augmented research data

## Architecture

The accepted release combines two split-specific branches (`42`, `43`). Each branch contains:

- 3 selected structure-aware GNN risk models;
- 5 temporal-topology XGBoost AFT experts using named node and edge features;
- validation-selected standardized-risk fusion with shared `alpha = 0.63`.

The two branch risks are averaged for web inference. In total, the release uses 6 GNN and 10 AFT members.

## Formal Evidence

Fusion weight selection uses validation data only. Test labels are not used for model or alpha selection.

| Split | Reference test C-index | Selected test C-index | Delta | Cox loss delta |
|---:|---:|---:|---:|---:|
| 42 | 0.743263 | 0.760866 | +0.017603 | -0.014535 |
| 43 | 0.737404 | 0.753247 | +0.015843 | -0.011890 |
| Mean | 0.740333 | **0.757056** | **+0.016723** | **-0.013212** |

The held-out metrics use research-table topology and deterministic eight-sample evaluation batches. They are not a direct validation score for the deployment-time average web model.

The historical exploratory score `0.8967` is retained only as a potential upper-bound observation. It did not survive the current formal rerun protocol and must not be reported as the current result.

## Web Topology

The web form does not directly measure functional scores or microbial edge weights. For each split, standardized Ridge models trained only on that split's 2160 training samples infer:

- 5 node function scores;
- 10 named edge weights.

These values are labeled as inferred topology in every response. Fixed median batch-normalization anchors make one patient's GNN score independent of other concurrent web requests.

## Current Code

- web bridge: `src/temporal_topology_bridge.py`
- web dispatch: `src/pipeline.py`
- GNN reference: `research/model_v2.py`
- GNN training: `research/train_v2.py`
- full-risk refiner: `research/full_risk_head_refiner_v2.py`
- AFT fusion: `experiments/temporal_independent_v3/topology_aft_fusion.py`
- five-seed runner: `experiments/temporal_independent_v3/seed_sweep.py`
- split consensus: `experiments/temporal_independent_v3/cross_split_consensus.py`
- config: `research_config_v2.yaml`

## Required Local Artifacts

`outputs/` is intentionally excluded from Git. The default web backend requires local artifacts under:

```text
outputs/current_mainline_v2/full_risk_head_refiner_v2/
outputs/current_mainline_v2/full_risk_expert_v2/
outputs/current_mainline_v2/temporal_independent_v3/
```

Missing artifacts produce an explicit startup error. There is no silent fallback.

## Backend Selection

Default:

```powershell
$env:GOA_MODEL_BACKEND = "temporal_topology"
```

Archived Cox comparison backend:

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
