# Archive Index

`archive/` stores superseded implementations and documents without deleting local research history. Archived code is not part of the default mainline unless an active module explicitly imports it.

## Categories

| Directory | Contents | Runtime status |
|---|---|---|
| `legacy_web_backends/` | Previous locked five-seed GNN + Cox web bridge | Optional through `GOA_MODEL_BACKEND=legacy_cox` |
| `legacy_root_configs/` | Superseded root-level experiment configurations | Historical or explicit manual use |
| `legacy_docs/` | Previous mainline notes, API examples, and workflow documents | Historical reference |
| `legacy_mainline_v1/` | First research mainline model and runners | Archived |
| `legacy_default_mainline/` | Earlier default GAT mainline | Archived |
| `legacy_models/` | Edge-aware and structure-aware model variants | Archived |
| `legacy_models_and_configs/` | Coupled legacy model/config snapshots | Archived |
| `legacy_trainers/` | Trainers and repeat/structure-test runners for old variants | Archived |
| `legacy_research_scripts/` | Superseded experiment scripts grouped by prior research route | Archived |
| `legacy_configs/` | Older edge-aware, small-sample, structure-aware, and prototype pharmacy-rule configs | Archived |
| `legacy_scripts/` | Previous app and batch entrypoints | Archived |
| `legacy_tools/` | Old tuning utilities | Archived |
| `legacy_examples/` | Previous request and API examples | Archived |
| `legacy_outputs_notes/` | Small historical summaries, not model checkpoints | Historical evidence only |

## Current Mainline Boundary

The active release is documented in the root `CURRENT_MAINLINE.md`. The default web bridge is `src/temporal_topology_bridge.py`.

The root `ctm_fusion_experiment/` directory is intentionally not moved here because several reproducibility scripts still import it. It remains a historical research dependency, not the production web backend.

The former `data/microbe_drug_rules.json` is retained as `legacy_configs/microbe_drug_rules_prototype.json`. The active pharmacy rule base is `data/pharmacy_rules_v3.json`; versioned medication-label and limited-interaction data live under `data/pharmacy_knowledge/`.

## Compatibility Rules

- Archived modules must not be imported by the default path unless explicitly documented.
- Moving a file into this directory must not silently change the active release.
- Paths retained for rollback should have a tested explicit selector.
- Historical metrics must not be presented as current formal results.
- `outputs/` remains local and is not duplicated into the archive.
