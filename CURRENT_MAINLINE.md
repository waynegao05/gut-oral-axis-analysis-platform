# Current Mainline

This file defines the **current best research path** in the repository so that experimental branches do not get mixed with the main workflow.

## Best current model

The current mainline model is:

- `research/model_structure_aware.py`

The corresponding training entry is:

- `research/train_structure_aware.py`

The repeated-run entry is:

- `research/repeat_runs_structure_aware.py`

The graph perturbation entry is:

- `research/graph_structure_tests_structure_aware.py`

## Best current config

Use this config as the mainline config:

- `research_config_structure_aware_best.yaml`

This config is the tuned v1 setting currently preferred over the default v1 config and over the v2 branch.

## Recommended commands

### Single run

```bash
python -m research.train_structure_aware --config research_config_structure_aware_best.yaml --device cpu
```

### Repeated runs

```bash
python -m research.repeat_runs_structure_aware --config research_config_structure_aware_best.yaml --seeds 7 21 42 123 2026 --device cpu
```

### Graph perturbation test

```bash
python -m research.graph_structure_tests_structure_aware --config research_config_structure_aware_best.yaml --seeds 7 21 42 123 2026 --device cpu
```

## What is no longer the mainline

The following branches are no longer the preferred mainline:

- default `research/train.py` branch for paper-level final experiments
- `edge_aware` branch
- `structure_aware_v2` branch
- tuning scripts as direct training entrypoints

They remain useful for comparison, but they should be treated as supporting experiments rather than the primary path.

## Practical rule

If you only want to continue the strongest current paper-facing model, use:

- `research/train_structure_aware.py`
- `research_config_structure_aware_best.yaml`

and ignore the rest unless you are doing baselines, ablations, or historical comparison.
