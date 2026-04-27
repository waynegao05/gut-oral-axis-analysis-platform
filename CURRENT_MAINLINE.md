# Current Mainline

This repository currently treats the research mainline as a conservative survival-analysis workflow rather than a complexity-first model branch.

## Main task

- right-censored survival risk prediction
- canonical labels: `time`, `event`
- primary metric: `c-index`

Detailed task notes live in:

- `research/TASK_DEFINITION.md`

## Current mainline code

- model: `research/model_v2.py`
- training: `research/train_v2.py`
- repeated runs: `research/repeat_runs_v2.py`
- graph perturbation: `research/graph_structure_tests_v2.py`
- graph preprocess sweep: `research/graph_preprocess_sweep.py`
- fixed-split head-to-head benchmark: `research/fixed_split_benchmark.py`
- baselines: `research/baseline_compare.py`
- graph-only baselines: `research/graph_specific_baselines.py`
- ensemble evaluation: `research/ensemble_v2.py`
- ablation entry: `research/ablate_structure_aware_v1.py`

## Current mainline config

- `research_config_v2.yaml`
- optional discrete-time branch: `research_config_v2_discrete_test.yaml`

## Mainline principles

- preserve a GNN-based research direction
- keep `research_config_v2.yaml` as the default `GNN + Cox` mainline
- keep task semantics survival-consistent across all runners
- prefer reproducibility and baseline validation over architectural novelty
- keep speculative complexity optional, not default

## Recommended commands

CUDA is the default for current deep-learning runners. Use `--device cpu` only for explicit CPU debugging.

For the current Windows + CUDA research environment (`torch 2.8.0+cu126`), install `torch_scatter`
from the official PyG wheel index to avoid `scatter(reduce='max')` performance warnings:

```bash
python -m pip install --no-cache-dir --force-reinstall torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

Single run:

```bash
python -m research.train_v2 --config research_config_v2.yaml --device cuda
```

Repeated runs:

```bash
python -m research.repeat_runs_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --device cuda
```

Graph perturbation:

```bash
python -m research.graph_structure_tests_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --device cuda
```

Graph preprocess sweep:

```bash
python -m research.graph_preprocess_sweep --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda
```

Fixed-split benchmark:

```bash
python -m research.fixed_split_benchmark --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cuda
```

Baseline comparison:

```bash
python -m research.baseline_compare --config research_config_v2.yaml --seeds 7 21 42 123 2026 --device cuda
```

Graph-specific baselines:

```bash
python -m research.graph_specific_baselines --config research_config_v2.yaml --seeds 7 21 42 123 2026 --device cuda
```
