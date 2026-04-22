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

## Mainline principles

- preserve a GNN-based research direction
- keep task semantics survival-consistent across all runners
- prefer reproducibility and baseline validation over architectural novelty
- keep speculative complexity optional, not default

## Recommended commands

Single run:

```bash
python -m research.train_v2 --config research_config_v2.yaml --device cpu
```

Repeated runs:

```bash
python -m research.repeat_runs_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --device cpu
```

Graph perturbation:

```bash
python -m research.graph_structure_tests_v2 --config research_config_v2.yaml --seeds 7 21 42 123 2026 --device cpu
```

Graph preprocess sweep:

```bash
python -m research.graph_preprocess_sweep --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cpu
```

Fixed-split benchmark:

```bash
python -m research.fixed_split_benchmark --config research_config_v2.yaml --seeds 7 21 42 123 2026 --split-seed 42 --device cpu
```

Baseline comparison:

```bash
python -m research.baseline_compare --config research_config_v2.yaml --seeds 7 21 42 123 2026
```

Graph-specific baselines:

```bash
python -m research.graph_specific_baselines --config research_config_v2.yaml --seeds 7 21 42 123 2026
```
