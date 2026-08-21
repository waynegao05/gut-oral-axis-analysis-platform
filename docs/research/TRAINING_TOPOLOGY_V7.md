# Training topology_v7

## Scope

The reference configuration `config/research/research_config_v2.yaml` uses topology_v7 generator_v2. All new output
is written below `outputs/current_mainline_v3/topology_v7_generator_v2`.
Generator_v1 runs remain under `outputs/current_mainline_v3/topology_v7` and are
also preserved under `archive/model_releases/topology_v7_generator_v1`.

The v7 cohort is model-generated. Its train, validation, and test partitions
must remain disjoint by `generation_group_id`. A v7 C-index measures recovery
of generated structure and is not external clinical validation.

## 1. Verify the environment

```powershell
nvidia-smi
python -m pytest -q tests/test_topology_v7.py tests/test_research_data_validation.py
```

The local RTX 4060 has 8 GB VRAM. Start with batch size 8. If CUDA reports an
out-of-memory error, retry with `-BatchSize 4`; do not run seeds in parallel.

## 2. Smoke training

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_topology_v7.ps1 `
  -Mode smoke -Device cuda -BatchSize 8
```

This runs two epochs and writes to:

```text
outputs/current_mainline_v3/topology_v7_generator_v2/smoke/split42_seed42/
```

## 3. One full model

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_topology_v7.ps1 `
  -Mode single -Device cuda -SplitSeed 42 -BatchSize 8 -Epochs 180 -Patience 18
```

Inspect `test_metrics.json`, `split_summary.json`, and `history.json` before
starting repeated runs. The split summary must report
`generation_group_disjoint_train_val_test_split`.

## 4. Five model seeds

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_topology_v7.ps1 `
  -Mode five-seed -Device cuda -SplitSeed 42 -BatchSize 8
```

The fixed model seeds are `7 21 42 123 2026` and run sequentially. Results are
summarized in:

```text
outputs/current_mainline_v3/topology_v7_generator_v2/split42_five_seed/research_repeat_runs_summary.json
```

## 5. Two split-specific five-seed runs

Run this only after the single-split results are stable:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_topology_v7.ps1 `
  -Mode two-split -Device cuda -BatchSize 8
```

Split 42 holds out generation group 1; split 43 holds out generation group 3.
This gives two distinct source-group tests before any ensemble or AFT fusion.

## 6. Required comparisons

```powershell
python -m research.baseline_compare `
  --config config/research/research_config_v2.yaml `
  --seeds 7 21 42 123 2026 `
  --split-seed 42 `
  --device cuda `
  --output-root outputs/current_mainline_v3/topology_v7_generator_v2/baselines_split42

python -m research.graph_structure_tests_v2 `
  --config config/research/research_config_v2.yaml `
  --seeds 7 21 42 123 2026 `
  --split-seed 42 `
  --device cuda `
  --output-root outputs/current_mainline_v3/topology_v7_generator_v2/graph_tests_split42
```

Do not promote v7 to the web backend from a single run. Promotion requires two
split-specific held-out results, five-seed stability, baseline comparison,
graph perturbation, loss/calibration review, artifact replay, and a separate
real-cohort evaluation.
