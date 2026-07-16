# Graph Usefulness Experiments

These checks test whether graph structure adds information beyond tabular features.

## Graph-Summary Baseline

```powershell
python -m research.graph_specific_baselines `
  --config research_config_v2.yaml `
  --seeds 7 21 42 123 2026 `
  --device cuda
```

Compare `graph_summary_mlp_cox` with the structure-aware GNN and the all-tabular MLP Cox baseline.

## Graph Perturbation

```powershell
python -m research.graph_structure_tests_v2 `
  --config research_config_v2.yaml `
  --seeds 7 21 42 123 2026 `
  --split-seed 42 `
  --device cuda `
  --output-root outputs/current_mainline_v2/cox_fixed_split_graph_structure_tests
```

Required variants are `original`, `shuffle_weights`, `shuffle_edges`, and `shuffle_edges_and_weights`. A reproducible drop after perturbation supports use of topology, but it does not by itself prove biological causality.

## Current Interpretation

The Cox reference showed perturbation gaps of about `0.0232-0.0254` C-index. The accepted temporal-topology AFT release additionally preserves ten named edge identities rather than only global graph summaries. Its formal cross-split result is documented in `CURRENT_MAINLINE.md`.

All claims remain limited to the synthetic/noisy augmented `topology_v6` dataset until external real-cohort validation is available.
