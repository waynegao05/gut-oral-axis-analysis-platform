# Graph usefulness experiments

This document explains how to test whether the GNN branch is learning real structural information or only reusing node-level/tabular signals.

## Experiment 1: Graph-summary MLP baseline

Purpose:
- Determine whether graph-derived tabular summaries are already enough without an explicit graph neural network.

Run:

```bash
python -m research.graph_specific_baselines --config research_config_small_sample.yaml --seeds 7 21 42 123 2026
```

Output:

```text
outputs/graph_specific_baselines_summary.json
```

Interpretation:
- If `graph_summary_mlp_cox` is close to or better than the full GNN, then graph information may matter, but the explicit GNN architecture may not be necessary.
- If `graph_summary_mlp_cox` remains clearly below the full GNN, then the graph branch is likely extracting additional structure-aware signal.

## Experiment 2: Graph structure perturbation test

Purpose:
- Test whether performance depends on the true edge structure.

Run:

```bash
python -m research.graph_structure_tests --config research_config_small_sample.yaml --seeds 7 21 42 123 2026
```

Default variants:
- `original`
- `shuffle_weights`
- `shuffle_edges`
- `shuffle_edges_and_weights`

Output:

```text
outputs/graph_structure_tests/graph_structure_tests_summary.json
```

Interpretation:
- If `shuffle_edges` causes only a minor drop, then the current GNN is probably not using graph topology strongly.
- If `shuffle_edges_and_weights` still remains close to `original`, then the graph branch may mainly act as a nonlinear tabular encoder over node features.
- If both perturbations cause a clear drop, then the model is sensitive to true graph structure, which supports the claim that graph modeling contributes meaningful information.

## Recommended paper-level logic

Use the following decision structure:

1. Compare full GNN vs `all_tabular_mlp_cox`
2. Compare full GNN vs `graph_summary_mlp_cox`
3. Compare full GNN on original vs perturbed graphs

Possible conclusions:

### Case A
Full GNN > all tabular MLP, and perturbation causes a clear drop.

Interpretation:
- Strongest evidence that the graph branch adds real structural value.

### Case B
Full GNN ≈ all tabular MLP, but perturbation causes a drop.

Interpretation:
- Graph structure contributes, but current data/task still allows strong tabular baselines to compete closely.

### Case C
Full GNN ≈ graph-summary MLP, and perturbation causes little or no drop.

Interpretation:
- Current task does not yet justify graph neural modeling; graph-derived summaries are likely sufficient.

## Windows helpers

You can also create batch launchers locally if needed, but the core experiments are already runnable with the module commands above.
