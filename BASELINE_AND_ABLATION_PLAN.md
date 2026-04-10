# Baseline comparison and ablation plan

This document defines a practical experiment matrix for the current repository.

## Why these baselines

The current repository already contains a graph-based multi-modal survival model. For a paper-level result section, the key question is not whether the GNN can run, but whether it improves upon simpler non-graph tabular alternatives under the same data split.

The baseline script added in `research/baseline_compare.py` is designed to be runnable with the existing repository dependencies and without adding extra survival libraries.

## Runnable baseline matrix

All runnable baselines use the same train/val/test split logic and report test C-index.

### Linear Cox-style baselines

1. `clinical_linear_cox`
   - Inputs: `age`, `bmi`, `smoking`, `family_history`
   - Purpose: lowest-complexity clinical-only baseline

2. `metabolite_linear_cox`
   - Inputs: `bile_acids`, `scfa`, `tryptophan_metabolism`
   - Purpose: metabolite-only baseline

3. `clinical_metabolite_linear_cox`
   - Inputs: clinical + metabolite features
   - Purpose: non-graph multimodal linear baseline

4. `graph_summary_linear_cox`
   - Inputs: graph-summary tabular features extracted from node abundance, node function scores, and edge statistics
   - Purpose: tests whether graph information helps even without a GNN

5. `all_tabular_linear_cox`
   - Inputs: clinical + metabolite + graph-summary features
   - Purpose: strongest simple linear baseline

### MLP Cox-style baselines

6. `clinical_metabolite_mlp_cox`
   - Inputs: clinical + metabolite features
   - Purpose: tests whether nonlinear tabular modeling is already sufficient without graph structure

7. `all_tabular_mlp_cox`
   - Inputs: clinical + metabolite + graph-summary features
   - Purpose: strongest non-graph nonlinear baseline currently runnable in the repository

## Main comparison target

The main model remains the graph-based multi-modal model trained with:

```bash
python -m research.train --config research_config_small_sample.yaml
```

## How to run baseline comparison

```bash
python -m research.baseline_compare --config research_config_small_sample.yaml --seeds 7 21 42 123 2026
```

The output file will be written to:

```text
outputs/baseline_compare_summary.json
```

## Recommended paper-level comparison table

Main text table:

- GNN multimodal model
- clinical_linear_cox
- clinical_metabolite_linear_cox
- all_tabular_linear_cox
- clinical_metabolite_mlp_cox
- all_tabular_mlp_cox

Supplementary table:

- metabolite_linear_cox
- graph_summary_linear_cox

## Recommended ablation matrix for the main model

These ablations should answer which modality contributes to performance.

### Modality ablations

A1. Clinical only  
A2. Metabolite only  
A3. Graph only  
A4. Clinical + metabolite  
A5. Clinical + graph  
A6. Metabolite + graph  
A7. Clinical + metabolite + graph (full model)

### Architecture ablations

A8. Replace GAT with graph-summary tabular linear baseline  
A9. Replace GAT with graph-summary tabular MLP baseline  
A10. Reduce graph branch and keep only tabular fusion  
A11. Keep graph branch but remove metabolite fusion  
A12. Keep graph branch but remove clinical fusion

## Minimal priority order

If time is limited, prioritize in this order:

1. Full GNN model vs `all_tabular_mlp_cox`
2. Full GNN model vs `clinical_metabolite_mlp_cox`
3. Full GNN model vs `clinical_metabolite_linear_cox`
4. Modality ablation: graph removed
5. Modality ablation: metabolite removed
6. Modality ablation: clinical removed

## Important interpretation note

If the GNN only matches, but does not exceed, `all_tabular_mlp_cox`, then the paper should not claim graph superiority. In that case the honest conclusion is that graph modeling is feasible but its incremental value over strong tabular baselines remains limited under the current synthetic or small-sample setting.
