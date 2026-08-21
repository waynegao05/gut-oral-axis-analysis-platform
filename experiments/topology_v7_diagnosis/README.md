# topology_v7 diagnosis

This experiment separates four possible causes of the low topology_v7
C-index without changing the main training flow:

1. weak survival-label signal;
2. generation-group distribution shift;
3. loss of information in the current GNN representation;
4. optimization or preprocessing failure in the current GNN.

It compares validation-selected Linear Cox, MLP-Cox, and XGBoost AFT models
under both the production group-disjoint split and a diagnostic random split.
The random split is not a publishable result; it is only a learnability check.

`generation_group_id` is retained for splitting and is always excluded from
model features. The experiment writes only below its requested output folder.

The report also replays a saved GNN checkpoint, reads the archived v6 reference,
and reconstructs the v7 outcome generator in memory. The reconstruction measures
how much outcome ordering comes from input features versus an unobserved random
residual; it does not rewrite v7.

```powershell
python -m experiments.topology_v7_diagnosis.diagnose `
  --config config/research/research_config_v2.yaml `
  --split-seeds 42 43 `
  --model-seeds 7 42 2026 `
  --device cuda `
  --output-dir outputs/topology_v7_diagnosis
```

Primary outputs:

- `diagnosis_summary.json`: full machine-readable evidence;
- `diagnosis_report.md`: concise interpretation table;
- `benchmark_runs.csv`: one row per model, split, and seed.
