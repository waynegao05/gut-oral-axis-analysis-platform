# Research Configurations

These YAML files configure research and training workflows. Commands and paths in the files are resolved from the repository root unless a script states otherwise.

- `research_config_v2.yaml`: reference topology V7 configuration used by the general research entry points.
- `research_config_v7_gnn_final.yaml`: finalized GNN candidate configuration.
- `research_config_v7_gnn_fullrisk.yaml`: full-risk GNN training variant.
- `research_config_v7_gnn_optimized.yaml`: tuned GNN experiment with auxiliary and ranking losses.
- `research_config_v7_v3_gnn_locked.yaml`: locked V7 V3 reproducibility configuration.

Release-time application settings remain in `config/releases/`; they are intentionally separate from training configuration.
