# Data Placement

This experiment reads the existing repository tables by default:

- `data/research/topology_v6_sample_graph_table.csv`
- `data/research/topology_v6_sample_clinical_table.csv`
- `data/research/topology_v6_sample_metabolite_table.csv`
- `data/research/topology_v6_sample_label_table.csv`

The source tables are not copied into this directory. `topology_v6` is treated
as synthetic/noisy augmented research data. Replace the paths in a copied YAML
configuration when evaluating another dataset.
