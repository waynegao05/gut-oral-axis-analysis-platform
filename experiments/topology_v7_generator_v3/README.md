# topology_v7 generator-v3 development gate

This folder is independent from the deployed model and from the existing
generator-v2 dataset.

The development benchmark compares generator-v3 and generator-v2 with the same
explicit five-group leave-one-group-out protocol. Each outer test group has one
fixed inner validation group; all remaining groups are training data. Public
anchor descendants are grouped during the domain-classifier cross-validation.

The current pilot is a development cohort. Its outer results may only decide
whether generator parameters are locked. After promotion, a new random seed is
used once to create the formal 3600-sample cohort. The pilot results are not
reported as final-model validation.

```powershell
python -m experiments.topology_v7_generator_v3.logo_benchmark `
  --candidate-data-dir outputs/topology_v7_generator_v3_pilots/covariance_compensated `
  --reference-data-dir data/research `
  --output-dir outputs/topology_v7_generator_v3_pilots/covariance_compensated_logo `
  --model-seeds 42 `
  --device cpu
```

For the independently generated formal cohort, add
`--scope formal_fixed_protocol_audit`. Formal outer-fold results must not change
the generator seed, GNN architecture, or fusion settings.
