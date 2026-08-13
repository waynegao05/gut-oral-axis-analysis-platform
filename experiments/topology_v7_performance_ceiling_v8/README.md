# topology_v7 performance-ceiling audit

This experiment is isolated from the deployed model. It does not modify the
V7 dataset, the locked GNN, or any web-backend path.

The audit-only target `survival_latent_risk` is read from provenance to answer
one narrow question: can the legal model inputs recover the deterministic
signal that generated the noisy survival proxy? It must never be used as a
candidate training target, deployment feature, or formal model result.

The evaluation uses five explicit outer generation-group holdouts. For each
outer group, the next group is used to select a latent-risk surrogate, and the
remaining three groups are used for fitting. The selected surrogate is refit on
the four non-test groups before the outer diagnostic evaluation.

Run:

```powershell
python -m experiments.topology_v7_performance_ceiling_v8.ceiling_audit
```

Outputs are written below
`outputs/topology_v7_performance_ceiling_v8/audit_only`.
