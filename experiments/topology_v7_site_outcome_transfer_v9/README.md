# topology_v7 site-resolved outcome-transfer expert

This experiment is independent from the deployed model and does not alter the
V7 data files.

The candidate is a single Cox boosting expert. It consumes clinical,
metabolite, node abundance, node function, and paired saliva/stool features.
Precomputed edge weights are excluded; nonlinear relations are learned inside
the trees.

One optional scalar prior is learned from the open Debelius tumour-microbiome
cohort. That cohort supplies a fixed-horizon survival contrast, not
right-censored follow-up, so the score is only an auxiliary transfer prior. It
is admitted only when its repeated cross-validated AUC is at least 0.60.

The development cohort is screened first:

```powershell
python -m experiments.topology_v7_site_outcome_transfer_v9.development_screen
```

The formal V7 cohort must not be run unless the predeclared development gate
passes and a separate immutable formal lock is written.
