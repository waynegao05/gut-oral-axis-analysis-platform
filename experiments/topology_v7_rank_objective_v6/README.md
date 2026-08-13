# Topology V7 rank-objective v6

V6 keeps the V5 exact internal edge computation and changes only the survival
training objective. The candidate adds a weak cumulative/dynamic horizon
ranking loss at 36, 60, and 84 time units to the Cox partial likelihood.

The coefficient 0.2 was selected before generating the V6 cohort from an
inner-validation-only prescreen on the V5 development cohort. Outer group 0
was excluded from all prescreen predictions and metrics.

For every V6 outer test group, all four remaining groups are each used once as
an inner validation group. The median of the four selected epochs is then
locked, and the model is trained from scratch on all four non-test groups.
This removes the single-validation-group dependency of V3-V5.

Development uses generator seed 20261016. Audit seed 20261017 is not generated
unless the candidate passes all locked C-index, integrated AUC, worst-group,
horizon AUC, and integrated Brier requirements.
