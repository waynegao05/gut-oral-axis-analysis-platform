# Topology V7 internal relation and oral-gut site v3

This experiment is isolated from the deployed backend and all earlier V7
workflows. It tests whether information lost by collapsing saliva and stool
into one graph abundance can improve survival ranking.

The new candidates never read CSV edge weights or cached graph-structure
features at inference. Each forward pass creates a complete symmetric graph.
The `equal` candidate lets the GAT attention layers learn relations internally;
the `site_context` candidate additionally generates bounded sample-specific
edge weights from node, saliva/stool, clinical, and metabolite inputs.

The protocol uses five nested LOGO folds. For each outer test group, the next
group is the inner validation group and the remaining three groups are used for
training. Checkpoint selection and standardization use training/validation only.
After selecting the epoch, the validation group is merged back into training
and the model is refit from scratch on all four non-test groups. The outer test
group is then evaluated once.

Development uses one predeclared model seed on a fresh generated cohort. A
candidate must pass all C-index, time-dependent AUC, Brier, subgroup, and
worst-group gates before the audit cohort can be generated. The one-time audit
then compares the locked candidate with the paired legacy baseline using seeds
`7`, `21`, `42`, `123`, and `2026`.

Only one temporal auxiliary candidate is included: the site-conditioned model
with a fixed `0.05` log-normal AFT loss. There is no AFT-weight grid or
test-driven alpha search.

Actual public data files remain under `data/public/` and are excluded from Git.
The source catalog and reproducible download/preparation code live in
`experiments/topology_v7_compositional_temporal_v1/`.
