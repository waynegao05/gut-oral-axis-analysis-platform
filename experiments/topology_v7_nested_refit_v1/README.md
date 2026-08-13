# topology_v7 nested refit v1

This experiment is isolated from the deployed backend and from the existing
topology_v7 generator-v3 formal result.

The development cohort is generated once with seed `20261002`. Five
leave-one-generation-group-out development folds compare a small, predeclared
candidate set. Every development model trains on all four non-held-out groups.

After development selection, `protocol_lock.json` records the selected
candidate, fixed training epoch count, input hashes, and the unused audit seed.
Only then is the audit cohort generated once with seed `20261003`.

The audit trains five seeds on all four non-test groups and evaluates each
held-out group exactly once. Generation-group identifiers are used only for
splitting or grouped training loss. They are never model inputs.

The current GNN already contains an identity-preserving readout over named
nodes and named edges. The fixed-edge candidates therefore use only a small,
zero-gated residual over the ten canonical undirected edges; they do not
replace the existing graph encoder.

## Completed result

Development used generator seed `20261002`, one model seed, and five
leave-one-generation-group-out folds. The pooled Cox refit baseline reached a
macro C-index of `0.746682` (minimum group `0.734203`, mean Cox loss
`5.370240`). No candidate passed the predeclared `+0.001` development gate:

| Candidate | Macro C-index delta |
| --- | ---: |
| pooled Cox + IPCW ranking | `+0.000004` |
| mixed pooled/group Cox | `-0.000193` |
| mixed Cox + IPCW ranking | `-0.000123` |
| fixed-edge mixed/ranking residual | `-0.000128` |
| fixed-edge pooled residual | `-0.000004` |

The protocol therefore locked `baseline_pooled_cox` at 46 epochs before the
audit cohort was generated.

Audit used the previously unseen generator seed `20261003`, all five model
seeds (`7`, `21`, `42`, `123`, `2026`), and all four non-test groups for every
fit. The five-seed ensemble reached:

- macro mean C-index: `0.744038`
- train-standardized pooled OOF C-index: `0.744104`
- minimum held-out-group C-index: `0.725977`
- mean held-out Cox loss: `5.422701`
- mean individual-model C-index: `0.743150`

Ensembling improved every held-out group by `0.000591` to `0.001123` over its
five individual-model mean, but no new objective or residual branch qualified
for adoption. The final decision is `keep_baseline_refit_candidate`.

The earlier generator-seed-`20261001` formal GNN result (`0.749040` macro) is
retained. It used a different generated cohort and a three-group training plus
one-group validation protocol, so its difference from this four-group refit
audit is not a paired model comparison.

Primary artifacts:

- `outputs/topology_v7_nested_refit_v1/development/development_summary.json`
- `experiments/topology_v7_nested_refit_v1/protocol_lock.json`
- `outputs/topology_v7_nested_refit_v1/audit/audit_summary.json`

This experiment does not modify the deployed backend, the locked V7 formal
result, or any archived model.
