# Residual CTM v4 C-Index Ensemble Design

## Purpose

v3 improved risk spread but did not reliably improve c-index. v4 removes the utility-objective selector and targets c-index directly.

## Changes

- Train multiple residual CTM seeds per fold while keeping the graph encoder and concat baseline fold-local.
- Add a pairwise ranking auxiliary loss on the final CTM residual risk.
- Build candidate residual corrections from single seeds, mean ensemble, top-2 ensemble, and validation-score softmax ensemble.
- Select the final candidate only by validation c-index.
- Report test c-index and comparable-pair diagnostics: improved pairs, regressed pairs, and net pair credit.

## Isolation

The flow is independent from v1/v2/v3:

- `ctm_fusion_experiment/train_residual_v4.py`
- `ctm_fusion_experiment/evaluate_residual_v4.py`
- `ctm_fusion_experiment/configs/residual_ctm_v4.yaml`
- `ctm_fusion_experiment/configs/residual_v4_smoke.yaml`
- `outputs/ctm_fusion_experiment/residual_v4_formal`
