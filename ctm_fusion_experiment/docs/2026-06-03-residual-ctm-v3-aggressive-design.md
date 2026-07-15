# Residual CTM v3 Aggressive Design

## Purpose

Direct CTM replacement degraded c-index. Residual CTM v1/v2 showed that CTM is more reasonable as a correction on top of the concat-Cox baseline, but the observed c-index gain is still small and not statistically significant. v3 keeps the residual architecture and makes the experiment more aggressive without touching earlier flows.

## Design

v3 uses the same fold-local graph encoder, concat baseline, and residual CTM training path as v2. The independent changes are:

- Wider residual correction capacity: `max_residual_gate` is raised to `1.5`, and the initial gate is less conservative.
- Less baseline distillation pressure and more separation pressure in the residual loss.
- A wider alpha search grid, including negative alpha and alpha greater than `1.0`.
- Validation-time alpha selection with a composite objective instead of pure c-index.

The selected test risk is:

```text
final_risk = baseline_risk + selected_alpha * ctm_delta
```

## Composite Objective

The validation objective is measured as improvement over the fold's baseline:

```text
c_index_delta
+ top_event_lift_weight * top_event_lift_delta
+ high_low_gap_weight * high_low_event_gap_delta
+ risk_spread_weight * relative_risk_std_delta
```

The default config keeps c-index dominant but allows utility gains to break ties or justify a small c-index tradeoff. `min_c_index_delta: -0.001` prevents the selector from accepting a large c-index regression. `min_objective_delta: 0.00005` prevents tiny numerical differences from selecting an aggressive alpha when the validation objective is effectively unchanged from baseline.

## New Metrics

v3 reports c-index plus operational risk-stratification metrics:

- `top_event_lift`: event rate in the highest-risk band divided by the overall event rate.
- `high_low_event_gap`: top-risk event rate minus bottom-risk event rate.
- `risk_std`: risk-score spread.
- `time_separation`: bottom-risk mean observed time minus top-risk mean observed time.

These metrics are not clinical validation. They are secondary signals for whether the residual CTM is improving risk concentration beyond the original c-index.

## Isolation

New v3 files are independent:

- `ctm_fusion_experiment/train_residual_v3.py`
- `ctm_fusion_experiment/evaluate_residual_v3.py`
- `ctm_fusion_experiment/configs/residual_ctm_v3.yaml`
- `ctm_fusion_experiment/configs/residual_v3_smoke.yaml`
- `ctm_fusion_experiment/utils/aggressive_calibration.py`
- `ctm_fusion_experiment/utils/utility_metrics.py`

v1 and v2 scripts/configs keep their original behavior.
