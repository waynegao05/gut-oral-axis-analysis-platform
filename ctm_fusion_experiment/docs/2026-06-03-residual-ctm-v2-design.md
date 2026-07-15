# Residual CTM v2: Validation-Calibrated Residual Gate

## Purpose

Residual CTM v1 avoided the direct CTM replacement collapse but produced only a
small, non-significant mean gain. v2 keeps the same residual training setup and
adds fold-local validation calibration for the residual strength.

## Rule

After training CTM delta, choose `alpha` on the validation split:

```text
final_risk = baseline_risk + alpha * ctm_delta
alpha in config["calibration"]["alpha_grid"]
```

The grid includes `alpha=0.0`, so the calibrated model can explicitly fall back
to the concat baseline if CTM delta hurts validation c-index.

## Independent Entry Points

- training: `python -m ctm_fusion_experiment.train_residual_v2`
- evaluation: `python -m ctm_fusion_experiment.evaluate_residual_v2`
- formal config: `ctm_fusion_experiment/configs/residual_ctm_v2.yaml`
- smoke config: `ctm_fusion_experiment/configs/residual_v2_smoke.yaml`
- formal output: `outputs/ctm_fusion_experiment/residual_v2_formal`
- smoke output: `outputs/ctm_fusion_experiment/residual_v2_smoke`

This does not overwrite direct CTM or residual v1 outputs.
