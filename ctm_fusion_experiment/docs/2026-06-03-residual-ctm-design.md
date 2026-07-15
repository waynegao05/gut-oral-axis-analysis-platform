# Residual CTM Fusion Design

## Purpose

The direct CTM replacement experiment under `ctm_fusion_experiment/train.py`
showed lower five-fold c-index than concat-Cox. This follow-up keeps that
original flow intact and adds a separate residual flow where CTM can only learn
a gated correction on top of the trained concat-Cox baseline.

## Isolation

New entry points and outputs are independent:

- training: `python -m ctm_fusion_experiment.train_residual`
- evaluation: `python -m ctm_fusion_experiment.evaluate_residual`
- formal config: `ctm_fusion_experiment/configs/residual_ctm.yaml`
- smoke config: `ctm_fusion_experiment/configs/residual_smoke.yaml`
- formal output: `outputs/ctm_fusion_experiment/residual_formal`
- smoke output: `outputs/ctm_fusion_experiment/residual_smoke`

The original direct CTM files and outputs remain usable:

- `ctm_fusion_experiment/train.py`
- `ctm_fusion_experiment/configs/ctm_fusion.yaml`
- `outputs/ctm_fusion_experiment/formal`

## Model

For each fold:

1. Train the same fold-local graph-only encoder and freeze graph embeddings.
2. Train the same concat-Cox baseline.
3. Freeze concat-Cox.
4. Train Residual CTM:

```text
baseline_risk = frozen_concat_cox(graph, clinical, metabolomics)
ctm_delta_per_tick = CTM(graph_token, clinical_token, metabolomics_token)
final_risk_per_tick = baseline_risk + sigmoid(gate) * ctm_delta_per_tick
```

The residual head is initialized to zero, so the model starts exactly at the
baseline prediction. This makes the first question "can CTM improve the
baseline?" rather than "can CTM rediscover the baseline?"

## Training Objective

Residual CTM uses:

- all-tick Cox deep supervision
- final-tick Cox loss
- best-tick Cox loss
- z-score distillation against frozen baseline risk
- a small anti-collapse risk-separation term

The direct "stable tick" objective is not used as the primary training target
because the first CTM run showed that stability can reward low-variance output.
