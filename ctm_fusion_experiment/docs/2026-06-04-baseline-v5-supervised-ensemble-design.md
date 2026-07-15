# Baseline v5 Supervised Ensemble Design

## Purpose

Residual CTM v4 showed that CTM corrections are not stable enough to reliably improve c-index. v5 shifts the c-index effort to the supervised graph encoder and concat-Cox baseline.

## Flow

- Keep one standard fold-local concat baseline as the reference.
- Train additional supervised graph encoder seeds.
- For each graph seed, train multiple concat-Cox baseline seeds with a light pairwise ranking auxiliary loss.
- Standardize each candidate risk on the validation split.
- Select a single model, top-k mean, all-model mean, or validation-softmax ensemble by validation c-index.
- Report selected test c-index, oracle best single-model c-index, and comparable-pair diagnostics.

## Isolation

This flow does not call CTM and does not overwrite previous experiments. Outputs are written under `outputs/ctm_fusion_experiment/baseline_v5_formal`.
