# Small-sample training notes

This note explains which configuration items are most important when the current dataset is small and the evaluation is unstable across random seeds.

## Where to change

Main file to edit:

- `research_config.yaml`

A more conservative preset has also been added:

- `research_config_small_sample.yaml`

## Why the original setup is unstable

With 120 samples and `test_ratio: 0.1`, the test split contains only about 12 samples. For survival analysis, this can produce a highly variable C-index because the number of comparable pairs is small and random split differences become amplified.

At the same time, the original model capacity is relatively strong for a small synthetic cohort:

- `hidden_dim: 64`
- `heads: 4`
- `dropout: 0.2`

This can lead to optimistic validation results under some splits and weak generalization under others.

## Conservative adjustments

The new preset uses the following changes:

- increase `test_ratio` from `0.1` to `0.2`
- keep `val_ratio` at `0.2`
- reduce `hidden_dim` from `64` to `32`
- reduce `heads` from `4` to `2`
- increase `dropout` from `0.2` to `0.35`
- increase `weight_decay` from `0.0001` to `0.001`
- reduce `lr` from `0.001` to `0.0005`
- reduce `batch_size` from `16` to `8`
- reduce `early_stop_patience` from `15` to `10`
- increase `min_delta` from `0.0001` to `0.0005`

## How to run

```bash
python -m research.train --config research_config_small_sample.yaml
```

## Interpretation reminder

This setting is intended to reduce variance and overfitting risk under small-sample conditions. It does not replace repeated experiments, repeated random splits, or cross-validation.
