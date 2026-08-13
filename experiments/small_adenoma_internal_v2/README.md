# Internal high-sensitivity small-adenoma model

This experiment is isolated from the web application and all survival-model
mainlines. It trains a dedicated classifier for small adenoma (<10 mm) versus
healthy controls and explicitly targets sensitivity above 64%.

The locked protocol requires that every reported patient is real. Synthetic
minority augmentation is permitted only inside a training partition. Model,
feature, and threshold selection never use the corresponding validation or
outer-test labels.

The target cannot be reported alone. False-positive rate, specificity, ROC AUC,
confidence intervals, and the real-patient numerator/denominator must accompany
the sensitivity result.

## Data preparation

The taxonomy-preparation R source is retained locally and intentionally
excluded from GitHub. Run the benchmark only against its approved, locally
prepared input table.

## Status

The benchmark and final research bundle are produced by `benchmark.py`. They are
not imported by the web application.
