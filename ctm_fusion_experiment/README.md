# CTM Fusion Parallel Experiment

This directory contains an isolated concat-versus-CTM survival-fusion
experiment. It does not modify the current `research/` mainline or the locked
web-facing Cox ensemble.

## Scientific Boundary

Each cross-validation fold trains a graph-only GNN Cox encoder on fold-local
training data, selects it on fold-local validation data, freezes it, and exports
graph embeddings. Baseline and CTM fusion heads then receive the same frozen
graph embeddings, clinical vectors, metabolomics vectors, labels, and split.
The only paired-model difference is the fusion method.

The default tables are the repository's `topology_v6` synthetic/noisy augmented
research data. Report results as method-development evidence, not clinical
validation.

## Layout

```text
ctm_fusion_experiment/
├─ configs/
├─ data/
├─ docs/
├─ models/
├─ tests/
├─ utils/
├─ evaluate.py
└─ train.py
```

## CPU Smoke Test

Run the small one-fold workflow:

```powershell
python -m ctm_fusion_experiment.train `
  --config ctm_fusion_experiment/configs/smoke.yaml `
  --device cpu
```

Artifacts are written to `outputs/ctm_fusion_experiment/smoke/`.

## Formal Five-Fold Experiment

Use a CUDA-enabled Python environment for the full run:

```powershell
python -m ctm_fusion_experiment.train `
  --config ctm_fusion_experiment/configs/ctm_fusion.yaml `
  --device cuda
```

Rebuild the paired summary without retraining:

```powershell
python -m ctm_fusion_experiment.evaluate `
  --output-dir outputs/ctm_fusion_experiment/formal
```

Rebuild figures without retraining:

```powershell
python -m ctm_fusion_experiment.plot_results `
  --output-dir outputs/ctm_fusion_experiment/formal
```

## Residual CTM Variant

The residual variant is independent from the direct CTM replacement flow. It
first trains concat-Cox, freezes it, and then trains CTM as a gated risk
correction:

```text
final_risk = frozen_concat_risk + gate * ctm_delta
```

Run a CPU smoke test:

```powershell
python -m ctm_fusion_experiment.train_residual `
  --config ctm_fusion_experiment/configs/residual_smoke.yaml `
  --device cpu
```

Run the formal five-fold residual experiment:

```powershell
python -m ctm_fusion_experiment.train_residual `
  --config ctm_fusion_experiment/configs/residual_ctm.yaml `
  --device cuda
```

Rebuild residual summaries:

```powershell
python -m ctm_fusion_experiment.evaluate_residual `
  --output-dir outputs/ctm_fusion_experiment/residual_formal
```

Residual outputs are written to `outputs/ctm_fusion_experiment/residual_*` and
do not overwrite `outputs/ctm_fusion_experiment/formal`.

## Residual CTM v2: Validation-Calibrated Alpha

Residual v2 trains CTM delta, then selects a fold-local residual scale on the
validation split:

```text
final_risk = baseline_risk + alpha * ctm_delta
alpha ∈ [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
```

Because `alpha=0.0` is allowed, v2 can safely fall back to the concat baseline
when CTM correction is not useful for a fold.

Smoke test:

```powershell
python -m ctm_fusion_experiment.train_residual_v2 `
  --config ctm_fusion_experiment/configs/residual_v2_smoke.yaml `
  --device cpu
```

Formal five-fold run:

```powershell
python -m ctm_fusion_experiment.train_residual_v2 `
  --config ctm_fusion_experiment/configs/residual_ctm_v2.yaml `
  --device cuda
```

Rebuild v2 summary:

```powershell
python -m ctm_fusion_experiment.evaluate_residual_v2 `
  --output-dir outputs/ctm_fusion_experiment/residual_v2_formal
```

Residual v2 outputs are written to
`outputs/ctm_fusion_experiment/residual_v2_*`.

## Outputs

Each fold writes:

- `graph_encoder.pt`
- `frozen_graph_embeddings.npz`
- `baseline_concat.pt`
- `ctm_fusion.pt`
- `baseline_history.json`
- `ctm_history.json`
- `ctm_analysis.json`
- `test_predictions.csv`
- `fold_summary.json`

The output root writes:

- `comparison_summary.json`
- `fold_comparison.csv`
- `plots/fold_cindex_comparison.png`
- `plots/parameter_comparison.png`
- `plots/training_curves.png`
- `plots/ctm_stable_tick_histogram.png`
- `plots/ctm_modality_attention.png`

`ctm_analysis.json` includes per-sample stable-tick histograms, batch-level
minimum-loss tick records, and mean modality attention per internal tick.
