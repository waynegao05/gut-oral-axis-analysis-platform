# Windows Quickstart for V3 workflow

This guide uses the repository in module mode so that imports work correctly on Windows.

## 1. Install dependencies

```bash
pip install -r requirements-research.txt
```

## 2. Preprocess CSV tables

If your data already matches the repository format, you can skip this step.

```bash
python research/preprocess_tables.py --clinical_csv data/research/sample_clinical_table.csv --metabolite_csv data/research/sample_metabolite_table.csv --graph_csv data/research/sample_graph_table.csv --output_dir outputs/preprocessed
```

## 3. Train with the default research config

```bash
python -m research.train --config research_config.yaml
```

## 4. Predict with the default research config

```bash
python -m research.predict --config research_config.yaml --checkpoint outputs/research/best_model.pt --split test
```

## 5. Explain with the default research config

```bash
python -m research.explain --config research_config.yaml --checkpoint outputs/research/best_model.pt --output outputs/research/explainability.json
```

## 6. Train with the conservative small-sample config

```bash
python -m research.train --config research_config_small_sample.yaml
```

The outputs will be written to:

```text
outputs/research_small_sample/
```

## 7. Predict with the conservative small-sample config

```bash
python -m research.predict --config research_config_small_sample.yaml --checkpoint outputs/research_small_sample/best_model.pt --split test
```

## 8. Explain with the conservative small-sample config

```bash
python -m research.explain --config research_config_small_sample.yaml --checkpoint outputs/research_small_sample/best_model.pt --output outputs/research_small_sample/explainability.json
```

## 9. Repeat runs across multiple seeds

Use the CLI helper below to avoid editing Python files by hand:

```bash
python -m research.repeat_runs_cli --config research_config_small_sample.yaml --seeds 7 21 42 123 2026
```

This writes a summary file to:

```text
outputs/research_small_sample_summary.json
```

## 10. Windows note

If you use `cmd`, do not split commands with `\`. Either keep the command on one line or use `^` as the Windows line continuation character.
