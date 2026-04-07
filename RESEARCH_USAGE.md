# Research Usage

## Install research dependencies

```bash
pip install -r requirements-research.txt
```

## Preprocess your tables

```bash
python research/preprocess_tables.py \
  --clinical_csv your_clinical.csv \
  --metabolite_csv your_metabolite.csv \
  --graph_csv your_graph.csv \
  --output_dir outputs/preprocessed
```

## Train the formal GAT-Cox model

```bash
python research/train.py --config research_config.yaml
```

## Run inference

```bash
python research/predict.py \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --split test
```

## Run explainability analysis

```bash
python research/explain.py \
  --config research_config.yaml \
  --checkpoint outputs/research/best_model.pt \
  --output outputs/research/explainability.json
```

## Current scope

This branch of the repository is formal research code for training and inference. It is suitable for experiments, ablations, and manuscript preparation. It is not yet a validated clinical deployment package and still requires real cohort data, external validation, and calibrated reporting.
