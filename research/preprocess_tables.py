from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_column(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            series = df[col].astype(float)
            std = float(series.std(ddof=0))
            if std == 0:
                df[col] = series
            else:
                df[col] = (series - float(series.mean())) / std
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_csv", required=True)
    parser.add_argument("--metabolite_csv", required=True)
    parser.add_argument("--graph_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/preprocessed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(args.clinical_csv)
    metabolites = pd.read_csv(args.metabolite_csv)
    graph = pd.read_csv(args.graph_csv)

    clinical_cols = [c for c in clinical.columns if c != "sample_id"]
    metabolite_cols = [c for c in metabolites.columns if c != "sample_id"]
    graph_cols = [c for c in ["abundance", "function_score", "edge_weight"] if c in graph.columns]

    clinical = normalize_column(clinical, clinical_cols)
    metabolites = normalize_column(metabolites, metabolite_cols)
    graph = normalize_column(graph, graph_cols)

    clinical.to_csv(output_dir / "clinical_preprocessed.csv", index=False)
    metabolites.to_csv(output_dir / "metabolite_preprocessed.csv", index=False)
    graph.to_csv(output_dir / "graph_preprocessed.csv", index=False)

    print(f"Saved preprocessed tables to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
