from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from research.losses import cox_ph_loss
from research.metrics import concordance_index


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    time_train: np.ndarray
    time_val: np.ndarray
    time_test: np.ndarray
    event_train: np.ndarray
    event_val: np.ndarray
    event_test: np.ndarray


class LinearCox(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPCox(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        hidden_mid = max(8, hidden_dim // 2)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_mid),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tabular_dataframe(config: dict) -> tuple[pd.DataFrame, dict]:
    graph_df = pd.read_csv(config["paths"]["graph_csv"])
    clinical_df = pd.read_csv(config["paths"]["clinical_csv"])
    metabolite_df = pd.read_csv(config["paths"]["metabolite_csv"])
    label_df = pd.read_csv(config["paths"]["label_csv"])

    node_names = sorted(graph_df["node_name"].drop_duplicates().tolist())

    abundance_pivot = (
        graph_df.drop_duplicates(subset=["sample_id", "node_name"])
        .pivot(index="sample_id", columns="node_name", values="abundance")
        .reindex(columns=node_names)
        .add_prefix("abundance__")
    )
    function_pivot = (
        graph_df.drop_duplicates(subset=["sample_id", "node_name"])
        .pivot(index="sample_id", columns="node_name", values="function_score")
        .reindex(columns=node_names)
        .add_prefix("function__")
    )

    edge_summary = graph_df.groupby("sample_id").agg(
        edge_weight_mean=("edge_weight", "mean"),
        edge_weight_std=("edge_weight", "std"),
        edge_weight_min=("edge_weight", "min"),
        edge_weight_max=("edge_weight", "max"),
        num_edges=("edge_weight", "count"),
        num_nodes=("node_name", pd.Series.nunique),
    )
    edge_summary["edge_weight_std"] = edge_summary["edge_weight_std"].fillna(0.0)
    edge_summary["graph_density"] = edge_summary["num_edges"] / np.maximum(
        edge_summary["num_nodes"] * np.maximum(edge_summary["num_nodes"] - 1, 1),
        1,
    )

    graph_features = (
        abundance_pivot.join(function_pivot, how="outer")
        .join(edge_summary, how="outer")
        .reset_index()
    )

    merged = (
        clinical_df.merge(metabolite_df, on="sample_id")
        .merge(label_df, on="sample_id")
        .merge(graph_features, on="sample_id")
    )

    feature_groups = {
        "clinical": config["model"]["clinical_columns"],
        "metabolite": config["model"]["metabolite_columns"],
        "graph_summary": [
            col
            for col in merged.columns
            if col.startswith("abundance__")
            or col.startswith("function__")
            or col.startswith("edge_weight_")
            or col in {"num_edges", "num_nodes", "graph_density"}
        ],
    }
    feature_groups["clinical_metabolite"] = (
        feature_groups["clinical"] + feature_groups["metabolite"]
    )
    feature_groups["all_tabular"] = (
        feature_groups["clinical"]
        + feature_groups["metabolite"]
        + feature_groups["graph_summary"]
    )
    return merged, feature_groups


def split_dataframe(
    df: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    df = df.iloc[indices].reset_index(drop=True)

    total = len(df)
    test_size = max(1, int(total * test_ratio)) if total >= 3 else 0
    val_size = max(1, int(total * val_ratio)) if total - test_size >= 3 else 0
    train_size = total - val_size - test_size
    if train_size <= 0:
        train_size = max(1, total - 2)
        val_size = 1 if total >= 2 else 0
        test_size = total - train_size - val_size

    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[train_size + val_size :].reset_index(drop=True)
    return train_df, val_df, test_df


def prepare_split_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
) -> SplitData:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_columns].to_numpy(dtype=float))
    X_val = scaler.transform(val_df[feature_columns].to_numpy(dtype=float))
    X_test = scaler.transform(test_df[feature_columns].to_numpy(dtype=float))
    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        time_train=train_df["time"].to_numpy(dtype=float),
        time_val=val_df["time"].to_numpy(dtype=float),
        time_test=test_df["time"].to_numpy(dtype=float),
        event_train=train_df["event"].to_numpy(dtype=float),
        event_val=val_df["event"].to_numpy(dtype=float),
        event_test=test_df["event"].to_numpy(dtype=float),
    )


def evaluate_model(model: torch.nn.Module, split: SplitData, which: str = "val") -> dict:
    model.eval()
    with torch.no_grad():
        if which == "val":
            X = torch.tensor(split.X_val, dtype=torch.float32)
            time = torch.tensor(split.time_val, dtype=torch.float32)
            event = torch.tensor(split.event_val, dtype=torch.float32)
            time_np = split.time_val
            event_np = split.event_val
        else:
            X = torch.tensor(split.X_test, dtype=torch.float32)
            time = torch.tensor(split.time_test, dtype=torch.float32)
            event = torch.tensor(split.event_test, dtype=torch.float32)
            time_np = split.time_test
            event_np = split.event_test

        risk = model(X)
        loss = cox_ph_loss(risk, time, event).item()
        risk_np = risk.detach().cpu().numpy()
        c_index = concordance_index(time_np, event_np, risk_np)

    return {
        "loss": float(loss),
        "c_index": float(c_index),
        "risk": risk_np.tolist(),
    }


def train_tabular_cox(
    split: SplitData,
    model_type: str,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> tuple[torch.nn.Module, dict, dict]:
    set_seed(seed)
    input_dim = split.X_train.shape[1]

    if model_type == "linear":
        model = LinearCox(input_dim)
    elif model_type == "mlp":
        model = MLPCox(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    X_train = torch.tensor(split.X_train, dtype=torch.float32)
    time_train = torch.tensor(split.time_train, dtype=torch.float32)
    event_train = torch.tensor(split.event_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_c = float("-inf")
    patience_count = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        risk = model(X_train)
        loss = cox_ph_loss(risk, time_train, event_train)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate_model(model, split, which="val")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": val_metrics["loss"],
                "val_c_index": val_metrics["c_index"],
            }
        )

        if val_metrics["c_index"] > best_val_c + min_delta:
            best_val_c = val_metrics["c_index"]
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_model(model, split, which="val")
    test_metrics = evaluate_model(model, split, which="test")
    return model, {"best_val_c_index": best_val_c, "history": history, **val_metrics}, test_metrics


def run_baseline_suite(config_path: str, seeds: List[int]) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df, feature_groups = build_tabular_dataframe(config)

    baseline_specs = {
        "clinical_linear_cox": {
            "features": feature_groups["clinical"],
            "model_type": "linear",
        },
        "metabolite_linear_cox": {
            "features": feature_groups["metabolite"],
            "model_type": "linear",
        },
        "clinical_metabolite_linear_cox": {
            "features": feature_groups["clinical_metabolite"],
            "model_type": "linear",
        },
        "graph_summary_linear_cox": {
            "features": feature_groups["graph_summary"],
            "model_type": "linear",
        },
        "all_tabular_linear_cox": {
            "features": feature_groups["all_tabular"],
            "model_type": "linear",
        },
        "clinical_metabolite_mlp_cox": {
            "features": feature_groups["clinical_metabolite"],
            "model_type": "mlp",
        },
        "all_tabular_mlp_cox": {
            "features": feature_groups["all_tabular"],
            "model_type": "mlp",
        },
    }

    all_runs: Dict[str, List[dict]] = {name: [] for name in baseline_specs}

    for seed in seeds:
        train_df, val_df, test_df = split_dataframe(
            df=df,
            seed=seed,
            val_ratio=config["train"]["val_ratio"],
            test_ratio=config["train"]["test_ratio"],
        )

        for baseline_name, spec in baseline_specs.items():
            split = prepare_split_data(train_df, val_df, test_df, spec["features"])
            _, val_metrics, test_metrics = train_tabular_cox(
                split=split,
                model_type=spec["model_type"],
                hidden_dim=max(16, int(config["train"].get("hidden_dim", 32))),
                dropout=float(config["train"].get("dropout", 0.3)),
                lr=float(config["train"].get("lr", 5e-4)),
                weight_decay=float(config["train"].get("weight_decay", 1e-3)),
                epochs=max(40, int(config["train"].get("epochs", 80))),
                patience=max(8, int(config["train"].get("early_stop_patience", 10))),
                min_delta=float(config["train"].get("min_delta", 5e-4)),
                seed=seed,
            )
            all_runs[baseline_name].append(
                {
                    "seed": seed,
                    "num_features": len(spec["features"]),
                    "model_type": spec["model_type"],
                    "best_val_c_index": val_metrics["best_val_c_index"],
                    "test_loss": test_metrics["loss"],
                    "test_c_index": test_metrics["c_index"],
                }
            )

    summary = {"config_path": config_path, "seeds": seeds, "baselines": {}}
    for baseline_name, runs in all_runs.items():
        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        summary["baselines"][baseline_name] = {
            "runs": runs,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_loss": statistics.mean(losses),
        }

    output_dir = Path("outputs/current_mainline")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "baseline_compare_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    args = parser.parse_args()

    summary = run_baseline_suite(config_path=args.config, seeds=args.seeds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()