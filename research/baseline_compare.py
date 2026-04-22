from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from research.data import build_sample_table, load_research_tables, preprocess_sample_graph, split_sample_table
from research.losses import build_time_bin_edges, cox_ph_loss
from research.metrics import concordance_index
from research.task import get_survival_task_definition


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
    time_bin_edges: np.ndarray | None = None


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


def is_sksurv_available() -> bool:
    return importlib.util.find_spec("sksurv") is not None


def build_tabular_dataframe(config: dict) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    graph_df, clinical_df, metabolite_df, label_df, data_summary = load_research_tables(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
    )

    sample_df = build_sample_table(clinical_df=clinical_df, metabolite_df=metabolite_df, label_df=label_df)
    graph_preprocess = config.get("graph_preprocess", {})
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

    processed_graph_df = pd.concat(
        [
            preprocess_sample_graph(
                sample_graph,
                keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
                min_edge_weight=graph_preprocess.get("min_edge_weight"),
            )
            for _, sample_graph in graph_df.groupby("sample_id")
        ],
        ignore_index=True,
    )

    edge_summary = processed_graph_df.groupby("sample_id").agg(
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
    merged = sample_df.merge(graph_features, on="sample_id", how="inner")

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
    feature_groups["clinical_metabolite"] = feature_groups["clinical"] + feature_groups["metabolite"]
    feature_groups["all_tabular"] = (
        feature_groups["clinical"] + feature_groups["metabolite"] + feature_groups["graph_summary"]
    )

    data_summary["baseline_feature_groups"] = {
        group_name: len(columns) for group_name, columns in feature_groups.items()
    }
    return merged, feature_groups, data_summary


def prepare_split_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    num_time_bins: int,
) -> SplitData:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_columns].to_numpy(dtype=float))
    X_val = scaler.transform(val_df[feature_columns].to_numpy(dtype=float))
    X_test = scaler.transform(test_df[feature_columns].to_numpy(dtype=float))
    time_bin_edges = build_time_bin_edges(
        torch.tensor(train_df["time"].to_numpy(dtype=float), dtype=torch.float32),
        num_bins=num_time_bins,
    ).cpu().numpy()
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
        time_bin_edges=time_bin_edges,
    )


def evaluate_cox_model(model: torch.nn.Module, split: SplitData, which: str = "val") -> dict[str, Any]:
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

    return {
        "loss": float(loss),
        "c_index": float(concordance_index(time_np, event_np, risk_np)),
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
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
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
        optimizer.zero_grad(set_to_none=True)
        risk = model(X_train)
        loss = cox_ph_loss(risk, time_train, event_train)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate_cox_model(model, split, which="val")
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

    val_metrics = evaluate_cox_model(model, split, which="val")
    test_metrics = evaluate_cox_model(model, split, which="test")
    return model, {"best_val_c_index": best_val_c, "history": history, **val_metrics}, test_metrics


def _expand_discrete_hazard_dataset(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    time_bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_bins = len(time_bin_edges)
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for idx in range(len(time)):
        bin_index = int(np.searchsorted(time_bin_edges, time[idx], side="right"))
        bin_index = min(bin_index, num_bins - 1)
        for bin_pos in range(bin_index + 1):
            time_one_hot = np.zeros(num_bins, dtype=float)
            time_one_hot[bin_pos] = 1.0
            X_rows.append(np.concatenate([X[idx], time_one_hot], axis=0))
            y_rows.append(int(event[idx] > 0 and bin_pos == bin_index))

    return np.stack(X_rows), np.asarray(y_rows, dtype=int)


def _predict_discrete_hazard(
    model: LogisticRegression,
    X: np.ndarray,
    time_bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_bins = len(time_bin_edges)
    hazards = np.zeros((len(X), num_bins), dtype=float)
    for bin_pos in range(num_bins):
        time_one_hot = np.zeros((len(X), num_bins), dtype=float)
        time_one_hot[:, bin_pos] = 1.0
        design = np.concatenate([X, time_one_hot], axis=1)
        hazards[:, bin_pos] = model.predict_proba(design)[:, 1]

    survival = np.cumprod(1.0 - np.clip(hazards, 1e-6, 1.0 - 1e-6), axis=1)
    risk = -survival.sum(axis=1)
    return hazards, risk


def _evaluate_discrete_hazard(
    model: LogisticRegression,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    time_bin_edges: np.ndarray,
) -> dict[str, Any]:
    expanded_X, expanded_y = _expand_discrete_hazard_dataset(X, time, event, time_bin_edges)
    predicted_prob = model.predict_proba(expanded_X)[:, 1]
    hazards, risk = _predict_discrete_hazard(model, X, time_bin_edges)
    return {
        "loss": float(log_loss(expanded_y, predicted_prob, labels=[0, 1])),
        "c_index": float(concordance_index(time, event, risk)),
        "risk": risk.tolist(),
        "mean_hazard": float(hazards.mean()),
    }


def train_discrete_hazard_logistic(
    split: SplitData,
    seed: int,
    max_iter: int = 500,
) -> tuple[LogisticRegression, dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    if split.time_bin_edges is None:
        raise ValueError("time_bin_edges must be provided for discrete hazard baseline.")

    train_X, train_y = _expand_discrete_hazard_dataset(
        split.X_train,
        split.time_train,
        split.event_train,
        split.time_bin_edges,
    )

    model = LogisticRegression(
        random_state=seed,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(train_X, train_y)

    val_metrics = _evaluate_discrete_hazard(
        model=model,
        X=split.X_val,
        time=split.time_val,
        event=split.event_val,
        time_bin_edges=split.time_bin_edges,
    )
    test_metrics = _evaluate_discrete_hazard(
        model=model,
        X=split.X_test,
        time=split.time_test,
        event=split.event_test,
        time_bin_edges=split.time_bin_edges,
    )
    return model, {"best_val_c_index": val_metrics["c_index"], **val_metrics}, test_metrics


def get_baseline_specs(feature_groups: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    return {
        "clinical_linear_cox": {
            "features": feature_groups["clinical"],
            "baseline_family": "linear_cox",
        },
        "metabolite_linear_cox": {
            "features": feature_groups["metabolite"],
            "baseline_family": "linear_cox",
        },
        "clinical_metabolite_linear_cox": {
            "features": feature_groups["clinical_metabolite"],
            "baseline_family": "linear_cox",
        },
        "graph_summary_linear_cox": {
            "features": feature_groups["graph_summary"],
            "baseline_family": "linear_cox",
        },
        "all_tabular_linear_cox": {
            "features": feature_groups["all_tabular"],
            "baseline_family": "linear_cox",
        },
        "all_tabular_discrete_hazard_logistic": {
            "features": feature_groups["all_tabular"],
            "baseline_family": "discrete_hazard_logistic",
        },
        "clinical_metabolite_mlp_cox": {
            "features": feature_groups["clinical_metabolite"],
            "baseline_family": "mlp_cox",
        },
        "all_tabular_mlp_cox": {
            "features": feature_groups["all_tabular"],
            "baseline_family": "mlp_cox",
        },
    }


def run_baseline_suite(
    config_path: str,
    seeds: List[int],
    output_root: str = "outputs/current_mainline_v2",
    split_seed: int | None = None,
    only_baselines: List[str] | None = None,
) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df, feature_groups, data_summary = build_tabular_dataframe(config)
    baseline_specs = get_baseline_specs(feature_groups)
    if only_baselines:
        missing = sorted(set(only_baselines).difference(baseline_specs))
        if missing:
            raise ValueError(
                f"Unknown baselines requested in --only: {missing}. "
                f"Available baselines: {sorted(baseline_specs)}"
            )
        baseline_specs = {name: baseline_specs[name] for name in only_baselines}

    all_runs: Dict[str, List[dict[str, Any]]] = {name: [] for name in baseline_specs}

    for seed in seeds:
        effective_split_seed = seed if split_seed is None else split_seed
        train_df, val_df, test_df, split_summary = split_sample_table(
            sample_df=df[["sample_id", "time", "event"] + feature_groups["all_tabular"]].copy(),
            seed=effective_split_seed,
            val_ratio=config["train"]["val_ratio"],
            test_ratio=config["train"]["test_ratio"],
        )

        for baseline_name, spec in baseline_specs.items():
            split = prepare_split_data(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_columns=spec["features"],
                num_time_bins=int(config["train"].get("num_time_bins", 12)),
            )
            if spec["baseline_family"] == "linear_cox":
                _, val_metrics, test_metrics = train_tabular_cox(
                    split=split,
                    model_type="linear",
                    hidden_dim=max(16, int(config["train"].get("hidden_dim", 32))),
                    dropout=float(config["train"].get("dropout", 0.3)),
                    lr=float(config["train"].get("lr", 5e-4)),
                    weight_decay=float(config["train"].get("weight_decay", 1e-3)),
                    epochs=max(40, int(config["train"].get("epochs", 80))),
                    patience=max(8, int(config["train"].get("early_stop_patience", 10))),
                    min_delta=float(config["train"].get("min_delta", 5e-4)),
                    seed=seed,
                )
            elif spec["baseline_family"] == "mlp_cox":
                _, val_metrics, test_metrics = train_tabular_cox(
                    split=split,
                    model_type="mlp",
                    hidden_dim=max(16, int(config["train"].get("hidden_dim", 32))),
                    dropout=float(config["train"].get("dropout", 0.3)),
                    lr=float(config["train"].get("lr", 5e-4)),
                    weight_decay=float(config["train"].get("weight_decay", 1e-3)),
                    epochs=max(40, int(config["train"].get("epochs", 80))),
                    patience=max(8, int(config["train"].get("early_stop_patience", 10))),
                    min_delta=float(config["train"].get("min_delta", 5e-4)),
                    seed=seed,
                )
            elif spec["baseline_family"] == "discrete_hazard_logistic":
                _, val_metrics, test_metrics = train_discrete_hazard_logistic(split=split, seed=seed)
            else:
                raise ValueError(f"Unsupported baseline family: {spec['baseline_family']}")

            all_runs[baseline_name].append(
                {
                    "seed": seed,
                    "num_features": len(spec["features"]),
                    "baseline_family": spec["baseline_family"],
                    "split_seed": effective_split_seed,
                    "best_val_c_index": val_metrics["best_val_c_index"],
                    "test_loss": test_metrics["loss"],
                    "test_c_index": test_metrics["c_index"],
                    "split_summary": split_summary,
                }
            )

    baselines_summary: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []
    for baseline_name, runs in all_runs.items():
        c_indices = [item["test_c_index"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        baseline_summary = {
            "runs": runs,
            "mean_test_c_index": statistics.mean(c_indices),
            "std_test_c_index": statistics.stdev(c_indices) if len(c_indices) > 1 else 0.0,
            "min_test_c_index": min(c_indices),
            "max_test_c_index": max(c_indices),
            "mean_test_loss": statistics.mean(losses),
        }
        baselines_summary[baseline_name] = baseline_summary
        ranking_rows.append(
            {
                "baseline_name": baseline_name,
                "mean_test_c_index": baseline_summary["mean_test_c_index"],
                "std_test_c_index": baseline_summary["std_test_c_index"],
                "mean_test_loss": baseline_summary["mean_test_loss"],
            }
        )

    ranking_rows.sort(key=lambda row: row["mean_test_c_index"], reverse=True)

    summary = {
        "config_path": config_path,
        "output_root": output_root,
        "task_definition": get_survival_task_definition(),
        "data_summary": data_summary,
        "seeds": seeds,
        "split_seed": split_seed,
        "selected_baselines": list(baseline_specs.keys()),
        "baseline_policy": {
            "plain_logistic_regression_replaced": True,
            "replacement_reason": (
                "Current main task is right-censored survival, not plain binary classification. "
                "A discrete-time logistic hazard baseline is used instead of ordinary logistic regression."
            ),
            "random_survival_forest_available": is_sksurv_available(),
            "random_survival_forest_note": (
                "scikit-survival is not installed in the current research environment; "
                "therefore RSF is not used as the mainline conventional non-deep baseline."
            ),
        },
        "baselines": baselines_summary,
        "ranking_table": ranking_rows,
    }

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "baseline_compare_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_config_v2.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 123, 2026])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--output-root", default="outputs/current_mainline_v2")
    args = parser.parse_args()

    summary = run_baseline_suite(
        config_path=args.config,
        seeds=args.seeds,
        output_root=args.output_root,
        split_seed=args.split_seed,
        only_baselines=args.only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
