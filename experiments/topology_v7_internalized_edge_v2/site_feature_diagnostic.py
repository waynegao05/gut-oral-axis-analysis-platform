from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from experiments.topology_v7_compositional_temporal_v1.metrics import (
    _evaluate_risk_source,
)
from research.baseline_compare import LinearCox, MLPCox
from research.losses import cox_ph_loss
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    ROOT
    / "outputs/topology_v7_internalized_edge_v2/cohorts/"
    "development_seed20261008"
)
DEFAULT_MODEL_ROOT = (
    ROOT
    / "outputs/topology_v7_internalized_edge_v2/development/"
    "legacy_precomputed_edge_gnn"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/topology_v7_internalized_edge_v2/diagnostics/"
    "site_feature_diagnostic.json"
)
HORIZONS = [36, 60, 84]
INTEGRATION_GRID = [24, 36, 48, 60, 72, 84, 96]


def _closure(values: pd.DataFrame) -> pd.DataFrame:
    adjusted = values.astype(float).clip(lower=0.0) + 1e-5
    return adjusted.div(adjusted.sum(axis=1), axis=0)


def build_site_feature_table(data_dir: Path) -> pd.DataFrame:
    oral_gut = pd.read_csv(
        data_dir / "topology_v7_sample_oral_gut_table.csv"
    )
    clinical = pd.read_csv(
        data_dir / "topology_v7_sample_clinical_table.csv"
    )
    metabolite = pd.read_csv(
        data_dir / "topology_v7_sample_metabolite_table.csv"
    )
    graph = pd.read_csv(
        data_dir / "topology_v7_sample_graph_table.csv"
    )
    labels = pd.read_csv(
        data_dir / "topology_v7_sample_label_table.csv"
    )
    site_columns = [
        "saliva_relative_abundance",
        "stool_relative_abundance",
        "fused_raw_abundance",
        "model_abundance",
    ]
    wide = oral_gut.pivot(
        index="sample_id", columns="taxon", values=site_columns
    )
    wide.columns = [
        f"{measurement}__{taxon}"
        for measurement, taxon in wide.columns
    ]
    taxa = sorted(oral_gut["taxon"].unique().tolist())
    saliva_columns = [
        f"saliva_relative_abundance__{taxon}" for taxon in taxa
    ]
    stool_columns = [
        f"stool_relative_abundance__{taxon}" for taxon in taxa
    ]
    saliva = _closure(wide[saliva_columns])
    stool = _closure(wide[stool_columns])
    log_saliva = np.log(saliva)
    log_stool = np.log(stool)
    clr_saliva = log_saliva.sub(log_saliva.mean(axis=1), axis=0)
    clr_stool = log_stool.sub(log_stool.mean(axis=1), axis=0)
    clr_saliva.columns = [f"clr_saliva__{taxon}" for taxon in taxa]
    clr_stool.columns = [f"clr_stool__{taxon}" for taxon in taxa]
    clr_delta = clr_saliva.to_numpy() - clr_stool.to_numpy()
    delta_frame = pd.DataFrame(
        clr_delta,
        index=wide.index,
        columns=[f"clr_oral_minus_gut__{taxon}" for taxon in taxa],
    )
    absolute_delta = pd.DataFrame(
        np.abs(clr_delta),
        index=wide.index,
        columns=[f"abs_clr_oral_minus_gut__{taxon}" for taxon in taxa],
    )
    midpoint = 0.5 * (
        saliva.to_numpy(dtype=float) + stool.to_numpy(dtype=float)
    )
    js_divergence = 0.5 * np.sum(
        saliva.to_numpy()
        * (
            np.log(saliva.to_numpy())
            - np.log(np.clip(midpoint, 1e-8, None))
        )
        + stool.to_numpy()
        * (
            np.log(stool.to_numpy())
            - np.log(np.clip(midpoint, 1e-8, None))
        ),
        axis=1,
    )
    summaries = pd.DataFrame(
        {
            "oral_gut_js_divergence": js_divergence,
            "oral_entropy": -np.sum(
                saliva.to_numpy() * np.log(saliva.to_numpy()), axis=1
            ),
            "gut_entropy": -np.sum(
                stool.to_numpy() * np.log(stool.to_numpy()), axis=1
            ),
            "oral_gut_clr_distance": np.sqrt(
                np.sum(clr_delta**2, axis=1)
            ),
        },
        index=wide.index,
    )
    function = (
        graph.drop_duplicates(["sample_id", "node_name"])
        .pivot(
            index="sample_id",
            columns="node_name",
            values="function_score",
        )
        .add_prefix("function__")
    )
    features = pd.concat(
        [
            wide,
            clr_saliva,
            clr_stool,
            delta_frame,
            absolute_delta,
            summaries,
            function,
        ],
        axis=1,
    ).reset_index()
    merged = (
        features.merge(clinical, on="sample_id", validate="one_to_one")
        .merge(metabolite, on="sample_id", validate="one_to_one")
        .merge(labels, on="sample_id", validate="one_to_one")
    )
    if not np.isfinite(
        merged.drop(
            columns=["sample_id", "time", "event", "generation_group_id"]
        ).to_numpy(dtype=float)
    ).all():
        raise ValueError("Site feature table contains non-finite values.")
    return merged


def _fit_cox(
    train_x: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    eval_x: np.ndarray,
    eval_time: np.ndarray,
    eval_event: np.ndarray,
    *,
    model_type: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    torch.manual_seed(42)
    if model_type == "linear":
        model = LinearCox(train_x.shape[1])
    elif model_type == "mlp":
        model = MLPCox(
            train_x.shape[1], hidden_dim=48, dropout=0.1
        )
    else:
        raise ValueError(model_type)
    model = model.to(device)
    train_x_tensor = torch.tensor(
        train_x, dtype=torch.float32, device=device
    )
    train_time_tensor = torch.tensor(
        train_time, dtype=torch.float32, device=device
    )
    train_event_tensor = torch.tensor(
        train_event, dtype=torch.float32, device=device
    )
    eval_x_tensor = torch.tensor(
        eval_x, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_c = float("-inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(train_x_tensor)
        loss = cox_ph_loss(
            risk, train_time_tensor, train_event_tensor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_risk = model(eval_x_tensor).cpu().numpy()
        score = concordance_index(eval_time, eval_event, eval_risk)
        if score > best_c + 2e-4:
            best_c = float(score)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                break
    if best_state is None:
        raise RuntimeError("Site Cox diagnostic produced no model.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_risk = model(train_x_tensor).cpu().numpy()
        eval_risk = model(eval_x_tensor).cpu().numpy()
    return train_risk, eval_risk, best_epoch


def _standardize_risk(
    train_risk: np.ndarray, eval_risk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(train_risk))
    scale = max(float(np.std(train_risk)), 1e-8)
    return (train_risk - mean) / scale, (eval_risk - mean) / scale


def run_diagnostic(
    *,
    data_dir: Path,
    model_root: Path,
    output: Path,
    device_arg: str,
) -> dict[str, Any]:
    frame = build_site_feature_table(data_dir)
    excluded = {"sample_id", "time", "event", "generation_group_id"}
    feature_columns = [
        column for column in frame.columns if column not in excluded
    ]
    device = torch.device(
        "cuda"
        if device_arg == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )
    model_types = ["linear", "mlp"]
    alpha_grid = np.linspace(0.0, 1.0, 11)
    fold_rows: list[dict[str, Any]] = []
    for holdout_group in range(5):
        train = frame.loc[
            frame["generation_group_id"] != holdout_group
        ].copy()
        evaluation = frame.loc[
            frame["generation_group_id"] == holdout_group
        ].copy()
        scaler = StandardScaler()
        train_x = scaler.fit_transform(
            train[feature_columns].to_numpy(dtype=float)
        )
        eval_x = scaler.transform(
            evaluation[feature_columns].to_numpy(dtype=float)
        )
        baseline_path = (
            model_root
            / f"holdout_group{holdout_group}"
            / "predictions.npz"
        )
        with np.load(baseline_path, allow_pickle=False) as values:
            baseline = {key: values[key].copy() for key in values.files}
        train_position = {
            sample_id: index
            for index, sample_id in enumerate(
                baseline["train_sample_ids"].astype(str)
            )
        }
        eval_position = {
            sample_id: index
            for index, sample_id in enumerate(
                baseline["eval_sample_ids"].astype(str)
            )
        }
        baseline_train_risk = np.asarray(
            [
                baseline["train_risk"][train_position[sample_id]]
                for sample_id in train["sample_id"].astype(str)
            ]
        )
        baseline_eval_risk = np.asarray(
            [
                baseline["eval_risk"][eval_position[sample_id]]
                for sample_id in evaluation["sample_id"].astype(str)
            ]
        )
        baseline_train_z, baseline_eval_z = _standardize_risk(
            baseline_train_risk, baseline_eval_risk
        )
        for model_type in model_types:
            site_train, site_eval, best_epoch = _fit_cox(
                train_x,
                train["time"].to_numpy(dtype=float),
                train["event"].to_numpy(dtype=float),
                eval_x,
                evaluation["time"].to_numpy(dtype=float),
                evaluation["event"].to_numpy(dtype=float),
                model_type=model_type,
                device=device,
            )
            site_train_z, site_eval_z = _standardize_risk(
                site_train, site_eval
            )
            for alpha in alpha_grid:
                train_risk = (
                    (1.0 - alpha) * baseline_train_z
                    + alpha * site_train_z
                )
                eval_risk = (
                    (1.0 - alpha) * baseline_eval_z
                    + alpha * site_eval_z
                )
                metrics = _evaluate_risk_source(
                    train_time=train["time"].to_numpy(dtype=float),
                    train_event=train["event"].to_numpy(dtype=int),
                    train_risk=train_risk,
                    eval_time=evaluation["time"].to_numpy(dtype=float),
                    eval_event=evaluation["event"].to_numpy(dtype=int),
                    eval_risk=eval_risk,
                    report_horizons=HORIZONS,
                    integration_grid=INTEGRATION_GRID,
                    uno_tau=96.0,
                )
                fold_rows.append(
                    {
                        "holdout_group": holdout_group,
                        "model_type": model_type,
                        "alpha": float(alpha),
                        "best_epoch": best_epoch,
                        "c_index": metrics["harrell_c_index"],
                        "integrated_auc": metrics[
                            "normalized_integrated_auc"
                        ],
                        "integrated_brier": metrics[
                            "normalized_integrated_brier_score"
                        ],
                    }
                )
    candidates: list[dict[str, Any]] = []
    for model_type in model_types:
        for alpha in alpha_grid:
            selected = [
                row
                for row in fold_rows
                if row["model_type"] == model_type
                and row["alpha"] == float(alpha)
            ]
            candidates.append(
                {
                    "model_type": model_type,
                    "alpha": float(alpha),
                    "macro_c_index": statistics.mean(
                        row["c_index"] for row in selected
                    ),
                    "macro_integrated_auc": statistics.mean(
                        row["integrated_auc"] for row in selected
                    ),
                    "macro_integrated_brier": statistics.mean(
                        row["integrated_brier"] for row in selected
                    ),
                    "folds": selected,
                }
            )
    best = max(
        candidates,
        key=lambda row: (
            row["macro_c_index"],
            row["macro_integrated_auc"],
        ),
    )
    report = {
        "schema_version": 1,
        "scope": "post_v2_development_diagnostic_only",
        "data_dir": data_dir.as_posix(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "candidate_grid": candidates,
        "best_candidate": best,
        "may_select_future_fresh_cohort_protocol": True,
        "may_not_claim_independent_validation": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--model-root", type=Path, default=DEFAULT_MODEL_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda"
    )
    args = parser.parse_args()
    report = run_diagnostic(
        data_dir=args.data_dir.resolve(),
        model_root=args.model_root.resolve(),
        output=args.output.resolve(),
        device_arg=args.device,
    )
    print(
        json.dumps(report["best_candidate"], ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
