from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SurvivalTaskDefinition:
    task_name: str = "right_censored_survival_risk_prediction"
    target_type: str = "survival"
    time_column: str = "time"
    event_column: str = "event"
    event_positive_value: int = 1
    event_negative_value: int = 0
    primary_metric: str = "c_index"
    allowed_model_outputs: tuple[str, ...] = ("cox_risk", "discrete_time_hazard")
    label_semantics: str = (
        "time is the observed follow-up duration; "
        "event=1 means observed event; event=0 means right-censored sample."
    )
    incompatible_baselines: tuple[str, ...] = (
        "plain_binary_logistic_regression",
        "plain_regression_on_time_ignoring_censoring",
    )
    conservative_baselines: tuple[str, ...] = (
        "linear_cox",
        "discrete_time_logistic_hazard",
        "tabular_mlp_cox",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_survival_task_definition() -> dict[str, Any]:
    return SurvivalTaskDefinition().to_dict()


def infer_dataset_origin(*paths: str) -> dict[str, Any]:
    joined = " ".join(str(path).lower() for path in paths)
    dataset_version = "unknown"
    if "topology_v6" in joined:
        dataset_version = "topology_v6"
    elif "expanded_v5" in joined:
        dataset_version = "expanded_v5"
    elif "expanded_v4" in joined:
        dataset_version = "expanded_v4"
    elif "expanded_v3" in joined:
        dataset_version = "expanded_v3"

    return {
        "dataset_version": dataset_version,
        "is_synthetic": dataset_version != "unknown",
        "is_augmented": "expanded_" in joined or "topology_v6" in joined,
        "contains_noise": "noisy" in joined or "topology_v6" in joined,
        "is_archived_source": "archive" in joined,
        "source_paths": [str(Path(path).as_posix()) for path in paths],
    }


def load_and_validate_survival_labels(label_csv: str) -> pd.DataFrame:
    label_df = pd.read_csv(label_csv)
    required = {"sample_id", "time", "event"}
    missing = sorted(required.difference(label_df.columns))
    if missing:
        raise ValueError(f"Label table is missing required columns: {missing}")

    if label_df["sample_id"].duplicated().any():
        duplicated = label_df.loc[label_df["sample_id"].duplicated(), "sample_id"].tolist()[:10]
        raise ValueError(f"Duplicate sample_id values found in label table: {duplicated}")

    label_df = label_df.copy()
    label_df["time"] = pd.to_numeric(label_df["time"], errors="raise")
    label_df["event"] = pd.to_numeric(label_df["event"], errors="raise")

    if label_df["time"].isna().any() or label_df["event"].isna().any():
        raise ValueError("Label table contains missing time/event values.")
    if (label_df["time"] <= 0).any():
        raise ValueError("Survival time must be strictly positive.")

    unique_events = sorted(label_df["event"].unique().tolist())
    if any(event_value not in {0, 1} for event_value in unique_events):
        raise ValueError(f"Event column must be binary with values in {{0,1}}. Found: {unique_events}")

    label_df["event"] = label_df["event"].astype(int)
    return label_df


def summarize_survival_labels(label_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "num_samples": int(len(label_df)),
        "num_events": int(label_df["event"].sum()),
        "num_censored": int((1 - label_df["event"]).sum()),
        "event_rate": float(label_df["event"].mean()),
        "time_min": float(label_df["time"].min()),
        "time_median": float(label_df["time"].median()),
        "time_mean": float(label_df["time"].mean()),
        "time_max": float(label_df["time"].max()),
    }
