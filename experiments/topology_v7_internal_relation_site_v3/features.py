from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = {
    "sample_id",
    "taxon",
    "saliva_relative_abundance",
    "stool_relative_abundance",
}
PROHIBITED_FEATURE_TOKENS = {
    "time",
    "event",
    "generation_group_id",
    "survival",
    "risk",
    "censor",
    "provenance",
}


@dataclass(frozen=True)
class SiteFeatureTable:
    frame: pd.DataFrame
    feature_columns: list[str]
    taxa: list[str]


def _closure(values: np.ndarray, pseudocount: float) -> np.ndarray:
    adjusted = np.clip(np.asarray(values, dtype=float), 0.0, None)
    adjusted = adjusted + float(pseudocount)
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)


def build_site_feature_table(
    oral_gut_csv: Path,
    *,
    pseudocount: float = 1e-5,
) -> SiteFeatureTable:
    raw = pd.read_csv(oral_gut_csv)
    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"Oral-gut table is missing columns: {missing}")
    if raw.duplicated(["sample_id", "taxon"]).any():
        raise ValueError("Oral-gut table has duplicate sample_id/taxon rows.")
    for column in (
        "saliva_relative_abundance",
        "stool_relative_abundance",
    ):
        values = pd.to_numeric(raw[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{column} contains non-finite values.")
        if (values < 0.0).any():
            raise ValueError(f"{column} contains negative abundance.")
        raw[column] = values.astype(float)

    taxa = sorted(raw["taxon"].astype(str).unique().tolist())
    sample_ids = sorted(raw["sample_id"].astype(str).unique().tolist())
    expected_rows = len(taxa) * len(sample_ids)
    if len(raw) != expected_rows:
        raise ValueError("Every sample must contain every oral-gut panel taxon.")

    pivot = raw.pivot(
        index="sample_id",
        columns="taxon",
        values=[
            "saliva_relative_abundance",
            "stool_relative_abundance",
        ],
    ).sort_index()
    saliva = _closure(
        pivot["saliva_relative_abundance"][taxa].to_numpy(dtype=float),
        pseudocount,
    )
    stool = _closure(
        pivot["stool_relative_abundance"][taxa].to_numpy(dtype=float),
        pseudocount,
    )
    log_saliva = np.log(saliva)
    log_stool = np.log(stool)
    clr_saliva = log_saliva - log_saliva.mean(axis=1, keepdims=True)
    clr_stool = log_stool - log_stool.mean(axis=1, keepdims=True)
    clr_delta = clr_saliva - clr_stool
    midpoint = 0.5 * (saliva + stool)
    summaries = np.column_stack(
        [
            0.5
            * np.sum(
                saliva * (np.log(saliva) - np.log(midpoint))
                + stool * (np.log(stool) - np.log(midpoint)),
                axis=1,
            ),
            -np.sum(saliva * np.log(saliva), axis=1),
            -np.sum(stool * np.log(stool), axis=1),
            np.sqrt(np.sum(clr_delta**2, axis=1)),
        ]
    )
    blocks = [
        saliva,
        stool,
        clr_saliva,
        clr_stool,
        clr_delta,
        np.abs(clr_delta),
        summaries,
    ]
    feature_columns = (
        [f"oral_abundance__{taxon}" for taxon in taxa]
        + [f"gut_abundance__{taxon}" for taxon in taxa]
        + [f"oral_clr__{taxon}" for taxon in taxa]
        + [f"gut_clr__{taxon}" for taxon in taxa]
        + [f"oral_minus_gut_clr__{taxon}" for taxon in taxa]
        + [f"abs_oral_minus_gut_clr__{taxon}" for taxon in taxa]
        + [
            "oral_gut_js_divergence",
            "oral_entropy",
            "gut_entropy",
            "oral_gut_clr_distance",
        ]
    )
    lowered = " ".join(feature_columns).lower()
    leaked = sorted(
        token for token in PROHIBITED_FEATURE_TOKENS if token in lowered
    )
    if leaked:
        raise ValueError(f"Prohibited site feature tokens: {leaked}")

    matrix = np.column_stack(blocks)
    if matrix.shape[1] != len(feature_columns):
        raise RuntimeError("Site feature names do not align with values.")
    if not np.isfinite(matrix).all():
        raise ValueError("Derived site features contain non-finite values.")
    frame = pd.DataFrame(
        matrix,
        index=pivot.index.astype(str),
        columns=feature_columns,
    )
    frame.index.name = "sample_id"
    return SiteFeatureTable(
        frame=frame,
        feature_columns=feature_columns,
        taxa=taxa,
    )


def fit_site_standardizer(
    table: SiteFeatureTable,
    train_sample_ids: Iterable[str],
) -> dict[str, Any]:
    identifiers = [str(value) for value in train_sample_ids]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Training sample IDs are not unique.")
    missing = sorted(set(identifiers).difference(table.frame.index))
    if missing:
        raise ValueError(f"Site features are missing training IDs: {missing[:5]}")
    values = table.frame.loc[identifiers].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    zero_variance = scale <= 1e-12
    scale = np.where(zero_variance, 1.0, scale)
    return {
        "fit_scope": "outer_training_groups_only",
        "feature_columns": list(table.feature_columns),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "zero_variance": zero_variance.tolist(),
        "num_fit_samples": len(identifiers),
    }


def attach_site_features(
    data_sets: Sequence[Sequence[Any]],
    table: SiteFeatureTable,
    standardizer: dict[str, Any],
) -> None:
    if standardizer["feature_columns"] != table.feature_columns:
        raise ValueError("Site standardizer columns do not match the table.")
    mean = np.asarray(standardizer["mean"], dtype=float)
    scale = np.asarray(standardizer["scale"], dtype=float)
    seen: set[str] = set()
    for data_set in data_sets:
        for item in data_set:
            sample_id = str(item.sample_id)
            if sample_id in seen:
                raise ValueError("A sample occurs in more than one data split.")
            seen.add(sample_id)
            if sample_id not in table.frame.index:
                raise ValueError(f"Site features are missing sample {sample_id}.")
            raw = table.frame.loc[sample_id].to_numpy(dtype=float)
            item.site_features = torch.tensor(
                (raw - mean) / scale,
                dtype=torch.float32,
            )
