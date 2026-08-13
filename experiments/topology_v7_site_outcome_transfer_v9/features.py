from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.topology_v7_diagnosis.diagnose import (
    GROUP_COLUMN,
    _feature_frame,
)
from experiments.topology_v7_generator_v3.logo_benchmark import FILE_NAMES
from experiments.topology_v7_site_outcome_transfer_v9.public_prior import (
    PANEL_TAXA,
    PublicPrior,
    panel_clr,
    predict_short_survival_risk,
)


ROOT = Path(__file__).resolve().parents[2]
ORAL_GUT_FILE = "topology_v7_sample_oral_gut_table.csv"


def _config_for_data_dir(template_path: Path, data_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(
        yaml.safe_load(template_path.read_text(encoding="utf-8"))
    )
    for key, filename in FILE_NAMES.items():
        config["paths"][key] = str((data_dir / filename).as_posix())
    return config


def _site_frame(data_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / ORAL_GUT_FILE)
    raw["sample_id"] = raw["sample_id"].astype(str)
    raw["taxon"] = raw["taxon"].astype(str).str.lower()
    expected = {taxon.lower() for taxon in PANEL_TAXA}
    missing = sorted(expected.difference(raw["taxon"].unique()))
    if missing:
        raise ValueError(f"Oral-gut table is missing panel taxa: {missing}")

    result = pd.DataFrame(
        {"sample_id": sorted(raw["sample_id"].unique().tolist())}
    ).set_index("sample_id")
    for site, value_column in (
        ("saliva", "saliva_relative_abundance"),
        ("stool", "stool_relative_abundance"),
    ):
        pivot = raw.pivot(
            index="sample_id",
            columns="taxon",
            values=value_column,
        )
        panel_columns = [taxon.lower() for taxon in PANEL_TAXA]
        values = pivot[panel_columns].astype(float).reindex(result.index)
        for taxon in PANEL_TAXA:
            result[f"{site}_raw__{taxon.lower()}"] = values[
                taxon.lower()
            ]
        clr = panel_clr(values.to_numpy(float))
        for index, taxon in enumerate(PANEL_TAXA):
            result[f"{site}_clr__{taxon.lower()}"] = clr[:, index]

    for taxon in PANEL_TAXA:
        key = taxon.lower()
        saliva = result[f"saliva_clr__{key}"]
        stool = result[f"stool_clr__{key}"]
        result[f"site_log_ratio__{key}"] = saliva - stool
        result[f"site_clr_gap__{key}"] = (saliva - stool).abs()
    return result.reset_index()


def build_feature_frame(
    *,
    template_config_path: Path,
    data_dir: Path,
    public_prior: PublicPrior,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    config = _config_for_data_dir(template_config_path, data_dir)
    base, _, metadata = _feature_frame(config)
    site = _site_frame(data_dir)
    frame = base.merge(site, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(base):
        raise RuntimeError("Site-resolved features do not cover every sample.")

    clinical_metabolite = list(config["model"]["clinical_columns"]) + list(
        config["model"]["metabolite_columns"]
    )
    node_columns = [
        column
        for column in frame.columns
        if column.startswith("node_abundance__")
        or column.startswith("node_function__")
    ]
    core = [*clinical_metabolite, *node_columns]
    site_columns = [
        column
        for column in frame.columns
        if column.startswith("saliva_")
        or column.startswith("stool_")
        or column.startswith("site_")
    ]
    stool_panel = frame[
        [f"stool_raw__{taxon.lower()}" for taxon in PANEL_TAXA]
    ].to_numpy(float)
    frame["debelius_short_survival_prior"] = predict_short_survival_risk(
        public_prior, stool_panel
    )

    feature_sets = {
        "core_no_edges": core,
        "site_resolved_no_edges": [*core, *site_columns],
    }
    if public_prior.report["transfer_gate_passed"]:
        feature_sets["site_resolved_public_prior_no_edges"] = [
            *core,
            *site_columns,
            "debelius_short_survival_prior",
        ]

    for name, columns in feature_sets.items():
        if any(
            "edge_weight" in column or column.startswith("edge_")
            for column in columns
        ):
            raise RuntimeError(f"Precomputed edge feature leaked into {name}.")
        if GROUP_COLUMN in columns:
            raise RuntimeError(f"Generation group leaked into {name}.")
        missing_columns = sorted(set(columns).difference(frame.columns))
        if missing_columns:
            raise ValueError(f"{name} is missing features: {missing_columns}")

    feature_metadata = {
        "data_dir": str(data_dir.as_posix()),
        "feature_set_sizes": {
            name: len(columns) for name, columns in feature_sets.items()
        },
        "precomputed_edge_weights_used": False,
        "edge_relationships": "learned internally as nonlinear tree interactions",
        "public_prior": public_prior.report,
        "base_feature_metadata": metadata,
    }
    return frame, feature_sets, feature_metadata
