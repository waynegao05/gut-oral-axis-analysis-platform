from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = (
    ROOT
    / "data/public/debelius_crc_survival_2023/extracted"
    / "ctmrbio-crc-survival-6ab25c3/ipynb/data"
)
METADATA_PATH = PUBLIC_ROOT / "paired_metadata.tsv"
TABLE_PATH = PUBLIC_ROOT / "tables/table.tsv"
TAXONOMY_PATH = PUBLIC_ROOT / "tables/taxonomy.txt"
SOURCE_URL = "https://zenodo.org/records/7690117"
PANEL_TAXA = (
    "Fusobacterium",
    "Porphyromonas",
    "Prevotella",
    "Streptococcus",
    "Lactobacillus",
)
CV_SEEDS = (7, 21, 42, 123, 2026)
MINIMUM_TRANSFER_AUC = 0.60


@dataclass(frozen=True)
class PublicPrior:
    model: Pipeline
    report: dict[str, Any]


def panel_clr(values: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(PANEL_TAXA):
        raise ValueError(
            f"Expected a two-dimensional {len(PANEL_TAXA)}-taxon matrix."
        )
    matrix = np.clip(matrix, 0.0, None)
    closed = (matrix + pseudocount) / (
        matrix.sum(axis=1, keepdims=True) + pseudocount * matrix.shape[1]
    )
    logged = np.log(closed)
    return logged - logged.mean(axis=1, keepdims=True)


def _extract_genus(taxonomy: pd.Series) -> pd.Series:
    return taxonomy.astype(str).str.extract(
        r"(?:^|;)g__([^;]+)", expand=False
    )


def load_debelius_tumour_panel() -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(METADATA_PATH, sep="\t", dtype=str)
    metadata["sample-id"] = metadata["sample-id"].astype(str)
    metadata = metadata.loc[
        metadata["tissue_type"].str.lower().eq("tumour tissue")
        & metadata["long_survival"].isin(["True", "False"])
    ].copy()

    taxonomy = pd.read_csv(TAXONOMY_PATH, sep="\t", dtype=str)
    taxonomy["genus"] = _extract_genus(taxonomy["Taxon"])
    taxonomy["Feature ID"] = taxonomy["Feature ID"].astype(str)
    table = pd.read_csv(TABLE_PATH, sep="\t", index_col=0)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    total_counts = table.sum(axis=0).astype(float)

    panel = pd.DataFrame(index=table.columns)
    for taxon in PANEL_TAXA:
        feature_ids = taxonomy.loc[
            taxonomy["genus"].str.lower().eq(taxon.lower()), "Feature ID"
        ]
        aligned = table.index.intersection(feature_ids.astype(str))
        if len(aligned) == 0:
            raise ValueError(f"Debelius table contains no genus {taxon}.")
        panel[taxon] = table.loc[aligned].sum(axis=0).astype(float)
    panel = panel.div(total_counts.replace(0.0, np.nan), axis=0).fillna(0.0)
    panel.index.name = "sample-id"
    panel = panel.reset_index()

    merged = metadata[
        ["sample-id", "host_subject_id", "long_survival"]
    ].merge(panel, on="sample-id", how="inner", validate="one_to_one")
    if merged["host_subject_id"].duplicated().any():
        raise RuntimeError("Expected one tumour sample per Debelius subject.")
    merged = merged.sort_values("host_subject_id").reset_index(drop=True)
    target = merged["long_survival"].eq("True").astype(int).to_numpy()
    return merged, target


def fit_public_prior() -> PublicPrior:
    cohort, target = load_debelius_tumour_panel()
    values = panel_clr(cohort[list(PANEL_TAXA)].to_numpy(float))
    aucs: list[float] = []
    for seed in CV_SEEDS:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            ),
        )
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probability = cross_val_predict(
            model,
            values,
            target,
            cv=folds,
            method="predict_proba",
        )[:, 1]
        aucs.append(float(roc_auc_score(target, probability)))

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=CV_SEEDS[0],
        ),
    )
    final_model.fit(values, target)
    mean_auc = float(np.mean(aucs))
    report = {
        "dataset": "debelius_crc_survival_2023",
        "source_url": SOURCE_URL,
        "endpoint": "long_survival fixed-horizon contrast",
        "num_tumour_subjects": int(len(cohort)),
        "num_long_survival": int(target.sum()),
        "num_short_survival": int((1 - target).sum()),
        "panel_taxa": list(PANEL_TAXA),
        "cv_seeds": list(CV_SEEDS),
        "repeated_cv_auc_values": aucs,
        "mean_repeated_cv_auc": mean_auc,
        "std_repeated_cv_auc": float(np.std(aucs, ddof=1)),
        "minimum_transfer_auc": MINIMUM_TRANSFER_AUC,
        "transfer_gate_passed": bool(mean_auc >= MINIMUM_TRANSFER_AUC),
        "risk_direction": "negative logit of long survival",
        "claim_boundary": (
            "Tumour tissue and fixed-horizon labels differ from the V7 "
            "oral-gut right-censored proxy; this is an auxiliary prior only."
        ),
    }
    return PublicPrior(model=final_model, report=report)


def predict_short_survival_risk(
    prior: PublicPrior,
    panel_relative_abundance: np.ndarray,
) -> np.ndarray:
    values = panel_clr(panel_relative_abundance)
    long_survival_margin = prior.model.decision_function(values)
    return -np.asarray(long_survival_margin, dtype=float)
