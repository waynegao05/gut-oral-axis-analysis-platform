from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.topology_v7_site_outcome_transfer_v9.public_prior import (
    PANEL_TAXA,
    panel_clr,
)


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = ROOT / "data/public/ac_icam_colon_survival_2023"
RAW_ROOT = PUBLIC_ROOT / "raw"
PROCESSED_ROOT = PUBLIC_ROOT / "processed"
CLINICAL_PATH = RAW_ROOT / "data_clinical_patient.txt"
RDATA_PATHS = {
    "normal": (
        RAW_ROOT
        / "16S_AC-ICAM246_Normal_Colon_OTU_table_and_tax_table.Rdata"
    ),
    "tumor": (
        RAW_ROOT
        / "16S_AC-ICAM246_Tumor_OTU_table_and_tax_table.Rdata"
    ),
}

R_EXTRACT_SCRIPT = r"""
args <- commandArgs(trailingOnly=TRUE)
input_path <- args[[1]]
output_path <- args[[2]]
load(input_path)

otu_names <- ls(pattern="^OTU_table_dat_246")
if (length(otu_names) != 1 || !("tax_table_dat" %in% ls())) {
  stop("AC-ICAM RData does not contain the expected matrices")
}

otu <- get(otu_names[[1]])
taxonomy <- tax_table_dat
if (!identical(rownames(otu), rownames(taxonomy))) {
  stop("OTU and taxonomy rows are not aligned")
}

genus <- sub("^D_5__", "", as.character(taxonomy[, "Genus"]))
selectors <- list(
  Fusobacterium = genus == "Fusobacterium",
  Porphyromonas = genus == "Porphyromonas",
  Prevotella = grepl("^Prevotella( [0-9]+)?$", genus),
  Streptococcus = genus == "Streptococcus",
  Lactobacillus = genus == "Lactobacillus"
)

total <- colSums(otu, na.rm=TRUE)
if (any(total <= 0)) {
  stop("At least one AC-ICAM sample has zero total abundance")
}

result <- data.frame(sample_id=colnames(otu), check.names=FALSE)
for (taxon in names(selectors)) {
  selected <- selectors[[taxon]]
  if (!any(selected)) {
    stop(paste("Missing genus", taxon))
  }
  result[[taxon]] <- colSums(
    otu[selected, , drop=FALSE],
    na.rm=TRUE
  ) / total
}
write.csv(result, output_path, row.names=FALSE, quote=FALSE)
"""


def _find_rscript() -> Path:
    executable = shutil.which("Rscript")
    if executable:
        return Path(executable)
    candidates = sorted(
        Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "Rscript is required to read the published AC-ICAM RData files."
    )


def extract_genus_panels(
    output_dir: Path = PROCESSED_ROOT,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rscript = _find_rscript()
    outputs: dict[str, Path] = {}
    for tissue, input_path in RDATA_PATHS.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing AC-ICAM source file: {input_path}")
        output_path = output_dir / f"{tissue}_shared_genus_panel.csv"
        with tempfile.TemporaryDirectory(prefix="ac_icam_r_") as temporary:
            temporary_root = Path(temporary)
            temporary_input = temporary_root / f"{tissue}.Rdata"
            temporary_output = temporary_root / f"{tissue}.csv"
            shutil.copy2(input_path, temporary_input)
            subprocess.run(
                [
                    str(rscript),
                    "-",
                    str(temporary_input),
                    str(temporary_output),
                ],
                input=R_EXTRACT_SCRIPT,
                text=True,
                check=True,
                cwd=temporary_root,
            )
            shutil.copy2(temporary_output, output_path)
        outputs[tissue] = output_path
    return outputs


def _patient_number(values: pd.Series, pattern: str) -> pd.Series:
    extracted = values.astype(str).str.extract(pattern, expand=False)
    if extracted.isna().any():
        examples = values.loc[extracted.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Could not parse patient identifiers: {examples}")
    return extracted.astype(int)


def _stage_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.extract(r"([1-4])", expand=False),
        errors="coerce",
    )


def _add_panel_features(
    frame: pd.DataFrame,
    tissue: str,
) -> list[str]:
    raw_columns = [f"{tissue}_raw__{taxon.lower()}" for taxon in PANEL_TAXA]
    values = frame[raw_columns].astype(float).to_numpy()
    clr_values = panel_clr(values)
    clr_columns: list[str] = []
    for index, taxon in enumerate(PANEL_TAXA):
        column = f"{tissue}_clr__{taxon.lower()}"
        frame[column] = clr_values[:, index]
        clr_columns.append(column)
    frame[f"{tissue}_log_panel_load"] = np.log(
        np.clip(values.sum(axis=1), 1e-12, None)
    )
    return clr_columns


def load_ac_icam_cohort(
    *,
    output_dir: Path = PROCESSED_ROOT,
    endpoint: str = "PFS",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    endpoint = endpoint.upper()
    if endpoint not in {"PFS", "OS"}:
        raise ValueError("endpoint must be PFS or OS")
    if not CLINICAL_PATH.exists():
        raise FileNotFoundError(f"Missing AC-ICAM clinical data: {CLINICAL_PATH}")

    panel_paths = extract_genus_panels(output_dir)
    panels: dict[str, pd.DataFrame] = {}
    for tissue, path in panel_paths.items():
        panel = pd.read_csv(path)
        panel["patient_number"] = _patient_number(
            panel["sample_id"], r"^0*(\d+)[TN]$"
        )
        if panel["patient_number"].duplicated().any():
            raise RuntimeError(f"Duplicate {tissue} patient identifiers.")
        rename = {
            taxon: f"{tissue}_raw__{taxon.lower()}"
            for taxon in PANEL_TAXA
        }
        panels[tissue] = panel.rename(columns=rename).drop(
            columns=["sample_id"]
        )

    clinical = pd.read_csv(
        CLINICAL_PATH,
        sep="\t",
        skiprows=4,
        dtype=str,
    )
    clinical["patient_number"] = _patient_number(
        clinical["PATIENT_ID"], r"P0*(\d+)$"
    )
    if clinical["patient_number"].duplicated().any():
        raise RuntimeError("Duplicate AC-ICAM clinical patient identifiers.")

    frame = clinical.merge(
        panels["tumor"],
        on="patient_number",
        how="inner",
        validate="one_to_one",
    ).merge(
        panels["normal"],
        on="patient_number",
        how="inner",
        validate="one_to_one",
    )
    frame["stage"] = _stage_number(frame["AJCC_PATH_STAGE"])
    frame["time"] = pd.to_numeric(
        frame[f"{endpoint}_MONTHS"], errors="coerce"
    )
    frame["event"] = (
        frame[f"{endpoint}_STATUS"].astype(str).str.startswith("1:")
    ).astype(float)
    frame["age"] = pd.to_numeric(frame["AGE_AT_DX"], errors="coerce")
    frame["mbr_score"] = pd.to_numeric(frame["MBR_SCORE"], errors="coerce")

    before_filter = len(frame)
    frame = frame.loc[
        frame["stage"].isin([1.0, 2.0, 3.0])
        & frame["time"].notna()
        & frame["time"].gt(0.0)
    ].copy()
    frame = frame.sort_values("patient_number").reset_index(drop=True)

    feature_columns: dict[str, list[str]] = {}
    for tissue in ("tumor", "normal"):
        clr_columns = _add_panel_features(frame, tissue)
        feature_columns[f"{tissue}_clr"] = clr_columns
        feature_columns[f"{tissue}_clr_load"] = [
            *clr_columns,
            f"{tissue}_log_panel_load",
        ]

    report = {
        "dataset": "ac_icam_colon_survival_2023",
        "endpoint": endpoint,
        "raw_paired_patients": int(before_filter),
        "stage_i_to_iii_complete_patients": int(len(frame)),
        "events": int(frame["event"].sum()),
        "censored": int(len(frame) - frame["event"].sum()),
        "feature_columns": feature_columns,
        "panel_taxa": list(PANEL_TAXA),
        "patient_id_mapping_complete": bool(before_filter == 246),
        "claim_boundary": (
            "AC-ICAM profiles tumor and adjacent normal colon tissue, not "
            "saliva or stool. Any V7 use is an auxiliary frozen prior."
        ),
    }
    return frame, report
