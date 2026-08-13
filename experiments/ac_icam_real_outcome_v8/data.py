from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = ROOT / "data/public/ac_icam_colon_survival_2023"
RAW_ROOT = PUBLIC_ROOT / "raw"
PROCESSED_ROOT = PUBLIC_ROOT / "processed_v8"
CLINICAL_PATH = RAW_ROOT / "data_clinical_patient.txt"
PANEL_PATH = EXPERIMENT_ROOT / "published_panels.json"
RDATA_PATHS = {
    "tumor": (
        RAW_ROOT
        / "16S_AC-ICAM246_Tumor_OTU_table_and_tax_table.Rdata"
    ),
    "normal": (
        RAW_ROOT
        / "16S_AC-ICAM246_Normal_Colon_OTU_table_and_tax_table.Rdata"
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
valid <- !is.na(genus) & nzchar(genus)
genus_counts <- rowsum(
  otu[valid, , drop=FALSE],
  group=genus[valid],
  reorder=TRUE
)
total <- colSums(otu, na.rm=TRUE)
if (any(total <= 0)) {
  stop("At least one AC-ICAM sample has zero total abundance")
}

relative <- sweep(genus_counts, 2, total, "/")
sample_by_genus <- t(relative)
result <- data.frame(
  sample_id=rownames(sample_by_genus),
  sample_by_genus,
  check.names=FALSE
)
write.csv(result, output_path, row.names=FALSE, quote=TRUE)
"""


@dataclass(frozen=True)
class V8Cohort:
    patients: pd.DataFrame
    tumor: np.ndarray
    normal: np.ndarray
    genera: tuple[str, ...]
    quality_report: dict[str, Any]

    def subset(
        self,
        *,
        endpoint: str,
        scope: str,
    ) -> "V8Cohort":
        endpoint = endpoint.upper()
        if endpoint not in {"PFS", "OS"}:
            raise ValueError("endpoint must be PFS or OS")
        if scope == "all_stage":
            stage_mask = self.patients["stage"].isin([1.0, 2.0, 3.0, 4.0])
        elif scope == "stage_i_iii":
            stage_mask = self.patients["stage"].isin([1.0, 2.0, 3.0])
        else:
            raise ValueError("scope must be all_stage or stage_i_iii")
        time = self.patients[f"{endpoint.lower()}_time"]
        mask = stage_mask & time.notna() & time.gt(0.0)
        indices = np.flatnonzero(mask.to_numpy())
        return V8Cohort(
            patients=self.patients.iloc[indices].reset_index(drop=True),
            tumor=np.asarray(self.tumor[indices], dtype=float),
            normal=np.asarray(self.normal[indices], dtype=float),
            genera=self.genera,
            quality_report={
                **self.quality_report,
                "selected_endpoint": endpoint,
                "selected_scope": scope,
                "selected_patients": int(len(indices)),
                "selected_events": int(
                    self.patients.loc[mask, f"{endpoint.lower()}_event"].sum()
                ),
            },
        )


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


def _patient_number(values: pd.Series, pattern: str) -> pd.Series:
    extracted = values.astype(str).str.extract(pattern, expand=False)
    if extracted.isna().any():
        examples = values.loc[extracted.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Could not parse patient identifiers: {examples}")
    return extracted.astype(int)


def _numeric_stage(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.extract(r"([0-4])", expand=False),
        errors="coerce",
    )


def _event_status(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.startswith("1:").astype(float)


def extract_full_genus_tables(
    *,
    output_dir: Path = PROCESSED_ROOT,
    force: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rscript = _find_rscript()
    outputs: dict[str, Path] = {}
    for tissue, source in RDATA_PATHS.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing AC-ICAM source file: {source}")
        destination = output_dir / f"{tissue}_genus_relative_abundance.csv"
        if destination.exists() and not force:
            outputs[tissue] = destination
            continue
        with tempfile.TemporaryDirectory(prefix="ac_icam_v8_r_") as temporary:
            temporary_root = Path(temporary)
            temporary_source = temporary_root / f"{tissue}.Rdata"
            temporary_output = temporary_root / f"{tissue}.csv"
            shutil.copy2(source, temporary_source)
            subprocess.run(
                [
                    str(rscript),
                    "-",
                    str(temporary_source),
                    str(temporary_output),
                ],
                input=R_EXTRACT_SCRIPT,
                text=True,
                check=True,
                cwd=temporary_root,
            )
            shutil.copy2(temporary_output, destination)
        outputs[tissue] = destination
    return outputs


def _load_genus_table(path: Path, tissue: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate {tissue} sample identifiers.")
    frame["patient_number"] = _patient_number(
        frame["sample_id"], r"^0*(\d+)[TN]$"
    )
    if frame["patient_number"].duplicated().any():
        raise RuntimeError(f"Duplicate {tissue} patient identifiers.")
    abundance_columns = [
        column
        for column in frame.columns
        if column not in {"sample_id", "patient_number"}
    ]
    values = frame[abundance_columns].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise RuntimeError(f"{tissue} genus table has invalid abundances.")
    return frame


def _published_panel_report(
    genera: list[str],
    tumor: np.ndarray,
    normal: np.ndarray,
) -> dict[str, Any]:
    panels = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    genus_index = {genus: index for index, genus in enumerate(genera)}
    mbr = [
        row["genus"]
        for row in panels["mbr_2023"]["features"]
    ]
    mrs_t = [
        row["genus"]
        for row in panels["mrs_16s_2025"]["tumor"]
    ]
    mrs_n = [
        row["genus"]
        for row in panels["mrs_16s_2025"]["normal"]
    ]

    def summarize(names: list[str], values: np.ndarray) -> dict[str, Any]:
        missing = [name for name in names if name not in genus_index]
        present = [name for name in names if name in genus_index]
        prevalence = {
            name: float(np.mean(values[:, genus_index[name]] > 0.0))
            for name in present
        }
        return {
            "declared_features": int(len(names)),
            "matched_features": int(len(present)),
            "missing_features": missing,
            "prevalence": prevalence,
        }

    return {
        "mbr_tumor": summarize(mbr, tumor),
        "mrs_tumor": summarize(mrs_t, tumor),
        "mrs_normal": summarize(mrs_n, normal),
        "candidate_use_allowed": False,
        "reason": (
            "Both published panels used AC-ICAM outcomes during feature "
            "selection. They are retained only as historical references."
        ),
    }


def build_v8_dataset(
    *,
    output_dir: Path = PROCESSED_ROOT,
    force: bool = False,
) -> dict[str, Path]:
    if not CLINICAL_PATH.exists():
        raise FileNotFoundError(f"Missing clinical source: {CLINICAL_PATH}")
    paths = extract_full_genus_tables(output_dir=output_dir, force=force)
    tumor = _load_genus_table(paths["tumor"], "tumor")
    normal = _load_genus_table(paths["normal"], "normal")
    tumor_genera = [
        column
        for column in tumor.columns
        if column not in {"sample_id", "patient_number"}
    ]
    normal_genera = [
        column
        for column in normal.columns
        if column not in {"sample_id", "patient_number"}
    ]
    if tumor_genera != normal_genera:
        raise RuntimeError("Tumor and normal genus columns are not identical.")

    paired = tumor[["sample_id", "patient_number"]].rename(
        columns={"sample_id": "tumor_sample_id"}
    ).merge(
        normal[["sample_id", "patient_number"]].rename(
            columns={"sample_id": "normal_sample_id"}
        ),
        on="patient_number",
        how="inner",
        validate="one_to_one",
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
        raise RuntimeError("Duplicate clinical patient identifiers.")
    merged = paired.merge(
        clinical,
        on="patient_number",
        how="left",
        validate="one_to_one",
    )
    if merged["PATIENT_ID"].isna().any():
        raise RuntimeError("At least one microbiome patient lacks clinical data.")

    patients = pd.DataFrame(
        {
            "patient_id": merged["PATIENT_ID"],
            "patient_number": merged["patient_number"],
            "tumor_sample_id": merged["tumor_sample_id"],
            "normal_sample_id": merged["normal_sample_id"],
            "age": pd.to_numeric(merged["AGE_AT_DX"], errors="coerce"),
            "sex": merged["SEX"],
            "tumor_location": merged["TUMOR_ANATOMIC_LOCATION"],
            "tumor_morphology": merged["TUMOR_MORPHOLOGY"],
            "stage": _numeric_stage(merged["AJCC_PATH_STAGE"]),
            "path_t": _numeric_stage(merged["PATH_TUMOR_STAGE"]),
            "path_n": _numeric_stage(merged["PATH_NODES_STAGE"]),
            "path_m": _numeric_stage(merged["PATH_METASTASIS_STAGE"]),
            "adjuvant_treatment": merged["ADJUVANT_TREATMENT"],
            "adjuvant_any": (
                ~merged["ADJUVANT_TREATMENT"]
                .fillna("No")
                .str.fullmatch("No")
            ).astype(float),
            "pfs_time": pd.to_numeric(merged["PFS_MONTHS"], errors="coerce"),
            "pfs_event": _event_status(merged["PFS_STATUS"]),
            "os_time": pd.to_numeric(merged["OS_MONTHS"], errors="coerce"),
            "os_event": _event_status(merged["OS_STATUS"]),
            "icr_score": pd.to_numeric(merged["ICRSCORE"], errors="coerce"),
            "published_mbr_score": pd.to_numeric(
                merged["MBR_SCORE"], errors="coerce"
            ),
            "published_mbr_group": merged["MBR_GROUP"],
        }
    ).sort_values("patient_number").reset_index(drop=True)

    tumor = tumor.sort_values("patient_number").reset_index(drop=True)
    normal = normal.sort_values("patient_number").reset_index(drop=True)
    if not np.array_equal(
        patients["patient_number"].to_numpy(),
        tumor["patient_number"].to_numpy(),
    ) or not np.array_equal(
        patients["patient_number"].to_numpy(),
        normal["patient_number"].to_numpy(),
    ):
        raise RuntimeError("Patient ordering failed during V8 construction.")

    tumor_values = tumor[tumor_genera].to_numpy(float)
    normal_values = normal[normal_genera].to_numpy(float)
    stage_i_iii = patients["stage"].isin([1.0, 2.0, 3.0])
    quality = {
        "dataset": "ac_icam_real_outcome_v8",
        "patients": int(len(patients)),
        "genera": int(len(tumor_genera)),
        "tumor_nonzero_prevalence": {
            "ge_0.10": int(np.sum(np.mean(tumor_values > 0.0, axis=0) >= 0.10)),
            "ge_0.20": int(np.sum(np.mean(tumor_values > 0.0, axis=0) >= 0.20)),
            "ge_0.50": int(np.sum(np.mean(tumor_values > 0.0, axis=0) >= 0.50)),
        },
        "normal_nonzero_prevalence": {
            "ge_0.10": int(np.sum(np.mean(normal_values > 0.0, axis=0) >= 0.10)),
            "ge_0.20": int(np.sum(np.mean(normal_values > 0.0, axis=0) >= 0.20)),
            "ge_0.50": int(np.sum(np.mean(normal_values > 0.0, axis=0) >= 0.50)),
        },
        "all_stage": {
            "patients": int(len(patients)),
            "pfs_events": int(patients["pfs_event"].sum()),
            "os_events": int(patients["os_event"].sum()),
        },
        "stage_i_iii": {
            "patients": int(stage_i_iii.sum()),
            "pfs_events": int(patients.loc[stage_i_iii, "pfs_event"].sum()),
            "os_events": int(patients.loc[stage_i_iii, "os_event"].sum()),
        },
        "clinical_missingness": {
            column: int(patients[column].isna().sum())
            for column in (
                "age",
                "sex",
                "tumor_location",
                "tumor_morphology",
                "stage",
                "path_t",
                "path_n",
                "path_m",
                "icr_score",
                "published_mbr_score",
                "pfs_time",
                "os_time",
            )
        },
        "published_panels": _published_panel_report(
            tumor_genera,
            tumor_values,
            normal_values,
        ),
        "outcome_leakage_columns_excluded": [
            "VITAL_STATUS",
            "CAUSE_OF_DEATH",
            "LOCAL_RECURRENCE_STATUS",
            "REGIONAL_RECURRENCE_STATUS",
            "DISTANT_RECURRENCE_STATUS",
            "NEW_PRIMARY_TUMOR_IN_FU",
            "MICROSCORE",
        ],
        "claim_boundary": (
            "Tumor and adjacent-normal colon tissue are measured. Saliva, "
            "stool and oral-gut transfer are not represented in this cohort."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    patients_path = output_dir / "patient_outcomes_and_clinical.csv"
    quality_path = output_dir / "quality_report.json"
    patients.to_csv(patients_path, index=False)
    quality_path.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "patients": patients_path,
        "tumor": paths["tumor"],
        "normal": paths["normal"],
        "quality": quality_path,
    }


def load_v8_cohort(
    *,
    processed_dir: Path = PROCESSED_ROOT,
    rebuild: bool = False,
) -> V8Cohort:
    required = {
        "patients": processed_dir / "patient_outcomes_and_clinical.csv",
        "tumor": processed_dir / "tumor_genus_relative_abundance.csv",
        "normal": processed_dir / "normal_genus_relative_abundance.csv",
        "quality": processed_dir / "quality_report.json",
    }
    if rebuild or not all(path.exists() for path in required.values()):
        build_v8_dataset(output_dir=processed_dir, force=rebuild)
    patients = pd.read_csv(required["patients"])
    tumor = _load_genus_table(required["tumor"], "tumor")
    normal = _load_genus_table(required["normal"], "normal")
    genera = tuple(
        column
        for column in tumor.columns
        if column not in {"sample_id", "patient_number"}
    )
    if list(genera) != [
        column
        for column in normal.columns
        if column not in {"sample_id", "patient_number"}
    ]:
        raise RuntimeError("Processed V8 genus columns are inconsistent.")
    tumor = tumor.sort_values("patient_number").reset_index(drop=True)
    normal = normal.sort_values("patient_number").reset_index(drop=True)
    patients = patients.sort_values("patient_number").reset_index(drop=True)
    patient_numbers = patients["patient_number"].to_numpy()
    if not np.array_equal(
        patient_numbers, tumor["patient_number"].to_numpy()
    ) or not np.array_equal(
        patient_numbers, normal["patient_number"].to_numpy()
    ):
        raise RuntimeError("Processed V8 patient alignment is inconsistent.")
    return V8Cohort(
        patients=patients,
        tumor=tumor[list(genera)].to_numpy(float),
        normal=normal[list(genera)].to_numpy(float),
        genera=genera,
        quality_report=json.loads(
            required["quality"].read_text(encoding="utf-8")
        ),
    )


if __name__ == "__main__":
    generated = build_v8_dataset()
    print(
        json.dumps(
            {name: str(path) for name, path in generated.items()},
            indent=2,
        )
    )
