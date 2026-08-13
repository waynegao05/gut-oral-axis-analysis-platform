from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIVE_GENERA = (
    "Fusobacterium",
    "Lactobacillus",
    "Porphyromonas",
    "Prevotella",
    "Streptococcus",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_identifier(value: Any) -> str:
    return str(value).strip().strip("'").strip('"')


def _relative_genus_abundance(
    counts: pd.DataFrame,
    feature_to_genus: dict[str, str],
) -> pd.DataFrame:
    numeric = counts.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Microbiome counts must be finite and non-negative.")
    totals = numeric.sum(axis=1).replace(0.0, np.nan)
    output = pd.DataFrame(index=numeric.index)
    for genus in FIVE_GENERA:
        columns = [
            feature
            for feature in numeric.columns
            if feature_to_genus.get(str(feature)) == genus
        ]
        numerator = (
            numeric[columns].sum(axis=1)
            if columns
            else pd.Series(0.0, index=numeric.index)
        )
        output[f"abundance_{genus}"] = (
            numerator / totals
        ).fillna(0.0)
    return output


def _alexander_features(raw_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    workbooks = [
        ("CRC", raw_dir / "CRC_microbiome_data_anon.xlsx"),
        ("KRCA", raw_dir / "KRCA_microbiome_data_anon.xlsx"),
    ]
    for cohort, path in workbooks:
        covariates = pd.read_excel(path, sheet_name="covariates")
        microbiome = pd.read_excel(path, sheet_name="microbiome_data")
        taxonomy = pd.read_excel(path, sheet_name="microbiome_labels")
        covariates["sample_id"] = covariates.iloc[:, 0].map(
            _clean_identifier
        )
        microbiome["sample_id"] = microbiome.iloc[:, 0].map(
            _clean_identifier
        )
        feature_to_genus = {
            str(otu): str(genus)
            for otu, genus in taxonomy[["OTU", "genus"]].itertuples(
                index=False, name=None
            )
            if pd.notna(genus)
        }
        count_columns = [
            column
            for column in microbiome.columns
            if str(column) in feature_to_genus
        ]
        abundance = _relative_genus_abundance(
            microbiome.set_index("sample_id")[count_columns],
            feature_to_genus,
        ).reset_index()
        selected_covariates = covariates.rename(
            columns={covariates.columns[0]: "source_sample_id"}
        )
        selected_covariates = selected_covariates.drop(
            columns=["source_sample_id"], errors="ignore"
        )
        merged = selected_covariates.merge(
            abundance, on="sample_id", how="inner", validate="one_to_one"
        )
        merged.insert(0, "cohort", cohort)
        merged["sample_id"] = cohort + "_" + merged["sample_id"]
        rows.append(merged)
    result = pd.concat(rows, ignore_index=True)
    if result["sample_id"].duplicated().any():
        raise ValueError("Alexander sample identifiers are not unique.")
    return result


def _taxonomy_genus(taxon: Any) -> str | None:
    matches = re.findall(r"(?:^|;)g__([^;]+)", str(taxon))
    for match in matches:
        value = match.strip()
        if value and value not in {"__", "uncultured", "unclassified"}:
            return value.split("/")[0]
    return None


def _debelius_features(extracted_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dirs = sorted(extracted_root.glob("*/ipynb/data"))
    if len(data_dirs) != 1:
        raise RuntimeError("Expected one extracted Debelius data directory.")
    data_dir = data_dirs[0]
    metadata = pd.read_csv(
        data_dir / "paired_metadata.tsv", sep="\t", dtype=str
    )
    count_table = pd.read_csv(
        data_dir / "tables/table.tsv", sep="\t", index_col=0
    ).transpose()
    taxonomy = pd.read_csv(data_dir / "tables/taxonomy.txt", sep="\t")
    feature_to_genus = {
        str(feature): genus
        for feature, taxon in taxonomy[
            ["Feature ID", "Taxon"]
        ].itertuples(index=False, name=None)
        if (genus := _taxonomy_genus(taxon)) is not None
    }
    abundance = _relative_genus_abundance(
        count_table,
        feature_to_genus,
    ).reset_index(names="sample-id")
    metadata["sample-id"] = metadata["sample-id"].map(_clean_identifier)
    abundance["sample-id"] = abundance["sample-id"].map(_clean_identifier)
    sample_table = metadata.merge(
        abundance, on="sample-id", how="inner", validate="one_to_one"
    )
    sample_table = sample_table.rename(columns={"sample-id": "sample_id"})
    keep_metadata = [
        "sample_id",
        "host_subject_id",
        "age_cat",
        "sex",
        "ana_location",
        "differentiation_grade",
        "asa_cat",
        "long_survival",
        "tissue_type",
        "stage_tnm",
        "type_of_treatment_preop",
        "radical_surgery",
        "radical_micro",
    ]
    abundance_columns = [
        f"abundance_{genus}" for genus in FIVE_GENERA
    ]
    sample_table = sample_table[keep_metadata + abundance_columns].copy()
    sample_table["long_survival"] = (
        sample_table["long_survival"].str.lower().map(
            {"true": 1, "false": 0}
        )
    )
    if sample_table["long_survival"].isna().any():
        raise ValueError("Debelius long_survival contains unknown values.")

    paired = sample_table.pivot(
        index="host_subject_id",
        columns="tissue_type",
        values=abundance_columns,
    )
    required_tissues = {"tumour tissue", "normal tissue"}
    if not required_tissues.issubset(
        set(paired.columns.get_level_values(1))
    ):
        raise ValueError("Debelius paired tissue columns are incomplete.")
    pair_table = pd.DataFrame(index=paired.index)
    for column in abundance_columns:
        pair_table[f"tumour_{column}"] = paired[
            (column, "tumour tissue")
        ]
        pair_table[f"normal_{column}"] = paired[
            (column, "normal tissue")
        ]
        pair_table[f"delta_{column}"] = (
            pair_table[f"tumour_{column}"]
            - pair_table[f"normal_{column}"]
        )
    labels = sample_table.groupby("host_subject_id")[
        "long_survival"
    ].nunique()
    if (labels != 1).any():
        raise ValueError("Debelius paired samples disagree on survival group.")
    pair_table["long_survival"] = sample_table.groupby(
        "host_subject_id"
    )["long_survival"].first()
    pair_table = pair_table.reset_index()
    return sample_table, pair_table


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _uchida_sample_metadata(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "DRA012322.sample.xml"
    root = ET.parse(path).getroot()
    rows: list[dict[str, str]] = []
    for sample in root.iter():
        if _xml_local_name(sample.tag) != "SAMPLE":
            continue
        row: dict[str, str] = {
            "sample_accession": str(sample.attrib.get("accession", ""))
        }
        for element in sample.iter():
            name = _xml_local_name(element.tag)
            text = (element.text or "").strip()
            if name in {"TITLE", "SCIENTIFIC_NAME", "SAMPLE_NAME"} and text:
                row[name.lower()] = text
            if name == "SAMPLE_ATTRIBUTE":
                tag_value = ""
                value = ""
                for child in element:
                    child_name = _xml_local_name(child.tag)
                    if child_name == "TAG":
                        tag_value = (child.text or "").strip()
                    elif child_name == "VALUE":
                        value = (child.text or "").strip()
                if tag_value:
                    row[tag_value] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or frame["sample_accession"].duplicated().any():
        raise ValueError("Uchida XML did not yield unique sample accessions.")
    return frame


def _write_frame(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "sha256": _sha256(path),
    }


def prepare_all() -> dict[str, Any]:
    output_root = (
        ROOT
        / "data/public/processed_topology_v7_compositional_temporal_v1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Any] = {}

    alexander = _alexander_features(
        ROOT / "data/public/alexander_crc_multiomics_2023/raw"
    )
    generated["alexander_sample_features"] = _write_frame(
        alexander, output_root / "alexander_sample_features.csv"
    )

    debelius_sample, debelius_pair = _debelius_features(
        ROOT / "data/public/debelius_crc_survival_2023/extracted"
    )
    generated["debelius_sample_features"] = _write_frame(
        debelius_sample, output_root / "debelius_sample_features.csv"
    )
    generated["debelius_patient_pair_features"] = _write_frame(
        debelius_pair,
        output_root / "debelius_patient_pair_features.csv",
    )

    uchida = _uchida_sample_metadata(
        ROOT / "data/public/uchida_crc_paired_oral_gut_2021/raw"
    )
    generated["uchida_sample_metadata"] = _write_frame(
        uchida, output_root / "uchida_sample_metadata.csv"
    )

    report = {
        "schema_version": 1,
        "scope": "observed_public_data_only",
        "five_genera": list(FIVE_GENERA),
        "generated": generated,
        "outcome_policy": {
            "right_censored_survival_inferred": False,
            "alexander_outcome_use": "none",
            "debelius_outcome_use": "binary_long_survival_only",
            "uchida_outcome_use": "diagnosis_only",
        },
        "cross_cohort_patient_join_performed": False,
    }
    report_path = output_root / "processed_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(prepare_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
