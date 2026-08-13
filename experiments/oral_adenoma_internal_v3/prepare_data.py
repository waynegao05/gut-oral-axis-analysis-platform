from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://www.thno.org/v10/p11595/thnov10p11595s2.zip"
SOURCE_SHA256 = "6a8fa8ae2f75f21a986e820d420395e089fc992821509432831b620c9c2dffac"
DEFAULT_RAW_DIR = ROOT / "data" / "public" / "zhang_oral_adenoma_2020" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "public" / "zhang_oral_adenoma_2020" / "processed"

ALLOWED_SAMPLE_TYPES = {"oral", "oral_swab", "buccal_swab", "saliva"}
FORBIDDEN_TOKENS = {
    "stool",
    "fecal",
    "faecal",
    "gut",
    "intestinal",
    "blood",
    "serum",
    "plasma",
    "tissue",
}
EXPECTED_GROUP_COUNTS = {"healthy": 58, "adenoma": 34}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_oral_only(values: pd.Series | list[str] | tuple[str, ...]) -> None:
    normalized = [str(value).strip().lower().replace("-", "_").replace(" ", "_") for value in values]
    disallowed = sorted(
        {value for value in normalized if any(token in value for token in FORBIDDEN_TOKENS)}
    )
    if disallowed:
        raise ValueError(f"Forbidden non-oral sample types detected: {disallowed}")
    unexpected = sorted(set(normalized).difference(ALLOWED_SAMPLE_TYPES))
    if unexpected:
        raise ValueError(f"Unexpected sample types; oral/saliva only: {unexpected}")


def download_supplement(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "gut-oral-axis-analysis-platform/1.0"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = sha256(destination)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Supplement SHA256 mismatch: expected {SOURCE_SHA256}, got {actual}")
    return destination


def extract_required_tables(archive_path: Path, raw_dir: Path) -> tuple[Path, Path]:
    required = ("thno_49515i2_2.xlsx", "thno_49515i2_7.xlsx")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = sorted(set(required).difference(names))
        if missing:
            raise ValueError(f"Supplement is missing required files: {missing}")
        for name in required:
            target = raw_dir / name
            with archive.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return raw_dir / required[0], raw_dir / required[1]


def prepare_dataset(metadata_path: Path, genus_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metadata = pd.read_excel(metadata_path)
    genus = pd.read_excel(genus_path)
    if not {"SampleID", "Group"}.issubset(metadata.columns):
        raise ValueError("Supplementary metadata must contain SampleID and Group.")
    if "Genus" not in genus.columns:
        raise ValueError("Supplementary abundance table must contain Genus.")
    if metadata["SampleID"].duplicated().any():
        raise ValueError("Duplicate sample IDs are not allowed.")
    if genus["Genus"].duplicated().any():
        raise ValueError("Duplicate genus names are not allowed.")

    mapping = {"Normal": "healthy", "Adenoma": "adenoma"}
    formal_metadata = metadata.loc[metadata["Group"].isin(mapping)].copy()
    formal_metadata["disease_group"] = formal_metadata["Group"].map(mapping)
    counts = formal_metadata["disease_group"].value_counts().to_dict()
    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(f"Unexpected oral formal-cohort counts: {counts}")

    metadata_columns = {
        "Genus",
        "log2FC",
        "wilcox.test.p_value",
        "q_value",
        "significance",
        "regulation",
        "mean",
        "mean_Group1",
        "mean_Group2",
    }
    abundance_sample_ids = [column for column in genus.columns if column not in metadata_columns]
    expected_ids = formal_metadata["SampleID"].astype(str).tolist()
    if set(abundance_sample_ids) != set(expected_ids):
        missing = sorted(set(expected_ids).difference(abundance_sample_ids))
        extra = sorted(set(abundance_sample_ids).difference(expected_ids))
        raise ValueError(f"Oral abundance/metadata mismatch; missing={missing[:5]}, extra={extra[:5]}")

    abundance = genus.set_index("Genus").loc[:, expected_ids].T.astype(float)
    values = abundance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Oral genus abundances must be finite percentages in [0, 100].")
    sums = values.sum(axis=1)
    if not np.all((sums >= 99.9) & (sums <= 100.1)):
        raise ValueError("Oral relative-abundance rows must sum to approximately 100%.")

    taxonomies = abundance.columns.astype(str).tolist()
    feature_ids = [f"oral_genus_{index:03d}" for index in range(len(taxonomies))]
    abundance.columns = feature_ids
    formal_metadata = formal_metadata.set_index("SampleID").loc[expected_ids]
    output = pd.DataFrame(
        {
            "sample_id": expected_ids,
            "subject_id": expected_ids,
            "sample_type": "oral_swab",
            "source_study": "Zhang_2020_Theranostics",
            "source_sample_prefix": ["".join(filter(str.isalpha, value)) for value in expected_ids],
            "disease_group": formal_metadata["disease_group"].to_numpy(),
            "adenoma_label": (formal_metadata["disease_group"] == "adenoma").astype(int).to_numpy(),
        }
    )
    assert_oral_only(output["sample_type"])
    output = pd.concat([output.reset_index(drop=True), abundance.reset_index(drop=True)], axis=1)
    feature_map = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "rank": "genus",
            "taxonomy": taxonomies,
            "source_table": "Table S7",
        }
    )
    quality = {
        "source_url": SOURCE_URL,
        "source_metadata_sha256": sha256(metadata_path),
        "source_genus_sha256": sha256(genus_path),
        "sample_type": "oral_swab",
        "real_patient_count": int(len(output)),
        "group_counts": counts,
        "unique_samples": int(output["sample_id"].nunique()),
        "unique_subjects": int(output["subject_id"].nunique()),
        "genus_features": int(len(feature_map)),
        "invalid_abundance_values": 0,
        "minimum_sample_sum_percent": float(sums.min()),
        "maximum_sample_sum_percent": float(sums.max()),
        "zero_fraction": float(np.mean(values == 0)),
        "forbidden_sample_types_found": [],
        "crc_samples_excluded": int((metadata["Group"] == "Cancer").sum()),
        "synthetic_samples": 0,
        "lesion_size_evidence": "cohort mean 0.8 +/- 0.3 cm; no individual size labels",
    }
    return output, feature_map, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or args.raw_dir / "thnov10p11595s2.zip"
    if args.archive:
        if sha256(archive) != SOURCE_SHA256:
            raise ValueError("Provided supplement archive failed the locked SHA256 check.")
    else:
        download_supplement(archive)
    metadata_path, genus_path = extract_required_tables(archive, args.raw_dir)
    frame, feature_map, quality = prepare_dataset(metadata_path, genus_path)
    frame.to_csv(args.processed_dir / "oral_adenoma_genus.csv", index=False)
    feature_map.to_csv(args.processed_dir / "oral_adenoma_feature_map.csv", index=False)
    (args.processed_dir / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(frame)} real oral-swab samples and {len(feature_map)} genera.")
    print(f"Output: {args.processed_dir.resolve()}")


if __name__ == "__main__":
    main()
