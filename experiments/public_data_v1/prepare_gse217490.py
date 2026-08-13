from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_ID = "russo_crc_oral_gut_2023"
GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE217nnn/GSE217490"
SOURCE_FILES = {
    "feature_table.tsv.gz": f"{GEO_BASE}/suppl/GSE217490_feature_table.tsv.gz",
    "taxonomy.tsv.gz": f"{GEO_BASE}/suppl/GSE217490_taxonomy.tsv.gz",
    "series_matrix.txt.gz": f"{GEO_BASE}/matrix/GSE217490_series_matrix.txt.gz",
}
SITE_NAMES = {
    "saliva": "saliva",
    "stool": "stool",
    "colon biopsy": "colon_biopsy",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "goa-public-data-v1/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def ensure_source_files(raw_dir: Path, *, download_missing: bool = True) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename, url in SOURCE_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(f"Missing required public source file: {path}")
            _download(url, path)
        paths[filename] = path
    return paths


def _split_characteristic(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Invalid GEO characteristic: {value!r}")
    label, content = value.split(":", 1)
    return label.strip().lower(), content.strip()


def read_geo_metadata(path: Path) -> pd.DataFrame:
    regular_rows: dict[str, list[str]] = {}
    characteristics: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or not row[0].startswith("!Sample_"):
                continue
            key = row[0]
            values = row[1:]
            if key == "!Sample_characteristics_ch1":
                label, _ = _split_characteristic(values[0])
                parsed = [_split_characteristic(value) for value in values]
                if any(item_label != label for item_label, _ in parsed):
                    raise ValueError(f"Mixed labels in GEO characteristic row: {label}")
                characteristics[label] = [content for _, content in parsed]
            else:
                regular_rows[key] = values

    required_regular = {
        "!Sample_description",
        "!Sample_geo_accession",
        "!Sample_source_name_ch1",
        "!Sample_title",
    }
    required_characteristics = {"patient id", "sex", "condition", "tumor type", "stadiation"}
    missing_regular = sorted(required_regular.difference(regular_rows))
    missing_characteristics = sorted(required_characteristics.difference(characteristics))
    if missing_regular or missing_characteristics:
        raise ValueError(
            "GEO metadata is missing required fields: "
            f"sample={missing_regular}, characteristics={missing_characteristics}"
        )

    row_count = len(regular_rows["!Sample_description"])
    all_columns = [*regular_rows.values(), *characteristics.values()]
    if any(len(values) != row_count for values in all_columns):
        raise ValueError("GEO sample metadata rows have inconsistent lengths.")

    source_names = regular_rows["!Sample_source_name_ch1"]
    sites = []
    for source_name in source_names:
        normalized = SITE_NAMES.get(source_name.strip().lower())
        if normalized is None:
            raise ValueError(f"Unsupported GSE217490 source_name: {source_name!r}")
        sites.append(normalized)

    metadata = pd.DataFrame(
        {
            "feature_sample_id": regular_rows["!Sample_description"],
            "geo_accession": regular_rows["!Sample_geo_accession"],
            "sample_title": regular_rows["!Sample_title"],
            "patient_id": characteristics["patient id"],
            "site": sites,
            "sex": characteristics["sex"],
            "condition": characteristics["condition"],
            "tumor_type": characteristics["tumor type"],
            "stage": characteristics["stadiation"],
        }
    )
    if metadata["feature_sample_id"].duplicated().any():
        raise ValueError("GSE217490 metadata contains duplicate feature sample identifiers.")
    return metadata


def _taxon_label(value: str) -> str:
    ranks: dict[str, str] = {}
    for segment in str(value).split(";"):
        segment = segment.strip()
        if "__" not in segment:
            continue
        prefix, name = segment.split("__", 1)
        ranks[prefix.lower()] = name.strip()

    unusable = {"", "uncultured", "uncultured_bacterium", "uncultured_organism", "unclassified"}
    for prefix, rank_name in (("g", "genus"), ("f", "family"), ("o", "order"), ("c", "class")):
        candidate = ranks.get(prefix, "")
        if candidate.lower() not in unusable:
            return f"{rank_name}:{candidate.replace(' ', '_')}"
    return "taxon:unclassified"


def read_genus_abundance(
    feature_path: Path,
    taxonomy_path: Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    taxonomy = pd.read_csv(taxonomy_path, sep="\t", compression="gzip")
    if not {"Feature ID", "Taxon"}.issubset(taxonomy.columns):
        raise ValueError("GSE217490 taxonomy table is missing Feature ID or Taxon.")
    taxonomy_map = dict(
        zip(taxonomy["Feature ID"].astype(str), taxonomy["Taxon"].map(_taxon_label))
    )

    counts = pd.read_csv(feature_path, sep="\t", compression="gzip", skiprows=1)
    feature_column = "#OTU ID"
    if feature_column not in counts.columns:
        raise ValueError("GSE217490 feature table is missing #OTU ID.")
    counts = counts.set_index(feature_column)
    counts.index = counts.index.astype(str)
    counts = counts.apply(pd.to_numeric, errors="raise")

    missing_samples = sorted(set(metadata["feature_sample_id"]).difference(counts.columns))
    if missing_samples:
        raise ValueError(f"Feature table is missing GEO samples: {missing_samples[:5]}")
    counts = counts.loc[:, metadata["feature_sample_id"].tolist()].T

    mapped_taxa = [taxonomy_map.get(feature_id, "taxon:unclassified") for feature_id in counts.columns]
    counts.columns = mapped_taxa
    counts = counts.T.groupby(level=0, sort=True).sum().T
    totals = counts.sum(axis=1)
    if (totals <= 0).any():
        bad = totals.index[totals <= 0].tolist()[:5]
        raise ValueError(f"Samples with zero total feature count: {bad}")
    return counts.div(totals, axis=0)


def _patient_metadata(metadata: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    fields = ["condition", "sex", "tumor_type", "stage"]
    selected = metadata.loc[metadata["patient_id"].isin(patient_ids)].copy()
    conflicts = []
    for field in fields:
        counts = selected.groupby("patient_id")[field].nunique(dropna=False)
        conflicts.extend(f"{patient_id}:{field}" for patient_id in counts.index[counts > 1])
    if conflicts:
        raise ValueError(f"Conflicting patient metadata values: {conflicts[:5]}")
    return selected.drop_duplicates("patient_id").set_index("patient_id")[fields]


def build_paired_patient_features(
    metadata: pd.DataFrame,
    abundance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    site_sets = metadata.groupby("patient_id")["site"].agg(set)
    paired_ids = set(site_sets.index[site_sets.map(lambda sites: {"saliva", "stool"}.issubset(sites))])
    paired_samples = metadata.loc[
        metadata["patient_id"].isin(paired_ids) & metadata["site"].isin(["saliva", "stool"])
    ].copy()

    sample_to_patient = paired_samples.set_index("feature_sample_id")["patient_id"]
    sample_to_site = paired_samples.set_index("feature_sample_id")["site"]
    paired_abundance = abundance.loc[paired_samples["feature_sample_id"]].copy()
    paired_abundance.insert(0, "site", paired_abundance.index.map(sample_to_site))
    paired_abundance.insert(0, "patient_id", paired_abundance.index.map(sample_to_patient))
    paired_abundance = paired_abundance.set_index(["patient_id", "site"])

    if paired_abundance.index.duplicated().any():
        raise ValueError("GSE217490 contains duplicate patient-site abundance profiles.")
    wide = paired_abundance.unstack("site")
    wide.columns = [f"{site}__{taxon}" for taxon, site in wide.columns]
    wide = wide.reindex(sorted(wide.columns), axis=1)

    patient_metadata = _patient_metadata(metadata, paired_ids)
    output = patient_metadata.join(wide, how="inner")
    output.insert(0, "target_crc", output["condition"].eq("Adenocarcinoma").astype(int))
    output = output.reset_index().sort_values("patient_id").reset_index(drop=True)

    long = paired_abundance.reset_index().melt(
        id_vars=["patient_id", "site"],
        var_name="taxon",
        value_name="relative_abundance",
    )
    long = long.loc[long["relative_abundance"] > 0].sort_values(
        ["patient_id", "site", "taxon"]
    )
    return output, long.reset_index(drop=True), paired_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(root: Path, *, download_missing: bool = True) -> dict[str, Any]:
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    sources = ensure_source_files(raw_dir, download_missing=download_missing)

    metadata = read_geo_metadata(sources["series_matrix.txt.gz"])
    abundance = read_genus_abundance(
        sources["feature_table.tsv.gz"],
        sources["taxonomy.tsv.gz"],
        metadata,
    )
    paired_features, abundance_long, paired_ids = build_paired_patient_features(metadata, abundance)

    metadata.to_csv(processed_dir / "sample_metadata.csv", index=False)
    paired_features.to_csv(processed_dir / "paired_patient_features.csv", index=False)
    abundance_long.to_csv(processed_dir / "paired_genus_abundance_long.csv", index=False)

    patient_conditions = metadata.drop_duplicates("patient_id")["condition"]
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "source_accessions": ["PRJNA899104", "GSE217490"],
        "source_urls": SOURCE_FILES,
        "task": "adenoma_vs_colorectal_adenocarcinoma_classification",
        "right_censored_survival_available": False,
        "patient_level_cross_cohort_join_allowed": False,
        "num_samples": int(len(metadata)),
        "num_patients": int(metadata["patient_id"].nunique()),
        "num_paired_saliva_stool_patients": int(len(paired_ids)),
        "num_taxa": int(abundance.shape[1]),
        "sample_sites": dict(Counter(metadata["site"])),
        "patient_conditions": dict(Counter(patient_conditions)),
        "paired_target_counts": {
            str(key): int(value)
            for key, value in paired_features["target_crc"].value_counts().sort_index().items()
        },
        "raw_sha256": {name: _sha256(path) for name, path in sources.items()},
        "notes": [
            "Patient is the grouping unit.",
            "Only patients with both saliva and stool are in paired_patient_features.csv.",
            "Relative abundance is aggregated to the deepest available genus/family/order/class label.",
            "The paper reports NMR results, but no patient-level metabolomics matrix is public in GEO or its supplement.",
            "No survival time or event was inferred or generated.",
        ],
    }
    (root / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the public GSE217490 oral-gut validation cohort."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/public") / DATASET_ID,
        help="Local dataset root. Raw and processed files are kept outside Git.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require source files to exist instead of downloading missing files.",
    )
    args = parser.parse_args()
    manifest = prepare_dataset(args.root, download_missing=not args.no_download)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
