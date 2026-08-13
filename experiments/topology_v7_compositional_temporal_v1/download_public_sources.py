from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "gut-oral-axis-research/1.0"
ZENODO_DATASETS = {
    "alexander_crc_multiomics_2023": {
        "record_id": 7326674,
        "destination": ROOT / "data/public/alexander_crc_multiomics_2023",
        "extract_zip": False,
    },
    "debelius_crc_survival_2023": {
        "record_id": 7690117,
        "destination": ROOT / "data/public/debelius_crc_survival_2023",
        "extract_zip": True,
    },
}
DDBJ_METADATA_DATASETS = {
    "uchida_crc_paired_oral_gut_2021": {
        "accessions": ["PRJDB11845", "DRA012322"],
        "destination": ROOT
        / "data/public/uchida_crc_paired_oral_gut_2021",
        "original_project_urls": [
            "https://ddbj.nig.ac.jp/resource/bioproject/PRJDB11845",
            "https://ddbj.nig.ac.jp/resource/sra-submission/DRA012322",
        ],
        "base_url": (
            "https://ddbj.nig.ac.jp/public/ddbj_database/dra/fastq/"
            "DRA012/DRA012322"
        ),
        "metadata_files": [
            "DRA012322.experiment.xml",
            "DRA012322.run.xml",
            "DRA012322.sample.xml",
            "DRA012322.study.xml",
            "DRA012322.submission.xml",
        ],
        "license_or_access_terms": (
            "DDBJ public archive; DDBJ data-use policy and original study "
            "consent apply."
        ),
        "raw_sequence_policy": (
            "FASTQ files are not downloaded automatically because the archive "
            "contains 206 runs. Use the recorded official directory for an "
            "explicit storage-reviewed download."
        ),
    }
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".part",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _extract_zip_safely(path: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as error:
                raise RuntimeError(
                    f"Refusing archive path outside destination: {member.filename}"
                ) from error
        archive.extractall(destination)
        extracted = [
            member.filename
            for member in archive.infolist()
            if not member.is_dir()
        ]
    return extracted


def download_zenodo_dataset(dataset_id: str) -> dict[str, Any]:
    specification = ZENODO_DATASETS[dataset_id]
    record_id = int(specification["record_id"])
    destination = Path(specification["destination"])
    raw_dir = destination / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    record = _request_json(api_url)
    if int(record["id"]) != record_id:
        raise RuntimeError("Zenodo record ID does not match the request.")

    downloaded_files: list[dict[str, Any]] = []
    extracted_files: list[str] = []
    for item in record["files"]:
        filename = str(item["key"]).replace("/", "_")
        path = raw_dir / filename
        direct_url = str(item["links"]["self"])
        if not path.exists():
            _download(direct_url, path)
        expected_size = int(item["size"])
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Unexpected file size for {path.name}.")
        expected_checksum = str(item["checksum"])
        algorithm, expected_digest = expected_checksum.split(":", maxsplit=1)
        if algorithm != "md5":
            raise RuntimeError(f"Unsupported Zenodo checksum: {algorithm}")
        actual_md5 = _md5(path)
        if actual_md5 != expected_digest:
            raise RuntimeError(f"MD5 mismatch for {path.name}.")
        downloaded_files.append(
            {
                "filename": filename,
                "source_key": str(item["key"]),
                "size_bytes": expected_size,
                "direct_download_url": direct_url,
                "zenodo_checksum": expected_checksum,
                "sha256": _sha256(path),
            }
        )
        if bool(specification["extract_zip"]) and zipfile.is_zipfile(path):
            extracted_files.extend(
                _extract_zip_safely(path, destination / "extracted")
            )

    license_value = record.get("metadata", {}).get("license")
    if isinstance(license_value, dict):
        license_value = license_value.get("id") or license_value.get("title")
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "record_id": record_id,
        "record_doi": record.get("doi"),
        "title": record["metadata"]["title"],
        "original_project_url": f"https://zenodo.org/records/{record_id}",
        "official_api_url": api_url,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_or_access_terms": license_value,
        "files": downloaded_files,
        "extracted_files": sorted(extracted_files),
        "raw_files_are_git_ignored": True,
    }
    manifest_path = destination / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def download_ddbj_metadata(dataset_id: str) -> dict[str, Any]:
    specification = DDBJ_METADATA_DATASETS[dataset_id]
    destination = Path(specification["destination"])
    raw_dir = destination / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(specification["base_url"]).rstrip("/")
    downloaded_files: list[dict[str, Any]] = []
    for filename in specification["metadata_files"]:
        source_url = f"{base_url}/{filename}"
        path = raw_dir / str(filename)
        if not path.exists():
            _download(source_url, path)
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty DDBJ metadata file: {filename}")
        downloaded_files.append(
            {
                "filename": str(filename),
                "direct_download_url": source_url,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "official_accessions": list(specification["accessions"]),
        "original_project_urls": list(
            specification["original_project_urls"]
        ),
        "official_download_directory": f"{base_url}/",
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_or_access_terms": specification[
            "license_or_access_terms"
        ],
        "raw_sequence_policy": specification["raw_sequence_policy"],
        "files": downloaded_files,
        "raw_fastq_downloaded": False,
        "raw_files_are_git_ignored": True,
    }
    manifest_path = destination / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    dataset_ids = sorted(
        {*ZENODO_DATASETS.keys(), *DDBJ_METADATA_DATASETS.keys()}
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", *dataset_ids],
        default="all",
    )
    args = parser.parse_args()
    selected = dataset_ids if args.dataset == "all" else [args.dataset]
    reports = []
    for dataset_id in selected:
        if dataset_id in ZENODO_DATASETS:
            reports.append(download_zenodo_dataset(dataset_id))
        else:
            reports.append(download_ddbj_metadata(dataset_id))
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
