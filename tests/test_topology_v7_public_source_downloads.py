from __future__ import annotations

import json
from pathlib import Path
import zipfile

from experiments.topology_v7_compositional_temporal_v1.download_public_sources import (
    _extract_zip_safely,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "experiments"
    / "topology_v7_compositional_temporal_v1"
    / "source_catalog.json"
)


def test_source_catalog_uses_official_https_pages() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    assert catalog["source_policy"]["official_project_page_required"] is True
    for source in catalog["sources"]:
        assert source["original_project_urls"]
        assert all(
            url.startswith("https://")
            for url in source["original_project_urls"]
        )
        if source["download_entrypoint"] is not None:
            assert source["download_entrypoint"].startswith("https://")
        assert source["license_or_access_terms"]


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    try:
        _extract_zip_safely(archive_path, tmp_path / "extracted")
    except RuntimeError as error:
        assert "outside destination" in str(error)
    else:
        raise AssertionError("Archive traversal must be rejected.")
