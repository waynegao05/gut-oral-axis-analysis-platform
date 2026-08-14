from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_ai_engine_bundle import (
    build_pyinstaller_arguments,
    load_artifact_entries,
)
from scripts.generate_release_manifest import write_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_ai_engine_artifact_manifest_is_complete() -> None:
    manifest = ROOT / "desktop" / "packaging" / "ai-engine-artifacts.json"
    entries = load_artifact_entries(ROOT, manifest)
    sources = {entry["source"].relative_to(ROOT).as_posix() for entry in entries}
    assert "config/releases/ac_icam_real_outcome_pfs_v8.json" in sources
    assert "data/pharmacy_knowledge" in sources
    assert "archive/datasets/topology_v6" in sources
    assert "outputs/current_mainline_v2/temporal_independent_v3" in sources
    assert "outputs/current_mainline_v2/full_risk_head_refiner_v2" in sources


def test_pyinstaller_plan_uses_only_declared_artifacts(tmp_path: Path) -> None:
    entries = load_artifact_entries(
        ROOT,
        ROOT / "desktop" / "packaging" / "ai-engine-artifacts.json",
    )
    arguments = build_pyinstaller_arguments(ROOT, tmp_path / "dist", tmp_path / "work", entries)
    add_data_count = sum(value == "--add-data" for value in arguments)
    assert add_data_count == len(entries)
    assert str(ROOT / "outputs") not in arguments


def test_artifact_destination_cannot_escape_bundle(tmp_path: Path) -> None:
    manifest = tmp_path / "unsafe.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "source": "config/releases/ac_icam_real_outcome_pfs_v8.json",
                        "destination": "../outside",
                        "kind": "file",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the bundle"):
        load_artifact_entries(ROOT, manifest)


def test_release_manifest_hashes_files_and_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "application.exe").write_bytes(b"application")
    output = tmp_path / "release-integrity.json"
    manifest = write_manifest(tmp_path, output)
    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"] == "application.exe"
    assert len(manifest["files"][0]["sha256"]) == 64
