from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

if __package__:
    from scripts.generate_python_dependency_inventory import write_inventory
    from scripts.generate_release_manifest import write_manifest
else:
    from generate_python_dependency_inventory import write_inventory
    from generate_release_manifest import write_manifest


HIDDEN_IMPORTS = (
    "src.ac_icam_v8_bridge",
    "src.temporal_topology_bridge",
    "src.oral_adenoma_bridge",
    "archive.legacy_web_backends.cox_ensemble_v1",
    "research.data",
    "research.model_v2",
    "research.task",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
)

COLLECT_SUBMODULES = (
    "uvicorn",
)

COLLECT_ALL: tuple[str, ...] = ()

EXCLUDED_MODULES = (
    "matplotlib",
    "IPython",
    "notebook",
    "jupyter",
    "jupyterlab",
    "streamlit",
    "tkinter",
    "_tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
)


def load_artifact_entries(repository_root: Path, manifest_path: Path) -> list[dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
        raise ValueError("Unsupported AI Engine artifact manifest.")

    entries: list[dict[str, object]] = []
    for raw_entry in payload["entries"]:
        if not isinstance(raw_entry, dict):
            raise ValueError("Artifact entries must be JSON objects.")
        source_value = str(raw_entry.get("source", ""))
        destination = str(raw_entry.get("destination", ""))
        kind = str(raw_entry.get("kind", ""))
        required = raw_entry.get("required") is True
        if not source_value or not destination or kind not in {"file", "directory"}:
            raise ValueError(f"Invalid artifact entry: {raw_entry}")
        if Path(destination).is_absolute() or ".." in Path(destination).parts:
            raise ValueError(f"Artifact destination escapes the bundle: {destination}")
        source = (repository_root / source_value).resolve()
        if not source.is_relative_to(repository_root):
            raise ValueError(f"Artifact source escapes the repository: {source_value}")
        exists = source.is_file() if kind == "file" else source.is_dir()
        if required and not exists:
            raise FileNotFoundError(f"Required AI Engine artifact is missing: {source}")
        if exists:
            entries.append(
                {
                    "source": source,
                    "destination": destination.replace("\\", "/"),
                    "kind": kind,
                }
            )
    return entries


def build_pyinstaller_arguments(
    repository_root: Path,
    output_root: Path,
    work_root: Path,
    artifact_entries: list[dict[str, object]],
) -> list[str]:
    arguments = [
        str(repository_root / "ai_engine" / "__main__.py"),
        "--name",
        "goa-ai-engine",
        "--onedir",
        "--console",
        "--noconfirm",
        "--clean",
        "--noupx",
        "--distpath",
        str(output_root),
        "--workpath",
        str(work_root / "work"),
        "--specpath",
        str(work_root / "spec"),
        "--paths",
        str(repository_root),
    ]
    for module in EXCLUDED_MODULES:
        arguments.extend(["--exclude-module", module])
    for module in HIDDEN_IMPORTS:
        arguments.extend(["--hidden-import", module])
    for module in COLLECT_SUBMODULES:
        arguments.extend(["--collect-submodules", module])
    for module in COLLECT_ALL:
        arguments.extend(["--collect-all", module])
    for entry in artifact_entries:
        arguments.extend(
            [
                "--add-data",
                f"{entry['source']}{os.pathsep}{entry['destination']}",
            ]
        )
    return arguments


def build_bundle(
    repository_root: Path,
    output_root: Path,
    work_root: Path,
    artifact_manifest: Path,
    *,
    plan_only: bool = False,
) -> Path:
    repository_root = repository_root.resolve()
    output_root = output_root.resolve()
    work_root = work_root.resolve()
    artifact_manifest = artifact_manifest.resolve()
    bundle_root = output_root / "goa-ai-engine"
    if bundle_root.exists():
        raise FileExistsError(
            f"AI Engine bundle already exists; choose a new output directory: {bundle_root}"
        )

    entries = load_artifact_entries(repository_root, artifact_manifest)
    arguments = build_pyinstaller_arguments(repository_root, output_root, work_root, entries)
    if plan_only:
        print(json.dumps({"arguments": arguments, "artifact_count": len(entries)}, indent=2))
        return bundle_root

    if os.name != "nt":
        raise RuntimeError("The Windows AI Engine bundle must be built on Windows.")
    try:
        import PyInstaller.__main__
    except ImportError as error:
        raise RuntimeError(
            "PyInstaller is not installed. Install requirements-desktop-build.txt first."
        ) from error

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    PyInstaller.__main__.run(arguments)
    executable = bundle_root / "goa-ai-engine.exe"
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller did not produce the expected executable: {executable}")

    shutil.copy2(artifact_manifest, bundle_root / "ai-engine-artifacts.json")
    write_inventory(bundle_root / "python-dependencies.json")
    write_manifest(bundle_root, bundle_root / "runtime-integrity.json")
    return bundle_root


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a standalone Windows AI Engine bundle.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=repository_root / "desktop" / "packaging" / "ai-engine-artifacts.json",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(
        repository_root,
        args.output_dir,
        args.work_dir,
        args.artifact_manifest,
        plan_only=args.plan_only,
    )
    if not args.plan_only:
        print(f"Standalone AI Engine written to {bundle}")


if __name__ == "__main__":
    main()
