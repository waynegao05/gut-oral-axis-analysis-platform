from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, output_path: Path, version_manifest: Path | None) -> dict[str, object]:
    root = root.resolve()
    output_path = output_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Release root does not exist: {root}")
    excluded = output_path.relative_to(root) if output_path.is_relative_to(root) else None
    files: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if excluded is not None and relative == excluded:
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": relative.as_posix(),
                "size": size,
                "sha256": hash_file(path),
            }
        )

    versions: dict[str, object] = {}
    if version_manifest is not None:
        versions = json.loads(version_manifest.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def write_manifest(root: Path, output_path: Path, version_manifest: Path | None = None) -> dict[str, object]:
    manifest = build_manifest(root, output_path, version_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SHA-256 release file manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version-manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "release-integrity.json"
    version_manifest = args.version_manifest.resolve() if args.version_manifest else None
    manifest = write_manifest(root, output, version_manifest)
    print(f"Hashed {manifest['file_count']} files ({manifest['total_bytes']} bytes).")


if __name__ == "__main__":
    main()
