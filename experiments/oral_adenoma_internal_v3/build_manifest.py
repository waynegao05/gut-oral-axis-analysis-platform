from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "oral_adenoma_internal_v3"


def build_manifest(output_dir: Path) -> Path:
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    target = output_dir / "artifact_manifest.json"
    target.write_text(
        json.dumps(
            {
                "protocol_id": "oral_adenoma_nested_oof_v3",
                "research_only_not_web": True,
                "files": records,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    target = build_manifest(args.output_dir)
    print(f"Manifest: {target.resolve()}")


if __name__ == "__main__":
    main()
