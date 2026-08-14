from __future__ import annotations

import argparse
from collections import deque
from importlib import metadata
import json
from pathlib import Path
from typing import Iterable

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


DIRECT_RUNTIME_DISTRIBUTIONS = (
    "fastapi",
    "uvicorn",
    "Flask",
    "pandas",
    "numpy",
    "networkx",
    "scikit-learn",
    "torch",
    "torch-geometric",
    "xgboost",
    "PyYAML",
)


def _license(metadata_values: metadata.PackageMetadata) -> str:
    expression = metadata_values.get("License-Expression")
    if expression:
        return expression.strip()
    declared = metadata_values.get("License")
    if declared and declared.strip() and declared.strip().upper() != "UNKNOWN":
        return declared.strip()
    classifiers = metadata_values.get_all("Classifier") or []
    matches = [value.removeprefix("License :: ") for value in classifiers if value.startswith("License :: ")]
    return "; ".join(matches) if matches else "not-declared"


def build_inventory(direct_distributions: Iterable[str] = DIRECT_RUNTIME_DISTRIBUTIONS) -> dict[str, object]:
    direct_names = {canonicalize_name(name) for name in direct_distributions}
    pending = deque(direct_distributions)
    visited: set[str] = set()
    records: list[dict[str, object]] = []
    missing: list[str] = []

    while pending:
        requested_name = pending.popleft()
        normalized = canonicalize_name(requested_name)
        if normalized in visited:
            continue
        visited.add(normalized)
        try:
            distribution = metadata.distribution(requested_name)
        except metadata.PackageNotFoundError:
            missing.append(requested_name)
            continue

        package_metadata = distribution.metadata
        name = package_metadata.get("Name") or requested_name
        project_urls = package_metadata.get_all("Project-URL") or []
        records.append(
            {
                "name": name,
                "version": distribution.version,
                "direct": canonicalize_name(name) in direct_names,
                "license": _license(package_metadata),
                "homepage": package_metadata.get("Home-page"),
                "project_urls": project_urls,
            }
        )
        for raw_requirement in distribution.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            pending.append(requirement.name)

    records.sort(key=lambda item: canonicalize_name(str(item["name"])))
    return {
        "schema_version": 1,
        "scope": "runtime dependency closure from the build environment",
        "direct_distributions": sorted(direct_names),
        "packages": records,
        "missing_distributions": sorted(missing, key=str.lower),
    }


def write_inventory(output_path: Path) -> dict[str, object]:
    inventory = build_inventory()
    if inventory["missing_distributions"]:
        missing = ", ".join(inventory["missing_distributions"])
        raise RuntimeError(f"Runtime dependency inventory is incomplete: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python runtime dependency and license inventory.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = write_inventory(args.output.resolve())
    print(f"Recorded {len(inventory['packages'])} Python runtime distributions.")


if __name__ == "__main__":
    main()
