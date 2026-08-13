from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "oral_adenoma_internal_v3"
    / "oral_adenoma_internal_model.joblib"
)
DEFAULT_METRICS = ROOT / "outputs" / "oral_adenoma_internal_v3" / "metrics.json"
DEFAULT_OUTPUT = ROOT / "config" / "releases" / "oral_adenoma_internal_v3.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_list(values: Any, name: str) -> list[float]:
    output = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return [float(value) for value in output]


def build_release(model_path: Path, metrics_path: Path) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    if bundle.get("protocol_id") != "oral_adenoma_nested_oof_v3":
        raise ValueError("Unexpected oral adenoma protocol in the source bundle.")
    if bundle.get("research_only_not_web") is not True:
        raise ValueError("The source bundle is not locked as research-only.")

    pipeline = bundle["pipeline"]
    clr = pipeline.named_steps["clr"]
    scaler = pipeline.named_steps["scale"]
    selector = pipeline.named_steps["select"]
    model = pipeline.named_steps["model"]

    feature_ids = [str(value) for value in bundle["feature_ids"]]
    taxonomies = [str(value) for value in bundle["taxonomies"]]
    if len(feature_ids) != len(taxonomies) or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Feature IDs and taxonomies must be one-to-one and unique.")

    selected_indices = [int(value) for value in selector.get_support(indices=True)]
    coefficient = _finite_list(model.coef_[0], "model coefficient")
    if len(selected_indices) != len(coefficient):
        raise ValueError("Selected feature count does not match coefficient count.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    primary = metrics["primary"]
    release = {
        "schema_version": 1,
        "release_name": "oral_adenoma_internal_v3",
        "protocol_id": str(bundle["protocol_id"]),
        "endpoint": str(bundle["endpoint"]),
        "research_only": True,
        "source_model_sha256": sha256(model_path),
        "sample_type": "oral_swab_only",
        "allowed_sample_types": [str(value) for value in bundle["allowed_sample_types"]],
        "feature_ids": feature_ids,
        "taxonomies": taxonomies,
        "selected_taxonomies": [str(value) for value in bundle["selected_taxonomies"]],
        "preprocessing": {
            "input_unit": "percent",
            "required_sum_range_percent": [99.9, 100.1],
            "pseudocount_percent": float(clr.pseudocount_percent),
            "scaler_mean": _finite_list(scaler.mean_, "scaler mean"),
            "scaler_scale": _finite_list(scaler.scale_, "scaler scale"),
            "selected_indices": selected_indices,
        },
        "model": {
            "type": "binary_logistic_regression",
            "classes": [int(value) for value in model.classes_],
            "coefficient": coefficient,
            "intercept": float(model.intercept_[0]),
        },
        "operating_threshold": float(bundle["threshold"]),
        "training": {
            "real_patient_count": int(bundle["training_real_patient_count"]),
            "group_counts": bundle["training_group_counts"],
            "selected_config_id": str(bundle["selected_config_id"]),
        },
        "formal_internal_metrics": primary,
        "claim_boundary": str(bundle["claim_boundary"]),
    }

    numeric_values = [
        release["operating_threshold"],
        release["preprocessing"]["pseudocount_percent"],
        release["model"]["intercept"],
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("Release metadata contains a non-finite value.")
    return release


def write_release(release: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(release, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the locked oral-adenoma sklearn model as audited JSON weights."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_release(
        build_release(args.model.resolve(), args.metrics.resolve()),
        args.output.resolve(),
    )
    print(f"Exported oral adenoma release: {output_path}")
    print(f"SHA256: {sha256(output_path)}")


if __name__ == "__main__":
    main()
