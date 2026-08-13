from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.oral_adenoma_internal_v3.prepare_data import assert_oral_only


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "outputs" / "oral_adenoma_internal_v3" / "oral_adenoma_internal_model.joblib"


def predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    if "sample_type" not in frame.columns:
        raise ValueError("Input must declare sample_type; oral or saliva only.")
    assert_oral_only(frame["sample_type"])
    feature_ids = [str(value) for value in bundle["feature_ids"]]
    missing = sorted(set(feature_ids).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required oral genus features: {missing[:5]}")
    values = frame.loc[:, feature_ids].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Oral genus abundances must be finite percentages in [0, 100].")
    sums = values.sum(axis=1)
    if not np.all((sums >= 99.9) & (sums <= 100.1)):
        raise ValueError("Oral genus abundances must sum to approximately 100% per sample.")
    probability = bundle["pipeline"].predict_proba(values)[:, 1]
    output = pd.DataFrame(
        {
            "adenoma_probability": probability,
            "threshold": float(bundle["threshold"]),
            "screen_positive": probability >= float(bundle["threshold"]),
        }
    )
    if "sample_id" in frame.columns:
        output.insert(0, "sample_id", frame["sample_id"].astype(str).to_numpy())
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    if not bundle.get("research_only_not_web"):
        raise ValueError("The supplied bundle is not the locked internal oral model.")
    frame = pd.read_csv(args.input_csv)
    output = predict_frame(bundle, frame)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Predicted {len(output)} oral samples: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
