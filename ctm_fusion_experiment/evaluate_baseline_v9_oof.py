from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctm_fusion_experiment.evaluate_baseline_v5 import build_baseline_v5_comparison_summary
from ctm_fusion_experiment.utils.reporting import write_csv, write_json


def build_baseline_v9_oof_summary(output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    result = build_baseline_v5_comparison_summary(output_path)
    result["interpretation"] = (
        "Baseline v9 uses fold-local frozen graph embeddings and inner-fold out-of-fold head predictions "
        "to select standardized risk ensembles before the outer test set is evaluated. The outer validation "
        "split can gate the OOF-selected ensemble back to the reference concat baseline."
    )
    write_json(output_path / "baseline_v9_oof_summary.json", result)
    write_csv(output_path / "baseline_v9_oof_fold_comparison.csv", result["folds"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/ctm_fusion_experiment/baseline_v9_oof_formal")
    args = parser.parse_args()
    print(json.dumps(build_baseline_v9_oof_summary(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
