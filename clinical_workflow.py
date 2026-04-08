from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clinical_report_builder import build_clinical_report
from src.clinical_standardizer import standardize_raw_payload
from src.pharmacy_advice import build_pharmacy_assistance
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw clinical JSON input")
    parser.add_argument("--output", default="outputs/clinical_report.json", help="Path to final report JSON")
    parser.add_argument("--standardized_output", default="outputs/standardized_input.json", help="Path to standardized model input JSON")
    args = parser.parse_args()

    raw_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    standardized = standardize_raw_payload(raw_payload)
    model_report = run_pipeline(standardized)
    pharmacy_advice = build_pharmacy_assistance(model_report, standardized.get("metadata", {}))
    final_report = build_clinical_report(standardized, model_report, pharmacy_advice)

    Path(args.standardized_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.standardized_output).write_text(json.dumps(standardized, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    print(f"\nStandardized model input saved to: {Path(args.standardized_output).resolve()}")
    print(f"Final report saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
