from __future__ import annotations

import json
from pathlib import Path

from src.export_utils import export_report
from src.pipeline import run_pipeline


DEFAULT_PAYLOAD = {
    "microbes": {
        "Fusobacterium": 0.18,
        "Porphyromonas": 0.15,
        "Prevotella": 0.10,
        "Streptococcus": 0.09,
        "Lactobacillus": 0.02,
    },
    "clinical": {
        "age": 52,
        "bmi": 24.5,
        "smoking": 1,
        "family_history": 1,
    },
    "metabolites": {
        "bile_acids": 0.8,
        "scfa": 0.3,
        "tryptophan_metabolism": 0.7,
    },
}


def main() -> None:
    report = run_pipeline(DEFAULT_PAYLOAD)
    output_path = export_report(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()
