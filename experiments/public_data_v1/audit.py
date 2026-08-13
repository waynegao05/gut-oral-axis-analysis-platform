from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.public_data_v1.registry import build_audit_report, load_registry


def _compact_missing(values: list[str]) -> str:
    return ",".join(values) if values else "none"


def format_table(report: dict[str, Any]) -> str:
    headers = ("dataset", "access", "full", "survival-core", "missing")
    rows = []
    for item in report["assessments"]:
        rows.append(
            (
                item["id"],
                item["access_mode"],
                "ready" if item["full_replacement_ready"] else "no",
                "ready"
                if item["survival_core_ready"]
                else ("after-access" if item["survival_core_satisfied"] else "no"),
                _compact_missing(item["missing_full_contract"]),
            )
        )

    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    def render(row: tuple[str, ...]) -> str:
        return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    summary = [
        render(headers),
        separator,
        *(render(row) for row in rows),
        "",
        f"Strict full replacement available: {report['strict_full_replacement_available']}",
        "Cross-cohort patient-level joins allowed: False",
    ]
    return "\n".join(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit public cohorts against the independent oral-gut survival contracts."
    )
    parser.add_argument("--registry", type=Path, help="Optional registry JSON path.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--output", type=Path, help="Also write the report to this path.")
    args = parser.parse_args()

    report = build_audit_report(load_registry(args.registry))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) if args.json else format_table(report)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
