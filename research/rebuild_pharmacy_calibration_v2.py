from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_TABLE = PROJECT_ROOT / "data" / "research" / "topology_v6_sample_graph_table.csv"
DEFAULT_KNOWLEDGE = PROJECT_ROOT / "data" / "pharmacy_rules_v3.json"


def load_knowledge(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pharmacy knowledge base root must be a JSON object.")
    return payload


def compute_panel_composition(
    graph_table_path: Path,
    marker_panel: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(
        graph_table_path,
        usecols=["sample_id", "node_name", "abundance"],
    )
    if frame.empty:
        raise ValueError("Graph table contains no rows.")

    duplicate_counts = frame.groupby(["sample_id", "node_name"])["abundance"].nunique()
    if (duplicate_counts > 1).any():
        raise ValueError("Graph table contains inconsistent duplicate node abundances.")

    nodes = frame.drop_duplicates(["sample_id", "node_name"])
    abundance = nodes.pivot(index="sample_id", columns="node_name", values="abundance")
    missing_markers = sorted(set(marker_panel).difference(abundance.columns))
    if missing_markers:
        raise ValueError(f"Graph table is missing markers: {', '.join(missing_markers)}")

    abundance = abundance.loc[:, list(marker_panel)].astype(float)
    if abundance.isna().any().any():
        raise ValueError("Graph table has incomplete marker panels.")
    if (abundance < 0.0).any().any():
        raise ValueError("Graph table contains negative marker abundance.")

    panel_total = abundance.sum(axis=1)
    if (panel_total <= 0.0).any():
        raise ValueError("Graph table contains a zero-total marker panel.")
    return abundance.div(panel_total, axis=0)


def expected_rule_thresholds(
    knowledge: Mapping[str, Any],
    graph_table_path: Path,
) -> tuple[dict[str, float], int]:
    calibration = knowledge["calibration"]
    marker_panel = [str(marker) for marker in calibration["required_marker_panel"]]
    composition = compute_panel_composition(graph_table_path, marker_panel)
    interpolation = str(calibration.get("quantile_interpolation", "linear"))

    thresholds: dict[str, float] = {}
    for rule in knowledge["marker_rules"]:
        quantile_name = str(rule["threshold_quantile"])
        if not quantile_name.startswith("q"):
            raise ValueError(f"Unsupported quantile label: {quantile_name}")
        quantile = float(quantile_name[1:]) / 100.0
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile is outside 0-100: {quantile_name}")
        marker = str(rule["marker"])
        thresholds[str(rule["rule_id"])] = float(
            composition[marker].quantile(quantile, interpolation=interpolation)
        )
    return thresholds, len(composition)


def verify_thresholds(
    knowledge: Mapping[str, Any],
    expected: Mapping[str, float],
    sample_count: int,
    *,
    tolerance: float = 5e-7,
) -> list[str]:
    errors: list[str] = []
    configured_count = int(knowledge["calibration"]["sample_count"])
    if configured_count != sample_count:
        errors.append(
            f"sample_count mismatch: configured={configured_count}, expected={sample_count}"
        )

    for rule in knowledge["marker_rules"]:
        rule_id = str(rule["rule_id"])
        configured = float(rule["threshold"])
        calculated = float(expected[rule_id])
        if abs(configured - calculated) > tolerance:
            errors.append(
                f"{rule_id} mismatch: configured={configured:.9f}, expected={calculated:.9f}"
            )
    return errors


def write_thresholds(
    path: Path,
    knowledge: dict[str, Any],
    expected: Mapping[str, float],
    sample_count: int,
) -> None:
    knowledge["calibration"]["sample_count"] = sample_count
    for rule in knowledge["marker_rules"]:
        rule["threshold"] = round(float(expected[str(rule["rule_id"])]), 6)
    path.write_text(
        json.dumps(knowledge, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild or verify pharmacy marker-composition quantiles."
    )
    parser.add_argument("--graph-table", type=Path, default=DEFAULT_GRAPH_TABLE)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write rounded thresholds to the knowledge JSON. Default is read-only verification.",
    )
    args = parser.parse_args()

    knowledge = load_knowledge(args.knowledge)
    expected, sample_count = expected_rule_thresholds(knowledge, args.graph_table)
    if args.write:
        write_thresholds(args.knowledge, knowledge, expected, sample_count)

    errors = verify_thresholds(knowledge, expected, sample_count)
    result = {
        "sample_count": sample_count,
        "value_scale": knowledge["calibration"]["value_scale"],
        "expected_thresholds": {
            rule_id: round(value, 9) for rule_id, value in expected.items()
        },
        "verified": not errors,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
