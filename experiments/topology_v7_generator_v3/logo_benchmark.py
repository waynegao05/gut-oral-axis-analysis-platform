from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.topology_v7_diagnosis.diagnose import (
    GROUP_COLUMN,
    FrameSplit,
    _aft_run,
    _as_builtin,
    _cox_run,
    _feature_frame,
)
from research.data import split_sample_table


FILE_NAMES = {
    "graph_csv": "topology_v7_sample_graph_table.csv",
    "clinical_csv": "topology_v7_sample_clinical_table.csv",
    "metabolite_csv": "topology_v7_sample_metabolite_table.csv",
    "label_csv": "topology_v7_sample_label_table.csv",
    "provenance_csv": "topology_v7_sample_provenance.csv",
    "manifest_json": "topology_v7_manifest.json",
}
MODEL_SPECS = (
    ("linear_cox", "edge_identity"),
    ("mlp_cox", "edge_identity"),
    ("xgb_aft", "edge_identity"),
    ("xgb_aft", "full_topology"),
)
PRIMARY_GATE_MODEL = ("xgb_aft", "edge_identity")


def _config_for_data_dir(template: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(template)
    for key, name in FILE_NAMES.items():
        config["paths"][key] = str((data_dir / name).as_posix())
    return config


def _explicit_logo_split(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    test_group: int,
    validation_group: int,
) -> FrameSplit:
    controlled = frame[
        ["sample_id", "time", "event", GROUP_COLUMN, *feature_columns]
    ].copy()
    train, val, test, summary = split_sample_table(
        controlled,
        seed=42,
        val_ratio=0.2,
        test_ratio=0.2,
        validation_group=validation_group,
        test_group=test_group,
    )
    if summary["test_groups"] != [str(test_group)]:
        raise RuntimeError("The requested outer test group was not isolated.")
    if summary["val_groups"] != [str(validation_group)]:
        raise RuntimeError("The requested inner validation group was not isolated.")
    return FrameSplit(train=train, val=val, test=test, summary=summary)


def _source_grouped_domain_shift(
    frame: pd.DataFrame,
    provenance: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    test_group: int,
    seed: int,
) -> dict[str, Any]:
    lineage_columns = {
        "sample_id",
        "primary_anchor_patient_id",
        "generation_group_id",
    }
    missing = sorted(lineage_columns.difference(provenance.columns))
    if missing:
        raise ValueError(f"Provenance is missing source-lineage columns: {missing}")

    lineage = provenance[list(lineage_columns)].copy()
    lineage["sample_id"] = lineage["sample_id"].astype(str)
    aligned = frame[["sample_id", GROUP_COLUMN, *feature_columns]].merge(
        lineage[["sample_id", "primary_anchor_patient_id"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    domain = (aligned[GROUP_COLUMN].astype(int) == int(test_group)).astype(int).to_numpy()
    source_groups = aligned["primary_anchor_patient_id"].astype(str).to_numpy()
    values = aligned[list(feature_columns)].astype(float).replace([np.inf, -np.inf], np.nan)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = cross_val_predict(
        pipeline,
        values,
        domain,
        groups=source_groups,
        cv=folds,
        method="predict_proba",
    )[:, 1]

    reference = values.loc[domain == 0]
    held_out = values.loc[domain == 1]
    medians = reference.median(axis=0).fillna(0.0)
    reference = reference.fillna(medians)
    held_out = held_out.fillna(medians)
    shifts: list[dict[str, Any]] = []
    for column in feature_columns:
        left = reference[column].to_numpy(float)
        right = held_out[column].to_numpy(float)
        pooled = math.sqrt((float(np.var(left)) + float(np.var(right))) / 2.0)
        shifts.append(
            {
                "feature": str(column),
                "absolute_smd": abs(float(np.mean(left) - np.mean(right)))
                / max(pooled, 1e-12),
                "normalized_wasserstein": float(wasserstein_distance(left, right))
                / max(float(np.std(left)), 1e-12),
            }
        )
    shifts.sort(key=lambda row: row["absolute_smd"], reverse=True)
    return {
        "test_group": int(test_group),
        "domain_classifier_auc": float(roc_auc_score(domain, probability)),
        "cross_validation": "stratified_group_kfold_by_primary_public_anchor",
        "num_primary_anchors": int(len(np.unique(source_groups))),
        "top_feature_shifts": shifts[:10],
    }


def _lineage_audit(provenance: pd.DataFrame) -> dict[str, Any]:
    anchor_groups: dict[str, set[int]] = {}
    for column in ("primary_anchor_patient_id", "secondary_anchor_patient_id"):
        if column not in provenance:
            raise ValueError(f"Provenance is missing {column}.")
        for anchor, group in provenance[[column, GROUP_COLUMN]].drop_duplicates().itertuples(
            index=False, name=None
        ):
            anchor_groups.setdefault(str(anchor), set()).add(int(group))
    overlaps = {
        anchor: sorted(groups)
        for anchor, groups in anchor_groups.items()
        if len(groups) > 1
    }
    return {
        "public_anchor_overlap_between_generation_groups": overlaps,
        "all_descendants_of_each_anchor_stay_in_one_group": not overlaps,
        "num_unique_public_anchors": int(len(anchor_groups)),
    }


def _aggregate_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(runs)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["dataset", "model_name", "feature_set"], sort=True, dropna=False
    ):
        scores = group["test_c_index"].astype(float)
        rows.append(
            {
                "dataset": str(keys[0]),
                "model_name": str(keys[1]),
                "feature_set": str(keys[2]),
                "num_runs": int(len(group)),
                "mean_test_c_index": float(scores.mean()),
                "std_test_c_index": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "min_test_c_index": float(scores.min()),
                "max_test_c_index": float(scores.max()),
            }
        )
    return rows


def _promotion_gate(
    runs: Sequence[dict[str, Any]],
    domain_shift: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    frame = pd.DataFrame(runs)
    primary = frame.loc[
        (frame["model_name"] == PRIMARY_GATE_MODEL[0])
        & (frame["feature_set"] == PRIMARY_GATE_MODEL[1])
    ].copy()
    means = primary.groupby("dataset")["test_c_index"].mean()
    minima = primary.groupby("dataset")["test_c_index"].min()
    paired = primary.pivot_table(
        index=["outer_test_group", "model_seed"],
        columns="dataset",
        values="test_c_index",
        aggfunc="first",
    ).dropna()
    deltas = paired["candidate_v3"] - paired["reference_v2"]
    candidate_domain = [
        float(row["domain_classifier_auc"])
        for row in domain_shift["candidate_v3"]
    ]
    reference_domain = [
        float(row["domain_classifier_auc"])
        for row in domain_shift["reference_v2"]
    ]
    checks = {
        "primary_model_mean_improves_by_at_least_0_005": bool(
            means["candidate_v3"] - means["reference_v2"] >= 0.005
        ),
        "primary_model_improves_in_at_least_three_outer_groups": bool(
            int((deltas > 0.0).sum()) >= 3
        ),
        "candidate_worst_group_not_more_than_0_005_below_reference": bool(
            minima["candidate_v3"] >= minima["reference_v2"] - 0.005
        ),
        "candidate_mean_source_grouped_domain_auc_at_most_0_85": bool(
            float(np.mean(candidate_domain)) <= 0.85
        ),
        "candidate_domain_auc_improves_by_at_least_0_03": bool(
            float(np.mean(reference_domain)) - float(np.mean(candidate_domain)) >= 0.03
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "primary_model": {
            "model_name": PRIMARY_GATE_MODEL[0],
            "feature_set": PRIMARY_GATE_MODEL[1],
            "candidate_mean_test_c_index": float(means["candidate_v3"]),
            "reference_mean_test_c_index": float(means["reference_v2"]),
            "mean_delta": float(means["candidate_v3"] - means["reference_v2"]),
            "candidate_min_test_c_index": float(minima["candidate_v3"]),
            "reference_min_test_c_index": float(minima["reference_v2"]),
            "fold_seed_deltas": [float(value) for value in deltas.tolist()],
        },
        "domain_shift": {
            "candidate_mean_auc": float(np.mean(candidate_domain)),
            "reference_mean_auc": float(np.mean(reference_domain)),
            "mean_auc_reduction": float(np.mean(reference_domain) - np.mean(candidate_domain)),
        },
        "usage": (
            "Development-cohort promotion gate only. Generator parameters must be locked "
            "before creating the independent formal cohort."
        ),
    }


def run_logo_benchmark(
    *,
    template_config_path: Path,
    candidate_data_dir: Path,
    reference_data_dir: Path,
    output_dir: Path,
    model_seeds: Sequence[int],
    device: str,
    scope: str = "development_generator_gate",
) -> dict[str, Any]:
    if scope not in {"development_generator_gate", "formal_fixed_protocol_audit"}:
        raise ValueError(f"Unsupported benchmark scope: {scope}")
    template = yaml.safe_load(template_config_path.read_text(encoding="utf-8"))
    datasets = {
        "candidate_v3": candidate_data_dir,
        "reference_v2": reference_data_dir,
    }
    runs: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    domain_shift: dict[str, list[dict[str, Any]]] = {}
    lineage: dict[str, Any] = {}

    for dataset_name, data_dir in datasets.items():
        config = _config_for_data_dir(template, data_dir)
        frame, feature_sets, feature_metadata = _feature_frame(config)
        provenance = pd.read_csv(data_dir / FILE_NAMES["provenance_csv"])
        provenance["sample_id"] = provenance["sample_id"].astype(str)
        lineage[dataset_name] = _lineage_audit(provenance)
        split_summaries[dataset_name] = {}
        domain_shift[dataset_name] = []

        groups = sorted(frame[GROUP_COLUMN].astype(int).unique().tolist())
        if groups != [0, 1, 2, 3, 4]:
            raise ValueError(f"Expected generation groups 0..4, got {groups}.")
        for outer_test_group in groups:
            validation_group = groups[(groups.index(outer_test_group) + 1) % len(groups)]
            split = _explicit_logo_split(
                frame,
                feature_sets["full_topology"],
                test_group=outer_test_group,
                validation_group=validation_group,
            )
            split_summaries[dataset_name][str(outer_test_group)] = split.summary
            domain_shift[dataset_name].append(
                _source_grouped_domain_shift(
                    frame,
                    provenance,
                    feature_sets["full_topology"],
                    test_group=outer_test_group,
                    seed=42 + outer_test_group,
                )
            )
            for model_name, feature_set in MODEL_SPECS:
                for model_seed in model_seeds:
                    if model_name == "linear_cox":
                        metrics = _cox_run(
                            split,
                            feature_sets[feature_set],
                            model_type="linear",
                            model_seed=int(model_seed),
                            device=device,
                        )
                    elif model_name == "mlp_cox":
                        metrics = _cox_run(
                            split,
                            feature_sets[feature_set],
                            model_type="mlp",
                            model_seed=int(model_seed),
                            device=device,
                        )
                    else:
                        metrics = _aft_run(
                            split,
                            feature_sets[feature_set],
                            model_seed=int(model_seed),
                        )
                    row = {
                        "dataset": dataset_name,
                        "outer_test_group": int(outer_test_group),
                        "inner_validation_group": int(validation_group),
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "model_seed": int(model_seed),
                        "num_features": int(len(feature_sets[feature_set])),
                        **metrics,
                    }
                    runs.append(row)
                    print(
                        f"{dataset_name} outer={outer_test_group} val={validation_group} "
                        f"{model_name}/{feature_set} seed={model_seed} "
                        f"test_c={metrics['test_c_index']:.4f}",
                        flush=True,
                    )
        split_summaries[dataset_name]["feature_metadata"] = feature_metadata

    summary = {
        "schema_version": 1,
        "scope": scope,
        "selection_boundary": (
            "Outer results may promote a locked generator to a new independent formal cohort; "
            "they may not be reused as formal final-model evidence."
            if scope == "development_generator_gate"
            else "Generator and model specifications were locked before this audit. These "
            "results may be reported, but may not tune the formal cohort or GNN architecture."
        ),
        "template_config_path": str(template_config_path.as_posix()),
        "candidate_data_dir": str(candidate_data_dir.as_posix()),
        "reference_data_dir": str(reference_data_dir.as_posix()),
        "model_seeds": [int(seed) for seed in model_seeds],
        "split_summaries": split_summaries,
        "lineage_audit": lineage,
        "source_grouped_domain_shift": domain_shift,
        "benchmark_runs": runs,
        "aggregates": _aggregate_runs(runs),
    }
    comparison = _promotion_gate(runs, domain_shift)
    if scope == "development_generator_gate":
        summary["promotion_gate"] = comparison
    else:
        comparison["usage"] = (
            "Formal fixed-protocol comparison only. Do not change the generator seed, data, "
            "GNN architecture, or fusion settings in response to these outer-fold results."
        )
        summary["fixed_protocol_comparison"] = comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logo_benchmark_summary.json").write_text(
        json.dumps(_as_builtin(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(runs).to_csv(output_dir / "logo_benchmark_runs.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generator-v3 with generator-v2 under explicit five-group LOGO."
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=Path("config/research/research_config_v7_gnn_optimized.yaml"),
    )
    parser.add_argument("--candidate-data-dir", type=Path, required=True)
    parser.add_argument("--reference-data-dir", type=Path, default=Path("data/research"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--scope",
        choices=["development_generator_gate", "formal_fixed_protocol_audit"],
        default="development_generator_gate",
    )
    args = parser.parse_args()
    result = run_logo_benchmark(
        template_config_path=args.template_config,
        candidate_data_dir=args.candidate_data_dir,
        reference_data_dir=args.reference_data_dir,
        output_dir=args.output_dir,
        model_seeds=args.model_seeds,
        device=args.device,
        scope=args.scope,
    )
    result_key = (
        "promotion_gate"
        if args.scope == "development_generator_gate"
        else "fixed_protocol_comparison"
    )
    print(json.dumps(_as_builtin(result[result_key]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
