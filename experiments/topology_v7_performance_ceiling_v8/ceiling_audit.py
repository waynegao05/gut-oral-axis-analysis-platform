from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.topology_v7_diagnosis.diagnose import (
    GROUP_COLUMN,
    _feature_frame,
)
from research.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "research_config_v7_v3_gnn_locked.yaml"
DEFAULT_GNN_ROOT = (
    ROOT / "outputs/topology_v7_generator_v3_formal/gnn_locked_logo"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/topology_v7_performance_ceiling_v8/audit_only"
)
TARGET_C_INDEX = 0.761
PROVENANCE_TARGET = "survival_latent_risk"


@dataclass(frozen=True)
class SurrogateSpec:
    name: str
    factory: Callable[[], RegressorMixin]


def _surrogate_specs(seed: int) -> list[SurrogateSpec]:
    specs = [
        SurrogateSpec(
            name=f"ridge_alpha_{alpha:g}",
            factory=lambda alpha=alpha: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=alpha),
            ),
        )
        for alpha in (0.1, 1.0, 10.0)
    ]
    for leaf in (2, 5, 10):
        specs.append(
            SurrogateSpec(
                name=f"extra_trees_leaf_{leaf}",
                factory=lambda leaf=leaf: make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesRegressor(
                        n_estimators=320,
                        min_samples_leaf=leaf,
                        max_features=0.8,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            )
        )
    for leaves in (15, 31):
        for l2 in (0.1, 1.0):
            specs.append(
                SurrogateSpec(
                    name=f"hist_gbdt_leaves_{leaves}_l2_{l2:g}",
                    factory=lambda leaves=leaves, l2=l2: make_pipeline(
                        SimpleImputer(strategy="median"),
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            learning_rate=0.05,
                            max_leaf_nodes=leaves,
                            min_samples_leaf=20,
                            l2_regularization=l2,
                            random_state=seed,
                        ),
                    ),
                )
            )
    return specs


def _finite_correlation(
    metric: Callable[[np.ndarray, np.ndarray], Any],
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    value = metric(left, right)
    statistic = value.statistic if hasattr(value, "statistic") else value[0]
    return float(statistic) if np.isfinite(statistic) else 0.0


def _prediction_metrics(
    *,
    latent: np.ndarray,
    prediction: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> dict[str, float]:
    return {
        "latent_r2": float(r2_score(latent, prediction)),
        "latent_mae": float(mean_absolute_error(latent, prediction)),
        "latent_pearson": _finite_correlation(pearsonr, latent, prediction),
        "latent_spearman": _finite_correlation(spearmanr, latent, prediction),
        "observed_outcome_c_index": float(
            concordance_index(time, event, prediction)
        ),
        "direct_latent_observed_outcome_c_index": float(
            concordance_index(time, event, latent)
        ),
    }


def _load_gnn_predictions(root: Path, group: int, split: str) -> pd.DataFrame:
    validation_group = (group + 1) % 5
    path = (
        root
        / f"outer_group{group}_val{validation_group}_five_seed"
        / f"{split}_ensemble_summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["predictions"])
    required = {"sample_id", "time", "event", "ensemble_risk"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing GNN prediction fields: {missing}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame


def _load_audit_frame(
    config_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame, feature_sets, feature_metadata = _feature_frame(config)
    provenance_path = ROOT / config["paths"]["provenance_csv"]
    provenance = pd.read_csv(provenance_path)
    provenance["sample_id"] = provenance["sample_id"].astype(str)
    required = {"sample_id", GROUP_COLUMN, PROVENANCE_TARGET}
    missing = sorted(required.difference(provenance.columns))
    if missing:
        raise ValueError(f"Provenance is missing audit fields: {missing}")

    audit = frame.merge(
        provenance[["sample_id", PROVENANCE_TARGET]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(audit) != len(frame):
        raise RuntimeError("Not every deployable sample has provenance audit data.")
    return audit, feature_sets["full_topology"], feature_metadata


def _select_surrogate(
    *,
    specs: list[SurrogateSpec],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    validation_y: np.ndarray,
) -> tuple[SurrogateSpec, list[dict[str, float | str]]]:
    scores: list[dict[str, float | str]] = []
    for spec in specs:
        model = spec.factory()
        model.fit(train_x, train_y)
        prediction = np.asarray(model.predict(validation_x), dtype=float)
        scores.append(
            {
                "name": spec.name,
                "validation_latent_r2": float(
                    r2_score(validation_y, prediction)
                ),
                "validation_latent_spearman": _finite_correlation(
                    spearmanr, validation_y, prediction
                ),
            }
        )
    scores.sort(
        key=lambda row: (
            float(row["validation_latent_r2"]),
            float(row["validation_latent_spearman"]),
        ),
        reverse=True,
    )
    selected_name = str(scores[0]["name"])
    selected = next(spec for spec in specs if spec.name == selected_name)
    return selected, scores


def _fold_audit(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    outer_group: int,
    gnn_root: Path,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    validation_group = (outer_group + 1) % 5
    test_mask = frame[GROUP_COLUMN].astype(int).to_numpy() == outer_group
    validation_mask = (
        frame[GROUP_COLUMN].astype(int).to_numpy() == validation_group
    )
    train_mask = ~(test_mask | validation_mask)
    train_validation_mask = ~test_mask

    x = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    latent = frame[PROVENANCE_TARGET].to_numpy(float)
    time = frame["time"].to_numpy(float)
    event = frame["event"].to_numpy(float)
    specs = _surrogate_specs(seed + outer_group)
    selected, validation_scores = _select_surrogate(
        specs=specs,
        train_x=x.loc[train_mask],
        train_y=latent[train_mask],
        validation_x=x.loc[validation_mask],
        validation_y=latent[validation_mask],
    )

    model = selected.factory()
    model.fit(x.loc[train_validation_mask], latent[train_validation_mask])
    test_prediction = np.asarray(model.predict(x.loc[test_mask]), dtype=float)
    test_frame = frame.loc[
        test_mask,
        ["sample_id", "time", "event", GROUP_COLUMN, PROVENANCE_TARGET],
    ].copy()
    test_frame["latent_surrogate_risk"] = test_prediction

    gnn = _load_gnn_predictions(gnn_root, outer_group, "test")
    test_frame = test_frame.merge(
        gnn[["sample_id", "ensemble_risk"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(test_frame) != int(test_mask.sum()):
        raise RuntimeError(
            f"Outer group {outer_group} does not align with saved GNN predictions."
        )

    metrics = _prediction_metrics(
        latent=test_frame[PROVENANCE_TARGET].to_numpy(float),
        prediction=test_frame["latent_surrogate_risk"].to_numpy(float),
        time=test_frame["time"].to_numpy(float),
        event=test_frame["event"].to_numpy(float),
    )
    metrics.update(
        {
            "gnn_observed_outcome_c_index": float(
                concordance_index(
                    test_frame["time"],
                    test_frame["event"],
                    test_frame["ensemble_risk"],
                )
            ),
            "gnn_latent_spearman": _finite_correlation(
                spearmanr,
                test_frame[PROVENANCE_TARGET].to_numpy(float),
                test_frame["ensemble_risk"].to_numpy(float),
            ),
        }
    )
    result = {
        "outer_test_group": int(outer_group),
        "inner_validation_group": int(validation_group),
        "train_groups": sorted(
            frame.loc[train_mask, GROUP_COLUMN].astype(int).unique().tolist()
        ),
        "test_size": int(test_mask.sum()),
        "selected_surrogate": selected.name,
        "selection_metric": "validation_latent_r2_then_spearman",
        "validation_candidates": validation_scores,
        "test_metrics": metrics,
    }
    return result, test_frame


def _macro(rows: list[dict[str, Any]], metric: str) -> float:
    return float(
        np.mean([float(row["test_metrics"][metric]) for row in rows])
    )


def run_audit(
    *,
    config_path: Path = DEFAULT_CONFIG,
    gnn_root: Path = DEFAULT_GNN_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = 20260723,
) -> dict[str, Any]:
    frame, feature_columns, feature_metadata = _load_audit_frame(config_path)
    groups = sorted(frame[GROUP_COLUMN].astype(int).unique().tolist())
    if groups != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected generation groups 0..4, got {groups}.")

    folds: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for outer_group in groups:
        fold, fold_predictions = _fold_audit(
            frame,
            feature_columns,
            outer_group=outer_group,
            gnn_root=gnn_root,
            seed=seed,
        )
        folds.append(fold)
        predictions.append(fold_predictions)
        metrics = fold["test_metrics"]
        print(
            f"outer={outer_group} surrogate={fold['selected_surrogate']} "
            f"latent_r2={metrics['latent_r2']:.4f} "
            f"surrogate_c={metrics['observed_outcome_c_index']:.4f} "
            f"latent_c={metrics['direct_latent_observed_outcome_c_index']:.4f} "
            f"gnn_c={metrics['gnn_observed_outcome_c_index']:.4f}",
            flush=True,
        )

    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(frame) or oof["sample_id"].duplicated().any():
        raise RuntimeError("OOF audit predictions are not a one-to-one sample cover.")
    pooled_surrogate = _prediction_metrics(
        latent=oof[PROVENANCE_TARGET].to_numpy(float),
        prediction=oof["latent_surrogate_risk"].to_numpy(float),
        time=oof["time"].to_numpy(float),
        event=oof["event"].to_numpy(float),
    )
    pooled_gnn_c = float(
        concordance_index(
            oof["time"], oof["event"], oof["ensemble_risk"]
        )
    )
    pooled_gnn_latent_spearman = _finite_correlation(
        spearmanr,
        oof[PROVENANCE_TARGET].to_numpy(float),
        oof["ensemble_risk"].to_numpy(float),
    )
    direct_latent_c = float(
        pooled_surrogate["direct_latent_observed_outcome_c_index"]
    )
    macro_latent_c = _macro(
        folds, "direct_latent_observed_outcome_c_index"
    )
    macro_surrogate_c = _macro(folds, "observed_outcome_c_index")
    macro_gnn_c = _macro(folds, "gnn_observed_outcome_c_index")

    target_checks = {
        "direct_deterministic_latent_pooled_c_index_exceeds_target": bool(
            direct_latent_c > TARGET_C_INDEX
        ),
        "direct_deterministic_latent_macro_c_index_exceeds_target": bool(
            macro_latent_c > TARGET_C_INDEX
        ),
        "every_group_direct_latent_c_index_exceeds_target": bool(
            all(
                row["test_metrics"][
                    "direct_latent_observed_outcome_c_index"
                ]
                > TARGET_C_INDEX
                for row in folds
            )
        ),
        "current_locked_gnn_macro_c_index_exceeds_target": bool(
            macro_gnn_c > TARGET_C_INDEX
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "audit_only_not_a_deployable_model",
        "target_c_index": TARGET_C_INDEX,
        "dataset": "topology_v7_generator_v3_formal",
        "sample_count": int(len(frame)),
        "num_deployable_features": int(len(feature_columns)),
        "feature_metadata": feature_metadata,
        "prohibited_target": PROVENANCE_TARGET,
        "folds": folds,
        "aggregate": {
            "direct_deterministic_latent_macro_c_index": macro_latent_c,
            "direct_deterministic_latent_pooled_c_index": direct_latent_c,
            "latent_surrogate_macro_c_index": macro_surrogate_c,
            "latent_surrogate_pooled_c_index": float(
                pooled_surrogate["observed_outcome_c_index"]
            ),
            "latent_surrogate_pooled_latent_r2": float(
                pooled_surrogate["latent_r2"]
            ),
            "latent_surrogate_pooled_latent_spearman": float(
                pooled_surrogate["latent_spearman"]
            ),
            "locked_gnn_macro_c_index": macro_gnn_c,
            "locked_gnn_pooled_c_index": pooled_gnn_c,
            "locked_gnn_pooled_latent_spearman": pooled_gnn_latent_spearman,
            "model_approximation_gap_to_direct_latent_macro": float(
                macro_latent_c - macro_gnn_c
            ),
            "target_minus_direct_latent_pooled": float(
                TARGET_C_INDEX - direct_latent_c
            ),
            "target_minus_locked_gnn_macro": float(
                TARGET_C_INDEX - macro_gnn_c
            ),
        },
        "target_checks": target_checks,
        "decision": {
            "model_only_target_supported_by_generator_signal": bool(
                target_checks[
                    "direct_deterministic_latent_pooled_c_index_exceeds_target"
                ]
                and target_checks[
                    "direct_deterministic_latent_macro_c_index_exceeds_target"
                ]
            ),
            "strict_stable_target_requires_new_independent_outcome_information": bool(
                not target_checks[
                    "direct_deterministic_latent_pooled_c_index_exceeds_target"
                ]
                or not target_checks[
                    "direct_deterministic_latent_macro_c_index_exceeds_target"
                ]
            ),
            "warning": (
                "The deterministic latent-risk C-index is a generator-declared "
                "signal benchmark, not a mathematical upper bound. A fitted model "
                "can exceed it by finite-sample chance, but that is not evidence of "
                "stable independent performance."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "performance_ceiling_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    oof.drop(columns=[PROVENANCE_TARGET]).to_csv(
        output_dir / "audit_only_oof_predictions_without_latent_target.csv",
        index=False,
    )
    lines = [
        "# topology_v7 performance-ceiling audit",
        "",
        "This is an audit-only decomposition, not a deployable model result.",
        "",
        f"- Target C-index: `{TARGET_C_INDEX:.3f}`.",
        f"- Direct deterministic latent-risk pooled C-index: `{direct_latent_c:.6f}`.",
        f"- Direct deterministic latent-risk macro C-index: `{macro_latent_c:.6f}`.",
        f"- Locked GNN macro C-index: `{macro_gnn_c:.6f}`.",
        f"- Locked GNN pooled C-index: `{pooled_gnn_c:.6f}`.",
        f"- Audit surrogate pooled latent R2: `{pooled_surrogate['latent_r2']:.6f}`.",
        f"- Audit surrogate pooled latent Spearman: `{pooled_surrogate['latent_spearman']:.6f}`.",
        f"- Model approximation gap to direct latent macro: `{macro_latent_c - macro_gnn_c:+.6f}`.",
        "",
        "## Decision",
        "",
        (
            "The current generator's deterministic signal does not support a "
            "stable model-only claim above 0.761. New independently observed "
            "outcome information or a scientifically predeclared benchmark "
            "redesign is required."
            if summary["decision"][
                "strict_stable_target_requires_new_independent_outcome_information"
            ]
            else "The deterministic signal supports the target; continue model optimization."
        ),
        "",
        summary["decision"]["warning"],
    ]
    (output_dir / "performance_ceiling_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit recoverable deterministic signal in topology_v7."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gnn-root", type=Path, default=DEFAULT_GNN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    result = run_audit(
        config_path=args.config,
        gnn_root=args.gnn_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
