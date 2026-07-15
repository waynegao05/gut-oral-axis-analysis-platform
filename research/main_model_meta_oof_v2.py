from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from research.main_model_ctm_residual_v2 import export_main_model_structured_features
from research.metrics import concordance_index
from research.risk_adapter_diagnostics_v2 import pairwise_change_diagnostics
from research.structured_ctm_oof_v2 import StructuredSplit, load_structured_splits
from research.train_v2 import resolve_device


DEFAULT_FEATURE_NPZ = "outputs/current_mainline_v2/main_model_ctm_residual_v2/main_model_structured_features.npz"
DEFAULT_OUTPUT_DIR = "outputs/current_mainline_v2/main_model_meta_oof_v2"
HYBRID_OOF_DELTA_WEIGHT = 0.25
SELECTION_POLICIES = ("validation_then_oof", "oof_then_validation", "hybrid_validation_oof")


@dataclass(frozen=True)
class MetaConfig:
    name: str
    seed: int
    model_type: str
    max_delta: float
    distillation_weight: float
    delta_l2_weight: float
    dropout: float


@dataclass(frozen=True)
class MetaCandidate:
    config: MetaConfig
    oof_risk: np.ndarray
    val_risk: np.ndarray
    test_risk: np.ndarray
    oof_c_index: float
    val_c_index: float
    test_c_index: float
    best_epochs: list[int]
    final_best_epoch: int


class LinearMetaResidual(nn.Module):
    def __init__(self, input_dim: int, max_delta: float) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor, baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.max_delta * torch.tanh(self.linear(features).squeeze(-1))
        return baseline + delta, delta


class MlpMetaResidual(nn.Module):
    def __init__(self, input_dim: int, max_delta: float, dropout: float) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(48, 24),
            nn.LayerNorm(24),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(24, 1),
        )

    def forward(self, features: torch.Tensor, baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.max_delta * torch.tanh(self.net(features).squeeze(-1))
        return baseline + delta, delta


def run_main_model_meta_oof_experiment(
    *,
    feature_npz: str | Path = DEFAULT_FEATURE_NPZ,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    model_types: Sequence[str] = ("linear", "mlp"),
    max_deltas: Sequence[float] = (0.03, 0.05, 0.10, 0.20),
    inner_folds: int = 5,
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    epochs: int = 120,
    patience: int = 18,
    min_oof_delta: float = 0.0,
    min_val_delta: float = 0.0,
    min_high_disagreement_val_delta: float = 0.0,
    selection_policy: str = "validation_then_oof",
    device_arg: str = "cuda",
    ensure_features: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_path = Path(feature_npz)
    if ensure_features and not feature_path.exists():
        export_main_model_structured_features(
            config_path="research_config_v2.yaml",
            checkpoint_glob="outputs/current_mainline_v2/cox_fixed_split_repeat/research_seed*/best_model.pt",
            structured_feature_npz="outputs/current_mainline_v2/structured_features_v2/structured_gnn_features_v2_all_splits.npz",
            split_seed=42,
            device_arg=device_arg,
            adapter_device_arg="cpu",
            output_npz_path=feature_path,
            output_json_path=feature_path.with_name("main_model_structured_features_summary.json"),
        )
    splits = load_structured_splits(feature_path)
    train, val, test = splits["train"], splits["val"], splits["test"]
    train_x, val_x, test_x, scaler = _build_meta_features(train, val, test)
    device = resolve_device(device_arg)
    configs = [
        MetaConfig(
            name=f"meta_{model_type}_seed{int(seed)}_d{_tag(max_delta)}",
            seed=int(seed),
            model_type=str(model_type),
            max_delta=float(max_delta),
            distillation_weight=0.10,
            delta_l2_weight=0.08,
            dropout=0.20,
        )
        for model_type in model_types
        for seed in seeds
        for max_delta in max_deltas
    ]
    candidates = [
        _evaluate_candidate_oof(
            config=config,
            train=train,
            val=val,
            test=test,
            train_x=train_x,
            val_x=val_x,
            test_x=test_x,
            inner_folds=inner_folds,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        for config in configs
    ]
    selection = _select_candidate(
        train=train,
        val=val,
        test=test,
        candidates=candidates,
        alpha_grid=alpha_grid,
        min_oof_delta=min_oof_delta,
        min_val_delta=min_val_delta,
        min_high_disagreement_val_delta=min_high_disagreement_val_delta,
        selection_policy=selection_policy,
    )
    pair_summary, _ = pairwise_change_diagnostics(
        sample_ids=test.sample_ids,
        time=test.time,
        event=test.event,
        baseline_risk=test.baseline_risk,
        selected_risk=np.asarray(selection["test_risk"], dtype=float),
    )
    result = {
        "references": {
            "train_main_c_index": concordance_index(train.time, train.event, train.baseline_risk),
            "val_main_c_index": concordance_index(val.time, val.event, val.baseline_risk),
            "test_main_c_index": concordance_index(test.time, test.event, test.baseline_risk),
        },
        "selected": {
            **{key: value for key, value in selection.items() if key not in {"oof_risk", "val_risk", "test_risk"}},
            "test_delta_vs_main": float(selection["test_c_index"] - concordance_index(test.time, test.event, test.baseline_risk)),
            "pair_change": pair_summary.__dict__,
        },
        "feature_scaler": scaler,
        "candidate_count": len(candidates),
        "selection_policy": selection_policy,
        "interpretation": (
            "Main-model meta OOF v2 trains small linear/MLP residual learners after the current research/ main model. "
            "Train inner-fold OOF is used as a non-inferiority gate; the default selector ranks eligible candidates "
            "by validation c-index because this fixed-split mainline uses validation as the locked model-selection split."
        ),
    }
    (output_path / "main_model_meta_oof_v2_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        "references": result["references"],
        "selected": {
            key: result["selected"][key]
            for key in (
                "candidate_name",
                "alpha",
                "oof_c_index",
                "oof_delta",
                "val_c_index",
                "val_delta",
                "high_disagreement_val_delta",
                "test_c_index",
                "test_delta_vs_main",
            )
        },
        "pair_change": result["selected"]["pair_change"],
        "candidate_count": len(candidates),
    }
    (output_path / "main_model_meta_oof_v2_compact_summary.json").write_text(
        json.dumps(compact, indent=2),
        encoding="utf-8",
    )
    return result


def _build_meta_features(
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    groups = ["risk_members", "risk_disagreement", "graph_embedding_mean", "graph_embedding_std", "latent_mean"]
    train_raw = np.hstack([train.groups[name] for name in groups] + [train.baseline_risk[:, None]])
    val_raw = np.hstack([val.groups[name] for name in groups] + [val.baseline_risk[:, None]])
    test_raw = np.hstack([test.groups[name] for name in groups] + [test.baseline_risk[:, None]])
    means = train_raw.mean(axis=0, keepdims=True)
    stds = np.maximum(train_raw.std(axis=0, keepdims=True), 1e-6)
    return (
        ((train_raw - means) / stds).astype(np.float32),
        ((val_raw - means) / stds).astype(np.float32),
        ((test_raw - means) / stds).astype(np.float32),
        {
            "groups": groups,
            "means": means.squeeze(0).astype(float).tolist(),
            "stds": stds.squeeze(0).astype(float).tolist(),
        },
    )


def _evaluate_candidate_oof(
    *,
    config: MetaConfig,
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    inner_folds: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> MetaCandidate:
    labels = train.event.astype(int)
    splitter = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(config.seed))
    oof_risk = np.zeros(len(train.sample_ids), dtype=float)
    best_epochs: list[int] = []
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(np.arange(len(labels)), labels), start=1):
        model, best_epoch = _fit_model(
            config=config,
            train=train,
            train_x=train_x,
            fit_indices=fit_indices,
            eval_indices=holdout_indices,
            epochs=epochs,
            patience=patience,
            seed=config.seed + fold * 1000,
            device=device,
        )
        best_epochs.append(best_epoch)
        oof_risk[holdout_indices] = _predict(model, train_x[holdout_indices], train.baseline_risk[holdout_indices], device)
    final_model, final_best_epoch = _fit_model(
        config=config,
        train=train,
        train_x=train_x,
        fit_indices=np.arange(len(train.sample_ids)),
        eval_indices=np.arange(len(val.sample_ids)),
        eval_split=val,
        eval_x=val_x,
        epochs=epochs,
        patience=patience,
        seed=config.seed + 999000,
        device=device,
    )
    val_risk = _predict(final_model, val_x, val.baseline_risk, device)
    test_risk = _predict(final_model, test_x, test.baseline_risk, device)
    return MetaCandidate(
        config=config,
        oof_risk=oof_risk,
        val_risk=val_risk,
        test_risk=test_risk,
        oof_c_index=concordance_index(train.time, train.event, oof_risk),
        val_c_index=concordance_index(val.time, val.event, val_risk),
        test_c_index=concordance_index(test.time, test.event, test_risk),
        best_epochs=best_epochs,
        final_best_epoch=final_best_epoch,
    )


def _fit_model(
    *,
    config: MetaConfig,
    train: StructuredSplit,
    train_x: np.ndarray,
    fit_indices: np.ndarray,
    eval_indices: np.ndarray,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    eval_split: StructuredSplit | None = None,
    eval_x: np.ndarray | None = None,
) -> tuple[nn.Module, int]:
    _set_seed(seed)
    eval_split = train if eval_split is None else eval_split
    eval_x = train_x if eval_x is None else eval_x
    model = _build_model(config, train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
    fit_x = torch.tensor(train_x[fit_indices], dtype=torch.float32, device=device)
    fit_base = torch.tensor(train.baseline_risk[fit_indices], dtype=torch.float32, device=device)
    fit_time = torch.tensor(train.time[fit_indices], dtype=torch.float32, device=device)
    fit_event = torch.tensor(train.event[fit_indices], dtype=torch.float32, device=device)
    eval_x_t = torch.tensor(eval_x[eval_indices], dtype=torch.float32, device=device)
    eval_base = torch.tensor(eval_split.baseline_risk[eval_indices], dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_eval = float("-inf")
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, delta = model(fit_x, fit_base)
        loss = (
            _cox_loss(risk, fit_time, fit_event)
            + config.distillation_weight * torch.mean((_zscore(risk) - _zscore(fit_base.detach())) ** 2)
            + config.delta_l2_weight * torch.mean(delta.pow(2))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_risk, _ = model(eval_x_t, eval_base)
        eval_c = concordance_index(
            eval_split.time[eval_indices],
            eval_split.event[eval_indices],
            eval_risk.detach().cpu().numpy(),
        )
        if eval_c > best_eval + 0.00025:
            best_eval = float(eval_c)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def _build_model(config: MetaConfig, input_dim: int) -> nn.Module:
    if config.model_type == "linear":
        return LinearMetaResidual(input_dim, config.max_delta)
    if config.model_type == "mlp":
        return MlpMetaResidual(input_dim, config.max_delta, config.dropout)
    raise ValueError(f"Unknown model_type: {config.model_type}")


def _select_candidate(
    *,
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
    candidates: Sequence[MetaCandidate],
    alpha_grid: Sequence[float],
    min_oof_delta: float,
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
    selection_policy: str,
) -> dict[str, Any]:
    baseline_oof_c = concordance_index(train.time, train.event, train.baseline_risk)
    baseline_val_c = concordance_index(val.time, val.event, val.baseline_risk)
    baseline_test_c = concordance_index(test.time, test.event, test.baseline_risk)
    high_mask = _high_disagreement_mask(val)
    baseline_high_c = concordance_index(val.time[high_mask], val.event[high_mask], val.baseline_risk[high_mask])
    rows: list[dict[str, Any]] = [
        {
            "candidate_name": "main_model",
            "alpha": 0.0,
            "oof_c_index": baseline_oof_c,
            "oof_delta": 0.0,
            "val_c_index": baseline_val_c,
            "val_delta": 0.0,
            "high_disagreement_val_delta": 0.0,
            "test_c_index": baseline_test_c,
            "eligible": True,
            "oof_risk": train.baseline_risk,
            "val_risk": val.baseline_risk,
            "test_risk": test.baseline_risk,
        }
    ]
    for candidate in candidates:
        for alpha in alpha_grid:
            alpha = float(alpha)
            oof_risk = train.baseline_risk + alpha * (candidate.oof_risk - train.baseline_risk)
            val_risk = val.baseline_risk + alpha * (candidate.val_risk - val.baseline_risk)
            test_risk = test.baseline_risk + alpha * (candidate.test_risk - test.baseline_risk)
            oof_c = concordance_index(train.time, train.event, oof_risk)
            val_c = concordance_index(val.time, val.event, val_risk)
            high_c = concordance_index(val.time[high_mask], val.event[high_mask], val_risk[high_mask])
            rows.append(
                {
                    "candidate_name": candidate.config.name,
                    "alpha": alpha,
                    "oof_c_index": oof_c,
                    "oof_delta": oof_c - baseline_oof_c,
                    "val_c_index": val_c,
                    "val_delta": val_c - baseline_val_c,
                    "high_disagreement_val_delta": high_c - baseline_high_c,
                    "test_c_index": concordance_index(test.time, test.event, test_risk),
                    "eligible": (
                        oof_c - baseline_oof_c >= float(min_oof_delta)
                        and val_c - baseline_val_c >= float(min_val_delta)
                        and high_c - baseline_high_c >= float(min_high_disagreement_val_delta)
                    ),
                    "best_epochs": candidate.best_epochs,
                    "final_best_epoch": candidate.final_best_epoch,
                    "oof_risk": oof_risk,
                    "val_risk": val_risk,
                    "test_risk": test_risk,
                }
            )
    eligible = [row for row in rows if row["eligible"]]
    if selection_policy == "oof_then_validation":
        selected = max(eligible, key=lambda row: (row["oof_c_index"], row["val_c_index"]))
    elif selection_policy == "validation_then_oof":
        selected = max(
            eligible,
            key=lambda row: (row["val_c_index"], row["high_disagreement_val_delta"], row["oof_c_index"]),
        )
    elif selection_policy == "hybrid_validation_oof":
        selected = max(
            eligible,
            key=lambda row: (
                row["val_delta"] + HYBRID_OOF_DELTA_WEIGHT * row["oof_delta"],
                row["val_c_index"],
                row["oof_c_index"],
                row["high_disagreement_val_delta"],
            ),
        )
    else:
        raise ValueError(f"Unknown selection_policy: {selection_policy}")
    return {
        "candidate_name": selected["candidate_name"],
        "alpha": float(selected["alpha"]),
        "oof_c_index": float(selected["oof_c_index"]),
        "oof_delta": float(selected["oof_delta"]),
        "val_c_index": float(selected["val_c_index"]),
        "val_delta": float(selected["val_delta"]),
        "high_disagreement_val_delta": float(selected["high_disagreement_val_delta"]),
        "test_c_index": float(selected["test_c_index"]),
        "candidate_table": [
            {key: value for key, value in row.items() if key not in {"oof_risk", "val_risk", "test_risk"}}
            for row in rows
        ],
        "oof_risk": selected["oof_risk"],
        "val_risk": selected["val_risk"],
        "test_risk": selected["test_risk"],
    }


def _predict(model: nn.Module, features: np.ndarray, baseline: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        risk, _ = model(
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(baseline, dtype=torch.float32, device=device),
        )
    return risk.detach().cpu().numpy().astype(float)


def _cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    sorted_risk = risk[order]
    sorted_event = event[order]
    log_cumsum = torch.logcumsumexp(sorted_risk, dim=0)
    observed = sorted_event > 0
    if not torch.any(observed):
        return torch.zeros((), dtype=risk.dtype, device=risk.device)
    return -torch.mean(sorted_risk[observed] - log_cumsum[observed])


def _high_disagreement_mask(split: StructuredSplit) -> np.ndarray:
    score = np.asarray(split.groups["risk_disagreement"][:, 0], dtype=float)
    return score >= float(np.quantile(score, 0.75))


def _zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values - values.mean()) / torch.clamp(values.std(unbiased=False), min=float(eps))


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tag(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-npz", default=DEFAULT_FEATURE_NPZ)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="7,21,42,123,2026")
    parser.add_argument("--model-types", default="linear,mlp")
    parser.add_argument("--max-deltas", default="0.03,0.05,0.1,0.2")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-oof-delta", type=float, default=0.0)
    parser.add_argument("--min-val-delta", type=float, default=0.0)
    parser.add_argument("--min-high-disagreement-val-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="validation_then_oof",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--no-ensure-features", action="store_true")
    args = parser.parse_args()
    result = run_main_model_meta_oof_experiment(
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        seeds=_parse_ints(args.seeds),
        model_types=_parse_strings(args.model_types),
        max_deltas=_parse_floats(args.max_deltas),
        inner_folds=args.inner_folds,
        alpha_grid=_parse_floats(args.alpha_grid),
        epochs=args.epochs,
        patience=args.patience,
        min_oof_delta=args.min_oof_delta,
        min_val_delta=args.min_val_delta,
        min_high_disagreement_val_delta=args.min_high_disagreement_val_delta,
        selection_policy=args.selection_policy,
        device_arg=args.device,
        ensure_features=not args.no_ensure_features,
    )
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
