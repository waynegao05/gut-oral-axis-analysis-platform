from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from research.main_model_ctm_residual_v2 import export_main_model_structured_features
from research.metrics import concordance_index
from research.risk_adapter_diagnostics_v2 import (
    calibration_proxy,
    continuous_nri_proxy,
    pairwise_change_diagnostics,
)
from research.structured_ctm_oof_v2 import StructuredSplit, load_structured_splits
from research.train_v2 import resolve_device


DEFAULT_FEATURE_NPZ = "outputs/current_mainline_v2/main_model_ctm_residual_v2/main_model_structured_features.npz"
DEFAULT_OUTPUT_DIR = "outputs/current_mainline_v2/main_model_hardpair_residual_v3"


@dataclass(frozen=True)
class HardPairConfig:
    seed: int
    max_delta: float
    hard_pair_weight: float
    distillation_weight: float
    delta_l2_weight: float
    gate_l1_weight: float
    hard_margin: float

    @property
    def name(self) -> str:
        return (
            f"hardpair_seed{self.seed}"
            f"_d{_tag(self.max_delta)}"
            f"_hp{_tag(self.hard_pair_weight)}"
            f"_m{_tag(self.hard_margin)}"
        )


@dataclass(frozen=True)
class CandidateResult:
    config: HardPairConfig
    val_risk: np.ndarray
    test_risk: np.ndarray
    val_c_index: float
    test_c_index: float
    best_epoch: int


class HardPairGatedResidual(nn.Module):
    def __init__(self, input_dim: int, gate_dim: int, hidden_dim: int = 48, max_delta: float = 0.2) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -1.25)

    def forward(self, features: torch.Tensor, gate_features: torch.Tensor, baseline: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_delta = torch.tanh(self.residual(features).squeeze(-1))
        learned_gate = torch.sigmoid(self.gate(gate_features).squeeze(-1))
        # Preserve the main model on very confident samples; allow more correction around the decision boundary.
        confidence_gate = torch.sigmoid(1.5 - torch.abs(baseline))
        gate = learned_gate * confidence_gate
        delta = self.max_delta * gate * raw_delta
        return {"risk": baseline + delta, "delta": delta, "gate": gate}


def run_main_model_hardpair_residual_experiment(
    *,
    feature_npz: str | Path = DEFAULT_FEATURE_NPZ,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = (7, 21, 42, 123, 2026),
    max_deltas: Sequence[float] = (0.05, 0.10, 0.20, 0.35),
    hard_pair_weights: Sequence[float] = (0.05, 0.10, 0.20),
    hard_margins: Sequence[float] = (0.00, 0.10, 0.20),
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    epochs: int = 120,
    patience: int = 18,
    min_val_delta: float = 0.0,
    min_high_disagreement_val_delta: float = 0.0,
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
    train_x, val_x, test_x, train_gate, val_gate, test_gate, scaler = _build_feature_matrices(train, val, test)
    device = resolve_device(device_arg)
    configs = [
        HardPairConfig(
            seed=int(seed),
            max_delta=float(max_delta),
            hard_pair_weight=float(hard_weight),
            distillation_weight=0.10,
            delta_l2_weight=0.08,
            gate_l1_weight=0.01,
            hard_margin=float(margin),
        )
        for seed in seeds
        for max_delta in max_deltas
        for hard_weight in hard_pair_weights
        for margin in hard_margins
    ]
    candidates = [
        _fit_candidate(
            config=config,
            train=train,
            val=val,
            test=test,
            train_x=train_x,
            val_x=val_x,
            test_x=test_x,
            train_gate=train_gate,
            val_gate=val_gate,
            test_gate=test_gate,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        for config in configs
    ]
    selection = _select_candidate(
        val=val,
        test=test,
        candidates=candidates,
        alpha_grid=alpha_grid,
        min_val_delta=min_val_delta,
        min_high_disagreement_val_delta=min_high_disagreement_val_delta,
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
            **{key: value for key, value in selection.items() if key not in {"val_risk", "test_risk"}},
            "test_delta_vs_main": float(selection["test_c_index"] - concordance_index(test.time, test.event, test.baseline_risk)),
            "pair_change": pair_summary.__dict__,
            "calibration_proxy": {
                "main": calibration_proxy(time=test.time, event=test.event, risk=test.baseline_risk),
                "selected": calibration_proxy(time=test.time, event=test.event, risk=np.asarray(selection["test_risk"], dtype=float)),
            },
            "continuous_nri_proxy": continuous_nri_proxy(
                event=test.event,
                baseline_risk=test.baseline_risk,
                selected_risk=np.asarray(selection["test_risk"], dtype=float),
            ),
        },
        "feature_scaler": scaler,
        "candidate_count": len(candidates),
        "interpretation": (
            "Hard-pair v3 keeps the current research/ main model as the fallback risk and trains a small gated MLP "
            "only to correct near-tie or baseline-misordered survival pairs. Selection is validation-gated; alpha=0 "
            "exactly recovers the main model."
        ),
    }
    output_file = output_path / "main_model_hardpair_residual_v3_summary.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        "references": result["references"],
        "selected": {
            key: result["selected"][key]
            for key in (
                "candidate_name",
                "alpha",
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
    (output_path / "main_model_hardpair_residual_v3_compact_summary.json").write_text(
        json.dumps(compact, indent=2),
        encoding="utf-8",
    )
    return result


def _build_feature_matrices(
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_groups = ["risk_members", "risk_disagreement", "graph_embedding_mean", "graph_embedding_std", "latent_mean"]
    train_raw = np.hstack([train.groups[name] for name in feature_groups] + [train.baseline_risk[:, None]])
    val_raw = np.hstack([val.groups[name] for name in feature_groups] + [val.baseline_risk[:, None]])
    test_raw = np.hstack([test.groups[name] for name in feature_groups] + [test.baseline_risk[:, None]])
    means = train_raw.mean(axis=0, keepdims=True)
    stds = np.maximum(train_raw.std(axis=0, keepdims=True), 1e-6)
    train_x = (train_raw - means) / stds
    val_x = (val_raw - means) / stds
    test_x = (test_raw - means) / stds
    gate_train_raw = np.hstack([train.groups["risk_disagreement"], np.abs(train.baseline_risk[:, None])])
    gate_val_raw = np.hstack([val.groups["risk_disagreement"], np.abs(val.baseline_risk[:, None])])
    gate_test_raw = np.hstack([test.groups["risk_disagreement"], np.abs(test.baseline_risk[:, None])])
    gate_means = gate_train_raw.mean(axis=0, keepdims=True)
    gate_stds = np.maximum(gate_train_raw.std(axis=0, keepdims=True), 1e-6)
    return (
        train_x.astype(np.float32),
        ((val_raw - means) / stds).astype(np.float32),
        ((test_raw - means) / stds).astype(np.float32),
        ((gate_train_raw - gate_means) / gate_stds).astype(np.float32),
        ((gate_val_raw - gate_means) / gate_stds).astype(np.float32),
        ((gate_test_raw - gate_means) / gate_stds).astype(np.float32),
        {
            "feature_groups": feature_groups,
            "feature_means": means.squeeze(0).astype(float).tolist(),
            "feature_stds": stds.squeeze(0).astype(float).tolist(),
            "gate_means": gate_means.squeeze(0).astype(float).tolist(),
            "gate_stds": gate_stds.squeeze(0).astype(float).tolist(),
        },
    )


def _fit_candidate(
    *,
    config: HardPairConfig,
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_gate: np.ndarray,
    val_gate: np.ndarray,
    test_gate: np.ndarray,
    epochs: int,
    patience: int,
    device: torch.device,
) -> CandidateResult:
    _set_seed(config.seed)
    model = HardPairGatedResidual(train_x.shape[1], train_gate.shape[1], max_delta=config.max_delta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    train_gate_t = torch.tensor(train_gate, dtype=torch.float32, device=device)
    val_gate_t = torch.tensor(val_gate, dtype=torch.float32, device=device)
    train_base = torch.tensor(train.baseline_risk, dtype=torch.float32, device=device)
    val_base = torch.tensor(val.baseline_risk, dtype=torch.float32, device=device)
    train_time = torch.tensor(train.time, dtype=torch.float32, device=device)
    train_event = torch.tensor(train.event, dtype=torch.float32, device=device)
    hard_pair_mask = _hard_pair_mask(train.time, train.event, train.baseline_risk, config.hard_margin, device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x_t, train_gate_t, train_base)
        cox = _cox_loss(output["risk"], train_time, train_event)
        hard_pair = _hard_pair_loss(output["risk"], hard_pair_mask)
        distill = torch.mean((_zscore(output["risk"]) - _zscore(train_base.detach())) ** 2)
        delta_l2 = torch.mean(output["delta"].pow(2))
        gate_l1 = torch.mean(output["gate"])
        loss = (
            cox
            + config.hard_pair_weight * hard_pair
            + config.distillation_weight * distill
            + config.delta_l2_weight * delta_l2
            + config.gate_l1_weight * gate_l1
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_risk = model(val_x_t, val_gate_t, val_base)["risk"].detach().cpu().numpy()
        val_c = concordance_index(val.time, val.event, val_risk)
        if val_c > best_val + 0.00025:
            best_val = float(val_c)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    model.load_state_dict(best_state)
    val_risk = _predict(model, val_x, val_gate, val.baseline_risk, device)
    test_risk = _predict(model, test_x, test_gate, test.baseline_risk, device)
    return CandidateResult(
        config=config,
        val_risk=val_risk,
        test_risk=test_risk,
        val_c_index=concordance_index(val.time, val.event, val_risk),
        test_c_index=concordance_index(test.time, test.event, test_risk),
        best_epoch=best_epoch,
    )


def _select_candidate(
    *,
    val: StructuredSplit,
    test: StructuredSplit,
    candidates: Sequence[CandidateResult],
    alpha_grid: Sequence[float],
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
) -> dict[str, Any]:
    baseline_val_c = concordance_index(val.time, val.event, val.baseline_risk)
    baseline_test_c = concordance_index(test.time, test.event, test.baseline_risk)
    high_mask = _high_disagreement_mask(val)
    baseline_high_c = concordance_index(val.time[high_mask], val.event[high_mask], val.baseline_risk[high_mask])
    rows: list[dict[str, Any]] = [
        {
            "candidate_name": "main_model",
            "alpha": 0.0,
            "val_c_index": baseline_val_c,
            "val_delta": 0.0,
            "high_disagreement_val_delta": 0.0,
            "test_c_index": baseline_test_c,
            "eligible": True,
            "val_risk": val.baseline_risk,
            "test_risk": test.baseline_risk,
        }
    ]
    for candidate in candidates:
        for alpha in alpha_grid:
            alpha = float(alpha)
            val_risk = val.baseline_risk + alpha * (candidate.val_risk - val.baseline_risk)
            test_risk = test.baseline_risk + alpha * (candidate.test_risk - test.baseline_risk)
            val_c = concordance_index(val.time, val.event, val_risk)
            high_c = concordance_index(val.time[high_mask], val.event[high_mask], val_risk[high_mask])
            row = {
                "candidate_name": candidate.config.name,
                "alpha": alpha,
                "val_c_index": val_c,
                "val_delta": val_c - baseline_val_c,
                "high_disagreement_val_delta": high_c - baseline_high_c,
                "test_c_index": concordance_index(test.time, test.event, test_risk),
                "eligible": (
                    val_c - baseline_val_c >= float(min_val_delta)
                    and high_c - baseline_high_c >= float(min_high_disagreement_val_delta)
                ),
                "best_epoch": candidate.best_epoch,
                "val_risk": val_risk,
                "test_risk": test_risk,
            }
            rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["val_c_index"], row["high_disagreement_val_delta"]))
    return {
        "candidate_name": selected["candidate_name"],
        "alpha": float(selected["alpha"]),
        "val_c_index": float(selected["val_c_index"]),
        "val_delta": float(selected["val_delta"]),
        "high_disagreement_val_delta": float(selected["high_disagreement_val_delta"]),
        "test_c_index": float(selected["test_c_index"]),
        "candidate_table": [
            {key: value for key, value in row.items() if key not in {"val_risk", "test_risk"}}
            for row in rows
        ],
        "val_risk": selected["val_risk"],
        "test_risk": selected["test_risk"],
    }


def _hard_pair_mask(
    time: np.ndarray,
    event: np.ndarray,
    baseline_risk: np.ndarray,
    hard_margin: float,
    device: torch.device,
) -> torch.Tensor:
    time_t = torch.tensor(time, dtype=torch.float32, device=device)
    event_t = torch.tensor(event, dtype=torch.bool, device=device)
    base_t = torch.tensor(baseline_risk, dtype=torch.float32, device=device)
    permissible = event_t[:, None] & (time_t[:, None] < time_t[None, :])
    baseline_diff = base_t[:, None] - base_t[None, :]
    hard = permissible & (baseline_diff < float(hard_margin))
    if not torch.any(hard):
        hard = permissible
    return hard


def _hard_pair_loss(risk: torch.Tensor, hard_pair_mask: torch.Tensor) -> torch.Tensor:
    diff = risk[:, None] - risk[None, :]
    return torch.nn.functional.softplus(-diff[hard_pair_mask]).mean()


def _cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    sorted_risk = risk[order]
    sorted_event = event[order]
    log_cumsum = torch.logcumsumexp(sorted_risk, dim=0)
    observed = sorted_event > 0
    if not torch.any(observed):
        return torch.zeros((), dtype=risk.dtype, device=risk.device)
    return -torch.mean(sorted_risk[observed] - log_cumsum[observed])


def _predict(
    model: HardPairGatedResidual,
    features: np.ndarray,
    gate_features: np.ndarray,
    baseline_risk: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(gate_features, dtype=torch.float32, device=device),
            torch.tensor(baseline_risk, dtype=torch.float32, device=device),
        )
    return output["risk"].detach().cpu().numpy().astype(float)


def _high_disagreement_mask(split: StructuredSplit) -> np.ndarray:
    score = np.asarray(split.groups["risk_disagreement"][:, 0], dtype=float)
    threshold = float(np.quantile(score, 0.75))
    return score >= threshold


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-npz", default=DEFAULT_FEATURE_NPZ)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="7,21,42,123,2026")
    parser.add_argument("--max-deltas", default="0.05,0.1,0.2,0.35")
    parser.add_argument("--hard-pair-weights", default="0.05,0.1,0.2")
    parser.add_argument("--hard-margins", default="0.0,0.1,0.2")
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-val-delta", type=float, default=0.0)
    parser.add_argument("--min-high-disagreement-val-delta", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--no-ensure-features", action="store_true")
    args = parser.parse_args()
    result = run_main_model_hardpair_residual_experiment(
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        seeds=_parse_ints(args.seeds),
        max_deltas=_parse_floats(args.max_deltas),
        hard_pair_weights=_parse_floats(args.hard_pair_weights),
        hard_margins=_parse_floats(args.hard_margins),
        alpha_grid=_parse_floats(args.alpha_grid),
        epochs=args.epochs,
        patience=args.patience,
        min_val_delta=args.min_val_delta,
        min_high_disagreement_val_delta=args.min_high_disagreement_val_delta,
        device_arg=args.device,
        ensure_features=not args.no_ensure_features,
    )
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
