from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from ctm_fusion_experiment.models.ctm import CTM
from research.losses import cox_ph_loss
from research.metrics import concordance_index
from research.risk_adapter_diagnostics_v2 import (
    calibration_proxy,
    continuous_nri_proxy,
    pairwise_change_diagnostics,
)
from research.train_v2 import resolve_device


GROUP_NAMES = [
    "risk_members",
    "risk_disagreement",
    "graph_embedding_mean",
    "graph_embedding_std",
    "latent_mean",
    "latent_std",
    "topology_targets",
]


@dataclass(frozen=True)
class StructuredSplit:
    sample_ids: list[str]
    time: np.ndarray
    event: np.ndarray
    baseline_risk: np.ndarray
    groups: dict[str, np.ndarray]


@dataclass(frozen=True)
class CandidateConfig:
    seed: int
    max_delta: float
    distillation_weight: float
    delta_l2_weight: float
    disagreement_l2_weight: float

    @property
    def name(self) -> str:
        return (
            f"structured_ctm_seed{self.seed}"
            f"_d{_tag_float(self.max_delta)}"
            f"_dist{_tag_float(self.distillation_weight)}"
            f"_l2{_tag_float(self.delta_l2_weight)}"
            f"_safe{_tag_float(self.disagreement_l2_weight)}"
        )


@dataclass(frozen=True)
class CandidateResult:
    name: str
    config: CandidateConfig
    oof_risk: np.ndarray
    val_risk: np.ndarray
    test_risk: np.ndarray
    oof_c_index: float
    val_c_index: float
    test_c_index: float
    best_epochs: list[int]
    final_best_epoch: int


class StructuredCTMResidualAdapter(nn.Module):
    def __init__(
        self,
        group_dims: dict[str, int],
        *,
        d_input: int = 32,
        d_model: int = 32,
        iterations: int = 3,
        memory_length: int = 4,
        nlm_hidden_dim: int = 16,
        n_heads: int = 4,
        n_synch_action: int = 16,
        n_synch_out: int = 16,
        n_self_pairs: int = 4,
        synapse_depth: int = 1,
        dropout: float = 0.20,
        max_delta: float = 0.35,
    ) -> None:
        super().__init__()
        self.group_names = [name for name in GROUP_NAMES if name in group_dims]
        self.max_delta = float(max_delta)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(group_dims[name], d_input),
                    nn.LayerNorm(d_input),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in self.group_names
            }
        )
        self.group_embedding = nn.Parameter(torch.zeros(len(self.group_names), d_input))
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)
        self.ctm = CTM(
            d_model=d_model,
            d_input=d_input,
            iterations=iterations,
            memory_length=memory_length,
            nlm_hidden_dim=nlm_hidden_dim,
            n_heads=n_heads,
            n_synch_action=n_synch_action,
            n_synch_out=n_synch_out,
            n_self_pairs=n_self_pairs,
            synapse_depth=synapse_depth,
            dropout=dropout,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(n_synch_out),
            nn.Linear(n_synch_out, 1),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(group_dims["risk_disagreement"], 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        nn.init.constant_(self.gate_network[-1].bias, -1.5)

    def forward(self, groups: dict[str, torch.Tensor], baseline_risk: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = torch.stack(
            [self.projections[name](groups[name]) for name in self.group_names],
            dim=1,
        )
        tokens = tokens + self.group_embedding.unsqueeze(0)
        ctm_output = self.ctm(tokens, track_attention=False)
        representation = ctm_output.representations[:, -1, :]
        raw_delta = self.max_delta * torch.tanh(self.delta_head(representation).squeeze(-1))
        disagreement = groups["risk_disagreement"]
        learned_gate = torch.sigmoid(self.gate_network(disagreement).squeeze(-1))
        # Positive standardized disagreement means ensemble members disagree more than the training mean.
        safety_gate = torch.sigmoid(-1.25 * disagreement[:, 0])
        gate = learned_gate * safety_gate
        delta = gate * raw_delta
        return {
            "risk": baseline_risk + delta,
            "delta": delta,
            "gate": gate,
            "raw_delta": raw_delta,
        }


def run_structured_ctm_oof_experiment(
    *,
    feature_npz_path: str | Path,
    inner_folds: int = 5,
    seeds: Sequence[int] = (7, 21, 42),
    max_deltas: Sequence[float] = (0.20, 0.35, 0.50),
    distillation_weights: Sequence[float] = (0.10,),
    delta_l2_weights: Sequence[float] = (0.08,),
    disagreement_l2_weights: Sequence[float] = (0.25,),
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    epochs: int = 100,
    patience: int = 14,
    min_oof_delta: float = 0.0,
    min_val_delta: float = -0.00025,
    min_high_disagreement_val_delta: float = 0.0,
    device_arg: str = "cuda",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_arg)
    splits = load_structured_splits(feature_npz_path)
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    candidate_configs = [
        CandidateConfig(
            seed=int(seed),
            max_delta=float(max_delta),
            distillation_weight=float(distillation_weight),
            delta_l2_weight=float(delta_l2_weight),
            disagreement_l2_weight=float(disagreement_l2_weight),
        )
        for seed in seeds
        for max_delta in max_deltas
        for distillation_weight in distillation_weights
        for delta_l2_weight in delta_l2_weights
        for disagreement_l2_weight in disagreement_l2_weights
    ]
    candidates = [
        _evaluate_candidate_oof(
            config=config,
            train=train,
            val=val,
            test=test,
            inner_folds=inner_folds,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        for config in candidate_configs
    ]
    selection = _select_candidate_with_alpha(
        train=train,
        val=val,
        test=test,
        candidates=candidates,
        alpha_grid=alpha_grid,
        min_oof_delta=min_oof_delta,
        min_val_delta=min_val_delta,
        min_high_disagreement_val_delta=min_high_disagreement_val_delta,
    )
    baseline_test = concordance_index(test.time, test.event, test.baseline_risk)
    selected_test = selection["test_c_index"]
    pair_summary, _ = pairwise_change_diagnostics(
        sample_ids=test.sample_ids,
        time=test.time,
        event=test.event,
        baseline_risk=test.baseline_risk,
        selected_risk=np.asarray(selection["test_risk"], dtype=float),
    )
    result = {
        "feature_npz_path": str(feature_npz_path),
        "device": str(device),
        "inner_folds": int(inner_folds),
        "candidate_grid": {
            "seeds": [int(value) for value in seeds],
            "max_deltas": [float(value) for value in max_deltas],
            "distillation_weights": [float(value) for value in distillation_weights],
            "delta_l2_weights": [float(value) for value in delta_l2_weights],
            "disagreement_l2_weights": [float(value) for value in disagreement_l2_weights],
            "alpha_grid": [float(value) for value in alpha_grid],
            "candidate_count": len(candidates),
        },
        "references": {
            "train_baseline_c_index": concordance_index(train.time, train.event, train.baseline_risk),
            "val_baseline_c_index": concordance_index(val.time, val.event, val.baseline_risk),
            "test_baseline_c_index": baseline_test,
        },
        "selected": {
            **{key: value for key, value in selection.items() if key not in {"val_risk", "test_risk", "oof_risk"}},
            "test_delta_vs_baseline": selected_test - baseline_test,
            "pair_change": pair_summary.__dict__,
            "calibration_proxy": {
                "baseline": calibration_proxy(time=test.time, event=test.event, risk=test.baseline_risk),
                "selected": calibration_proxy(time=test.time, event=test.event, risk=np.asarray(selection["test_risk"])),
            },
            "continuous_nri_proxy": continuous_nri_proxy(
                event=test.event,
                baseline_risk=test.baseline_risk,
                selected_risk=np.asarray(selection["test_risk"], dtype=float),
            ),
        },
        "candidates": [
            {
                "name": candidate.name,
                "oof_c_index": candidate.oof_c_index,
                "val_c_index": candidate.val_c_index,
                "test_c_index": candidate.test_c_index,
                "best_epochs": candidate.best_epochs,
                "final_best_epoch": candidate.final_best_epoch,
                "config": candidate.config.__dict__,
            }
            for candidate in candidates
        ],
        "test_predictions": [
            {
                "sample_id": sample_id,
                "time": float(time),
                "event": float(event),
                "baseline_risk": float(baseline),
                "selected_risk": float(selected),
            }
            for sample_id, time, event, baseline, selected in zip(
                test.sample_ids,
                test.time,
                test.event,
                test.baseline_risk,
                selection["test_risk"],
            )
        ],
        "interpretation": (
            "Structured CTM residual candidates are selected primarily by train inner-fold OOF c-index. "
            "The outer validation split is used as a safety gate, including high-disagreement non-inferiority; "
            "alpha=0 fallback is always available."
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    return _jsonable(result)


def load_structured_splits(feature_npz_path: str | Path) -> dict[str, StructuredSplit]:
    data = np.load(feature_npz_path)
    return {split: _load_split(data, split) for split in ("train", "val", "test")}


def _load_split(data: np.lib.npyio.NpzFile, split: str) -> StructuredSplit:
    groups = {
        "risk_members": np.asarray(data[f"{split}_standardized_risk_matrix"], dtype=float).T,
        "risk_disagreement": np.asarray(data[f"{split}_risk_disagreement"], dtype=float),
        "graph_embedding_mean": np.asarray(data[f"{split}_graph_embedding_mean"], dtype=float),
        "graph_embedding_std": np.asarray(data[f"{split}_graph_embedding_std"], dtype=float),
        "latent_mean": np.asarray(data[f"{split}_latent_mean"], dtype=float),
        "latent_std": np.asarray(data[f"{split}_latent_std"], dtype=float),
        "topology_targets": np.hstack(
            [
                np.asarray(data[f"{split}_graph_target_mean"], dtype=float),
                np.asarray(data[f"{split}_graph_cluster_target_mean"], dtype=float),
            ]
        ),
    }
    return StructuredSplit(
        sample_ids=[str(value) for value in data[f"{split}_sample_ids"].tolist()],
        time=np.asarray(data[f"{split}_time"], dtype=float),
        event=np.asarray(data[f"{split}_event"], dtype=float),
        baseline_risk=np.asarray(data[f"{split}_standardized_topk_risk"], dtype=float),
        groups=groups,
    )


def _evaluate_candidate_oof(
    *,
    config: CandidateConfig,
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
    inner_folds: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> CandidateResult:
    _set_seed(config.seed)
    labels = train.event.astype(int)
    splitter = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(config.seed))
    oof_risk = np.zeros(len(train.sample_ids), dtype=float)
    best_epochs: list[int] = []
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(np.arange(len(labels)), labels), start=1):
        scaler = _fit_group_scaler(train.groups, fit_indices)
        model, best_epoch = _fit_model(
            config=config,
            train_split=train,
            train_indices=fit_indices,
            eval_split=train,
            eval_indices=holdout_indices,
            scaler=scaler,
            epochs=epochs,
            patience=patience,
            device=device,
            seed=config.seed + fold * 1000,
        )
        oof_risk[holdout_indices] = _predict_model(model, train, holdout_indices, scaler, device)
        best_epochs.append(best_epoch)

    final_scaler = _fit_group_scaler(train.groups, np.arange(len(train.sample_ids)))
    final_model, final_best_epoch = _fit_model(
        config=config,
        train_split=train,
        train_indices=np.arange(len(train.sample_ids)),
        eval_split=val,
        eval_indices=np.arange(len(val.sample_ids)),
        scaler=final_scaler,
        epochs=epochs,
        patience=patience,
        device=device,
        seed=config.seed + 900000,
    )
    val_risk = _predict_model(final_model, val, np.arange(len(val.sample_ids)), final_scaler, device)
    test_risk = _predict_model(final_model, test, np.arange(len(test.sample_ids)), final_scaler, device)
    return CandidateResult(
        name=config.name,
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
    config: CandidateConfig,
    train_split: StructuredSplit,
    train_indices: np.ndarray,
    eval_split: StructuredSplit,
    eval_indices: np.ndarray,
    scaler: dict[str, tuple[np.ndarray, np.ndarray]],
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> tuple[StructuredCTMResidualAdapter, int]:
    _set_seed(seed)
    group_dims = {name: train_split.groups[name].shape[1] for name in GROUP_NAMES}
    model = StructuredCTMResidualAdapter(group_dims, max_delta=config.max_delta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    train_groups = _groups_to_tensors(train_split, train_indices, scaler, device)
    eval_groups = _groups_to_tensors(eval_split, eval_indices, scaler, device)
    train_base = torch.tensor(train_split.baseline_risk[train_indices], dtype=torch.float32, device=device)
    eval_base = torch.tensor(eval_split.baseline_risk[eval_indices], dtype=torch.float32, device=device)
    train_time = torch.tensor(train_split.time[train_indices], dtype=torch.float32, device=device)
    train_event = torch.tensor(train_split.event[train_indices], dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_eval = float("-inf")
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(train_groups, train_base)
        cox = cox_ph_loss(output["risk"], train_time, train_event)
        distill = torch.mean((_zscore_torch(output["risk"]) - _zscore_torch(train_base.detach())) ** 2)
        delta_l2 = torch.mean(output["delta"].pow(2))
        # Penalize residual magnitude more when member disagreement is above the train mean.
        disagreement_weight = torch.relu(train_groups["risk_disagreement"][:, 0])
        disagreement_l2 = torch.mean(disagreement_weight * output["delta"].pow(2))
        loss = (
            cox
            + config.distillation_weight * distill
            + config.delta_l2_weight * delta_l2
            + config.disagreement_l2_weight * disagreement_l2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_risk = model(eval_groups, eval_base)["risk"].detach().cpu().numpy()
        eval_c = concordance_index(eval_split.time[eval_indices], eval_split.event[eval_indices], eval_risk)
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


def _predict_model(
    model: StructuredCTMResidualAdapter,
    split: StructuredSplit,
    indices: np.ndarray,
    scaler: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        groups = _groups_to_tensors(split, indices, scaler, device)
        baseline = torch.tensor(split.baseline_risk[indices], dtype=torch.float32, device=device)
        return model(groups, baseline)["risk"].detach().cpu().numpy().astype(float)


def _select_candidate_with_alpha(
    *,
    train: StructuredSplit,
    val: StructuredSplit,
    test: StructuredSplit,
    candidates: Sequence[CandidateResult],
    alpha_grid: Sequence[float],
    min_oof_delta: float,
    min_val_delta: float,
    min_high_disagreement_val_delta: float,
) -> dict[str, Any]:
    baseline_oof_c = concordance_index(train.time, train.event, train.baseline_risk)
    baseline_val_c = concordance_index(val.time, val.event, val.baseline_risk)
    baseline_test_c = concordance_index(test.time, test.event, test.baseline_risk)
    high_disagreement_mask = _high_disagreement_mask(val)
    baseline_high_val_c = concordance_index(
        val.time[high_disagreement_mask],
        val.event[high_disagreement_mask],
        val.baseline_risk[high_disagreement_mask],
    )
    rows = [
        {
            "candidate_name": "baseline",
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
            high_val_c = concordance_index(
                val.time[high_disagreement_mask],
                val.event[high_disagreement_mask],
                val_risk[high_disagreement_mask],
            )
            oof_delta = oof_c - baseline_oof_c
            val_delta = val_c - baseline_val_c
            high_delta = high_val_c - baseline_high_val_c
            eligible = (
                oof_delta >= float(min_oof_delta)
                and val_delta >= float(min_val_delta)
                and high_delta >= float(min_high_disagreement_val_delta)
            )
            rows.append(
                {
                    "candidate_name": candidate.name,
                    "alpha": alpha,
                    "oof_c_index": oof_c,
                    "oof_delta": oof_delta,
                    "val_c_index": val_c,
                    "val_delta": val_delta,
                    "high_disagreement_val_delta": high_delta,
                    "test_c_index": concordance_index(test.time, test.event, test_risk),
                    "eligible": bool(eligible),
                    "oof_risk": oof_risk,
                    "val_risk": val_risk,
                    "test_risk": test_risk,
                }
            )
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(eligible_rows, key=lambda row: (row["oof_c_index"], row["val_c_index"]))
    return {
        "candidate_name": selected["candidate_name"],
        "alpha": float(selected["alpha"]),
        "oof_c_index": float(selected["oof_c_index"]),
        "oof_delta": float(selected["oof_delta"]),
        "val_c_index": float(selected["val_c_index"]),
        "val_delta": float(selected["val_delta"]),
        "high_disagreement_val_delta": float(selected["high_disagreement_val_delta"]),
        "test_c_index": float(selected["test_c_index"]),
        "selection_policy": "max_oof_with_val_and_high_disagreement_safety_gate",
        "candidate_table": [
            {
                key: value
                for key, value in row.items()
                if key not in {"oof_risk", "val_risk", "test_risk"}
            }
            for row in rows
        ],
        "oof_risk": selected["oof_risk"],
        "val_risk": selected["val_risk"],
        "test_risk": selected["test_risk"],
    }


def _high_disagreement_mask(split: StructuredSplit) -> np.ndarray:
    values = split.groups["risk_disagreement"][:, 0]
    return values >= np.quantile(values, 0.75)


def _fit_group_scaler(groups: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scaler: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, values in groups.items():
        subset = np.asarray(values[indices], dtype=float)
        mean = subset.mean(axis=0, keepdims=True)
        std = np.maximum(subset.std(axis=0, keepdims=True), 1e-6)
        scaler[name] = (mean, std)
    return scaler


def _groups_to_tensors(
    split: StructuredSplit,
    indices: np.ndarray,
    scaler: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = {}
    for name in GROUP_NAMES:
        mean, std = scaler[name]
        values = (split.groups[name][indices] - mean) / std
        tensors[name] = torch.tensor(values, dtype=torch.float32, device=device)
    return tensors


def _zscore_torch(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / torch.clamp(values.std(unbiased=False), min=1e-6)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _tag_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/current_mainline_v2/structured_features_v2/structured_gnn_features_v2_all_splits.npz",
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seeds", default="7,21,42")
    parser.add_argument("--max-deltas", default="0.2,0.35,0.5")
    parser.add_argument("--distillation-weights", default="0.1")
    parser.add_argument("--delta-l2-weights", default="0.08")
    parser.add_argument("--disagreement-l2-weights", default="0.25")
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--min-oof-delta", type=float, default=0.0)
    parser.add_argument("--min-val-delta", type=float, default=-0.00025)
    parser.add_argument("--min-high-disagreement-val-delta", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--output",
        default="outputs/current_mainline_v2/structured_ctm_oof_v2/structured_ctm_oof_v2_summary.json",
    )
    args = parser.parse_args()
    result = run_structured_ctm_oof_experiment(
        feature_npz_path=args.features,
        inner_folds=args.inner_folds,
        seeds=_parse_int_list(args.seeds),
        max_deltas=_parse_float_list(args.max_deltas),
        distillation_weights=_parse_float_list(args.distillation_weights),
        delta_l2_weights=_parse_float_list(args.delta_l2_weights),
        disagreement_l2_weights=_parse_float_list(args.disagreement_l2_weights),
        alpha_grid=_parse_float_list(args.alpha_grid),
        epochs=args.epochs,
        patience=args.patience,
        min_oof_delta=args.min_oof_delta,
        min_val_delta=args.min_val_delta,
        min_high_disagreement_val_delta=args.min_high_disagreement_val_delta,
        device_arg=args.device,
        output_path=args.output,
    )
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
