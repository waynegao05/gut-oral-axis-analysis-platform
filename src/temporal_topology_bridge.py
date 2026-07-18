from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data

from config.settings import (
    PROJECT_ROOT,
    RESEARCH_MODEL_CONFIG_PATH,
    TEMPORAL_TOPOLOGY_DEVICE,
    TEMPORAL_TOPOLOGY_FULL_RISK_ROOT,
    TEMPORAL_TOPOLOGY_INFERENCE_METRICS,
    TEMPORAL_TOPOLOGY_MODEL_SEEDS,
    TEMPORAL_TOPOLOGY_RELEASE_METRICS,
    TEMPORAL_TOPOLOGY_RELEASE_NAME,
    TEMPORAL_TOPOLOGY_RELEASE_NOTE,
    TEMPORAL_TOPOLOGY_ROOT,
    TEMPORAL_TOPOLOGY_SPLIT_SEEDS,
)
from experiments.temporal_independent_v3.topology_aft_fusion import (
    build_topology_fingerprint_from_frames,
)
from research.data import build_sample_table, load_research_tables
from research.full_risk_head_refiner_v2 import CachedCoxHeadRefiner
from research.model_v2 import DeepStructureAwareGATCoxModelV2


_MIN_STD = 1e-6


@dataclass
class ResearchModelPrediction:
    risk_result: Dict[str, object]
    model_features: Dict[str, object]


@dataclass(frozen=True)
class _Scaler:
    mean: float
    std: float

    @classmethod
    def from_values(cls, values: np.ndarray) -> "_Scaler":
        array = np.asarray(values, dtype=float)
        return cls(mean=float(array.mean()), std=max(float(array.std()), _MIN_STD))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "_Scaler":
        return cls(
            mean=float(values["train_mean"]),
            std=max(float(values["train_std"]), _MIN_STD),
        )

    def transform(self, value: float) -> float:
        return float((float(value) - self.mean) / self.std)


@dataclass(frozen=True)
class _TopologyEstimate:
    function_scores: dict[str, float]
    edge_weights: dict[tuple[str, str], float]
    out_of_training_range: list[str]
    out_of_training_range_details: list[dict[str, float | str]]


class _TopologyInferenceModel:
    """Infer sample-specific topology using only fields available to the web API."""

    def __init__(
        self,
        *,
        input_columns: Sequence[str],
        node_order: Sequence[str],
        edge_pairs: Sequence[tuple[str, str]],
        function_model: Any,
        edge_model: Any,
        defaults: np.ndarray,
        input_minimums: np.ndarray,
        input_maximums: np.ndarray,
        function_minimums: np.ndarray,
        function_maximums: np.ndarray,
        edge_minimums: np.ndarray,
        edge_maximums: np.ndarray,
        function_alpha: float,
        edge_alpha: float,
        training_size: int,
    ) -> None:
        self.input_columns = list(input_columns)
        self.node_order = list(node_order)
        self.edge_pairs = list(edge_pairs)
        self.function_model = function_model
        self.edge_model = edge_model
        self.defaults = np.asarray(defaults, dtype=float)
        self.input_minimums = np.asarray(input_minimums, dtype=float)
        self.input_maximums = np.asarray(input_maximums, dtype=float)
        self.function_minimums = np.asarray(function_minimums, dtype=float)
        self.function_maximums = np.asarray(function_maximums, dtype=float)
        self.edge_minimums = np.asarray(edge_minimums, dtype=float)
        self.edge_maximums = np.asarray(edge_maximums, dtype=float)
        self.function_alpha = float(function_alpha)
        self.edge_alpha = float(edge_alpha)
        self.training_size = int(training_size)

    @classmethod
    def fit(
        cls,
        *,
        train_sample_ids: Sequence[str],
        input_frame: pd.DataFrame,
        function_frame: pd.DataFrame,
        edge_frame: pd.DataFrame,
        node_order: Sequence[str],
        edge_pairs: Sequence[tuple[str, str]],
    ) -> "_TopologyInferenceModel":
        ids = np.asarray(train_sample_ids).astype(str)
        train_x = input_frame.loc[ids].to_numpy(dtype=float)
        train_function = function_frame.loc[ids].to_numpy(dtype=float)
        train_edges = edge_frame.loc[ids].to_numpy(dtype=float)
        if not all(np.isfinite(values).all() for values in (train_x, train_function, train_edges)):
            raise ValueError("Topology inference training data contains non-finite values.")

        alphas = np.logspace(-3.0, 3.0, 13)
        function_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        edge_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        function_model.fit(train_x, train_function)
        edge_model.fit(train_x, train_edges)

        return cls(
            input_columns=input_frame.columns.tolist(),
            node_order=node_order,
            edge_pairs=edge_pairs,
            function_model=function_model,
            edge_model=edge_model,
            defaults=train_x.mean(axis=0),
            input_minimums=train_x.min(axis=0),
            input_maximums=train_x.max(axis=0),
            function_minimums=train_function.min(axis=0),
            function_maximums=train_function.max(axis=0),
            edge_minimums=train_edges.min(axis=0),
            edge_maximums=train_edges.max(axis=0),
            function_alpha=float(function_model.named_steps["ridgecv"].alpha_),
            edge_alpha=float(edge_model.named_steps["ridgecv"].alpha_),
            training_size=len(ids),
        )

    def default_for(self, column: str) -> float:
        return float(self.defaults[self.input_columns.index(column)])

    def predict(self, values: Mapping[str, float]) -> _TopologyEstimate:
        row = np.asarray(
            [float(values.get(column, self.defaults[index])) for index, column in enumerate(self.input_columns)],
            dtype=float,
        )
        if not np.isfinite(row).all():
            raise ValueError("Web topology inference received a non-finite input value.")

        tolerance = 1e-9
        outside = (row < self.input_minimums - tolerance) | (row > self.input_maximums + tolerance)
        outside_indices = np.flatnonzero(outside)
        out_of_range = [self.input_columns[index] for index in outside_indices]
        out_of_range_details = [
            {
                "field": self.input_columns[index],
                "value": float(row[index]),
                "training_minimum": float(self.input_minimums[index]),
                "training_maximum": float(self.input_maximums[index]),
            }
            for index in outside_indices
        ]

        function_values = np.asarray(self.function_model.predict(row.reshape(1, -1))[0], dtype=float)
        edge_values = np.asarray(self.edge_model.predict(row.reshape(1, -1))[0], dtype=float)
        function_values = np.clip(function_values, self.function_minimums, self.function_maximums)
        edge_values = np.clip(edge_values, self.edge_minimums, self.edge_maximums)

        return _TopologyEstimate(
            function_scores={
                node_name: float(function_values[index])
                for index, node_name in enumerate(self.node_order)
            },
            edge_weights={
                edge_pair: float(edge_values[index])
                for index, edge_pair in enumerate(self.edge_pairs)
            },
            out_of_training_range=out_of_range,
            out_of_training_range_details=out_of_range_details,
        )


@dataclass
class _SplitRuntime:
    split_seed: int
    config: dict[str, Any]
    selected_mainline_name: str
    gnn_models: list[DeepStructureAwareGATCoxModelV2]
    refiners: list[CachedCoxHeadRefiner | None]
    member_means: np.ndarray
    member_stds: np.ndarray
    member_weights: np.ndarray
    aft_models: list[xgb.Booster]
    aft_member_scalers: list[_Scaler]
    expert_ensemble_scaler: _Scaler
    consensus_mainline_scaler: _Scaler
    consensus_expert_scaler: _Scaler
    feature_columns: list[str]
    imputation_medians: np.ndarray
    topology_model: _TopologyInferenceModel
    reference_risks: dict[str, float]
    artifact_paths: list[str]
    evaluation_sample_ids: dict[str, list[str]]
    anchor_weighted_degree: float
    anchor_data: Data | None


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_torch(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class TemporalTopologyModelBridge:
    """Formal web bridge for the latest cross-split temporal-topology model."""

    def __init__(self, *, device: str | None = None) -> None:
        self.release_name = TEMPORAL_TOPOLOGY_RELEASE_NAME
        self.release_note = TEMPORAL_TOPOLOGY_RELEASE_NOTE
        self.release_metrics = dict(TEMPORAL_TOPOLOGY_RELEASE_METRICS)
        self.topology_inference_metrics = dict(TEMPORAL_TOPOLOGY_INFERENCE_METRICS)
        self.device = self._resolve_device(device or TEMPORAL_TOPOLOGY_DEVICE)

        self.base_config = yaml.safe_load(RESEARCH_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
        (
            self.graph_df,
            self.clinical_df,
            self.metabolite_df,
            self.label_df,
            self.data_summary,
        ) = load_research_tables(
            graph_csv=self.base_config["paths"]["graph_csv"],
            clinical_csv=self.base_config["paths"]["clinical_csv"],
            metabolite_csv=self.base_config["paths"]["metabolite_csv"],
            label_csv=self.base_config["paths"]["label_csv"],
        )
        self.graph_df["sample_id"] = self.graph_df["sample_id"].astype(str)
        self.clinical_df["sample_id"] = self.clinical_df["sample_id"].astype(str)
        self.metabolite_df["sample_id"] = self.metabolite_df["sample_id"].astype(str)
        self.label_df["sample_id"] = self.label_df["sample_id"].astype(str)
        self.sample_df = build_sample_table(
            clinical_df=self.clinical_df,
            metabolite_df=self.metabolite_df,
            label_df=self.label_df,
        )
        self.sample_df["sample_id"] = self.sample_df["sample_id"].astype(str)
        self.sample_index = self.sample_df.set_index("sample_id", drop=False)

        first_sample_id = str(self.graph_df["sample_id"].iloc[0])
        first_graph = self.graph_df.loc[self.graph_df["sample_id"] == first_sample_id]
        self.node_order = first_graph["node_name"].drop_duplicates().astype(str).tolist()
        self.edge_pairs = [
            (str(src), str(dst))
            for src, dst in first_graph[["src", "dst"]].itertuples(index=False, name=None)
        ]
        self.graph_template = first_graph[["node_name", "src", "dst"]].reset_index(drop=True).copy()
        self.reference_abundance_total = self._reference_abundance_total()
        self.sample_max_weighted_degree = self._sample_max_weighted_degrees()

        fingerprint, feature_columns, _ = build_topology_fingerprint_from_frames(
            self.graph_df,
            self.sample_df,
            data_summary=self.data_summary,
        )
        self.full_feature_columns = list(feature_columns)
        self.fingerprint_index = fingerprint.assign(
            sample_id=fingerprint["sample_id"].astype(str)
        ).set_index("sample_id", drop=False)
        (
            self.topology_input_frame,
            self.function_target_frame,
            self.edge_target_frame,
        ) = self._build_topology_learning_frames()

        consensus_path = TEMPORAL_TOPOLOGY_ROOT / "cross_split_consensus" / "cross_split_consensus_summary.json"
        consensus_summary = self._read_json(consensus_path)
        self.consensus_alpha = float(consensus_summary["selection"]["selected"]["alpha"])
        if not np.isclose(self.consensus_alpha, float(self.release_metrics["consensus_alpha"])):
            raise RuntimeError("Configured consensus alpha does not match the saved research artifact.")

        self.runtimes = {
            int(split_seed): self._load_split_runtime(int(split_seed))
            for split_seed in TEMPORAL_TOPOLOGY_SPLIT_SEEDS
        }
        self._initialize_reference_distribution()

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        normalized = str(value).strip().lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GOA_TEMPORAL_DEVICE=cuda was requested, but CUDA is unavailable.")
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("GOA_TEMPORAL_DEVICE must be cpu, cuda, or auto.")
        return torch.device(normalized)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required temporal-topology artifact is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _resolve_artifact(path_value: str | Path) -> Path:
        path = Path(path_value)
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Required temporal-topology artifact is missing: {resolved}")
        return resolved

    @staticmethod
    def _artifact_label(path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return path.name

    def _reference_abundance_total(self) -> float:
        nodes = self.graph_df.drop_duplicates(["sample_id", "node_name"])
        return float(nodes.groupby("sample_id")["abundance"].sum().mean())

    def _sample_max_weighted_degrees(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for sample_id, graph in self.graph_df.groupby("sample_id", sort=False):
            weighted_degree = {name: 0.0 for name in self.node_order}
            for src, dst, weight in graph[["src", "dst", "edge_weight"]].itertuples(
                index=False, name=None
            ):
                weighted_degree[str(src)] += float(weight)
                weighted_degree[str(dst)] += float(weight)
            result[str(sample_id)] = max(weighted_degree.values())
        return result

    def _build_topology_learning_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        node_rows = self.graph_df.drop_duplicates(["sample_id", "node_name"]).copy()
        abundance = node_rows.pivot(index="sample_id", columns="node_name", values="abundance")
        abundance = abundance.reindex(columns=self.node_order)
        abundance.columns = [f"abundance::{name}" for name in self.node_order]

        function = node_rows.pivot(index="sample_id", columns="node_name", values="function_score")
        function = function.reindex(columns=self.node_order)
        function.columns = [f"function::{name}" for name in self.node_order]

        graph = self.graph_df.copy()
        graph["edge_key"] = graph["src"].astype(str) + " -> " + graph["dst"].astype(str)
        edge_names = [f"{src} -> {dst}" for src, dst in self.edge_pairs]
        edges = graph.pivot(index="sample_id", columns="edge_key", values="edge_weight")
        edges = edges.reindex(columns=edge_names)
        edges.columns = [f"edge::{name}" for name in edge_names]

        sample = self.sample_index.copy()
        clinical_columns = list(self.base_config["model"]["clinical_columns"])
        metabolite_columns = list(self.base_config["model"]["metabolite_columns"])
        inputs = sample[clinical_columns + metabolite_columns].join(abundance, how="inner")
        inputs.index = inputs.index.astype(str)
        function.index = function.index.astype(str)
        edges.index = edges.index.astype(str)
        return inputs.astype(float), function.astype(float), edges.astype(float)

    def _build_gnn_model(self, config: Mapping[str, Any]) -> DeepStructureAwareGATCoxModelV2:
        model_config = config["model"]
        train_config = config["train"]
        return DeepStructureAwareGATCoxModelV2(
            node_feature_dim=len(model_config["node_feature_columns"]),
            clinical_dim=len(model_config["clinical_columns"]),
            metabolite_dim=len(model_config["metabolite_columns"]),
            hidden_dim=int(train_config["hidden_dim"]),
            heads=int(train_config["heads"]),
            dropout=float(train_config["dropout"]),
            edge_hidden_dim=int(train_config.get("edge_hidden_dim", 24)),
            num_layers=int(train_config.get("num_layers", 4)),
            layer_attn_heads=int(train_config.get("layer_attn_heads", 4)),
            contrastive_temperature=float(train_config.get("contrastive_temperature", 0.2)),
            survival_head_type=str(train_config.get("survival_head_type", "cox")),
            num_time_bins=int(train_config.get("num_time_bins", 12)),
            use_layer_attention=bool(train_config.get("use_layer_attention", False)),
        ).to(self.device)

    def _load_split_runtime(self, split_seed: int) -> _SplitRuntime:
        summary_path = TEMPORAL_TOPOLOGY_FULL_RISK_ROOT / f"split{split_seed}_three_seed_summary.json"
        summary = self._read_json(summary_path)
        config_path = self._resolve_artifact(summary["config_path"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        selected_name = str(summary["selected"]["name"])
        checkpoint_paths = [self._resolve_artifact(value) for value in summary["checkpoints"]]
        checkpoint_names = [str(value) for value in summary["checkpoint_names"]]
        if len(checkpoint_paths) != 3 or len(checkpoint_names) != 3:
            raise RuntimeError(f"Split {split_seed} must contain exactly three selected GNN checkpoints.")

        artifact_lookup = {
            (str(row["checkpoint_name"]), str(row["mode"])): self._resolve_artifact(row["path"])
            for row in summary["artifacts"]
        }
        gnn_models: list[DeepStructureAwareGATCoxModelV2] = []
        refiners: list[CachedCoxHeadRefiner | None] = []
        artifact_paths: list[str] = []
        for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
            model = self._build_gnn_model(config)
            model.load_state_dict(_load_torch(checkpoint_path, self.device))
            model.eval()
            gnn_models.append(model)
            artifact_paths.append(self._artifact_label(checkpoint_path))

            if selected_name == "reference":
                refiners.append(None)
                continue
            refiner_path = artifact_lookup.get((checkpoint_name, selected_name))
            if refiner_path is None:
                raise FileNotFoundError(
                    f"Split {split_seed} is missing {selected_name} for {checkpoint_name}."
                )
            payload = _load_torch(refiner_path, self.device)
            refiner = CachedCoxHeadRefiner(
                mode=selected_name,
                fusion=copy.deepcopy(model.fusion),
                risk_head=copy.deepcopy(model.risk_head),
            ).to(self.device)
            refiner.load_state_dict(payload["state_dict"])
            refiner.eval()
            refiners.append(refiner)
            artifact_paths.append(self._artifact_label(refiner_path))

        selected_predictions_path = (
            TEMPORAL_TOPOLOGY_FULL_RISK_ROOT
            / f"split{split_seed}_three_seed_summary_selected_predictions.npz"
        )
        selected_predictions = _load_npz(selected_predictions_path)
        saved_names = selected_predictions["checkpoint_names"].astype(str).tolist()
        if saved_names != checkpoint_names:
            raise RuntimeError(f"Split {split_seed} checkpoint order does not match its prediction artifact.")
        val_member_matrix = np.asarray(
            selected_predictions["val_selected_member_risk_matrix"], dtype=float
        )
        member_means = val_member_matrix.mean(axis=1)
        member_stds = np.maximum(val_member_matrix.std(axis=1), _MIN_STD)
        member_weights = np.asarray(summary["selected"]["weights"], dtype=float)
        if member_weights.shape != (3,) or not np.isclose(member_weights.sum(), 1.0):
            raise RuntimeError(f"Split {split_seed} has invalid mainline ensemble weights.")
        evaluation_sample_ids = {
            split_name: selected_predictions[f"{split_name}_sample_ids"].astype(str).tolist()
            for split_name in ("train", "val", "test")
        }
        batch_size = int(config["train"]["batch_size"])
        batch_maximums = [
            max(self.sample_max_weighted_degree[sample_id] for sample_id in sample_ids[start:start + batch_size])
            for sample_ids in evaluation_sample_ids.values()
            for start in range(0, len(sample_ids), batch_size)
        ]
        anchor_weighted_degree = float(np.median(np.asarray(batch_maximums, dtype=float)))

        ensemble_dir = TEMPORAL_TOPOLOGY_ROOT / f"split{split_seed}_five_seed_ensemble"
        ensemble_summary = self._read_json(ensemble_dir / "seed_ensemble_summary.json")
        ensemble_predictions = _load_npz(ensemble_dir / "seed_ensemble_predictions.npz")
        train_ids = ensemble_predictions["train_sample_ids"].astype(str)
        member_rows = {int(row["model_seed"]): row for row in ensemble_summary["member_rows"]}

        feature_columns: list[str] | None = None
        aft_models: list[xgb.Booster] = []
        aft_member_scalers: list[_Scaler] = []
        for model_seed in TEMPORAL_TOPOLOGY_MODEL_SEEDS:
            run_dir = TEMPORAL_TOPOLOGY_ROOT / f"split{split_seed}_seed{int(model_seed)}"
            run_summary = self._read_json(run_dir / "summary.json")
            run_features = [str(value) for value in run_summary["feature_metadata"]["feature_columns"]]
            if feature_columns is None:
                feature_columns = run_features
            elif run_features != feature_columns:
                raise RuntimeError(f"Split {split_seed} AFT members do not share the same feature order.")

            selected_expert = run_summary["selected_expert"]
            member_row = member_rows[int(model_seed)]
            if str(selected_expert["name"]) != str(member_row["selected_expert"]):
                raise RuntimeError(f"Split {split_seed}, seed {model_seed} selected expert mismatch.")
            model_path = self._resolve_artifact(selected_expert["model_path"])
            booster = xgb.Booster()
            booster.load_model(model_path)
            aft_models.append(booster)
            aft_member_scalers.append(_Scaler.from_mapping(member_row["risk_scaler"]))
            artifact_paths.append(self._artifact_label(model_path))

        if feature_columns is None or feature_columns != self.full_feature_columns:
            raise RuntimeError(f"Split {split_seed} feature schema does not match the topology builder.")
        train_features = self.fingerprint_index.loc[train_ids, feature_columns].astype(float)
        imputation_medians = (
            train_features.replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0).to_numpy(float)
        )
        topology_model = _TopologyInferenceModel.fit(
            train_sample_ids=train_ids,
            input_frame=self.topology_input_frame,
            function_frame=self.function_target_frame,
            edge_frame=self.edge_target_frame,
            node_order=self.node_order,
            edge_pairs=self.edge_pairs,
        )

        consensus_path = (
            TEMPORAL_TOPOLOGY_ROOT
            / "cross_split_consensus"
            / f"split{split_seed}_consensus_predictions.npz"
        )
        consensus_predictions = _load_npz(consensus_path)
        if not np.isclose(float(consensus_predictions["selected_alpha"]), self.consensus_alpha):
            raise RuntimeError(f"Split {split_seed} consensus alpha mismatch.")
        reference_risks: dict[str, float] = {}
        for split_name in ("train", "val", "test"):
            sample_ids = consensus_predictions[f"{split_name}_sample_ids"].astype(str)
            risks = np.asarray(consensus_predictions[f"{split_name}_selected_risk"], dtype=float)
            reference_risks.update(dict(zip(sample_ids.tolist(), risks.tolist())))

        artifact_paths.extend(
            [
                self._artifact_label(summary_path),
                self._artifact_label(selected_predictions_path),
                self._artifact_label(consensus_path),
            ]
        )
        runtime = _SplitRuntime(
            split_seed=split_seed,
            config=config,
            selected_mainline_name=selected_name,
            gnn_models=gnn_models,
            refiners=refiners,
            member_means=member_means,
            member_stds=member_stds,
            member_weights=member_weights,
            aft_models=aft_models,
            aft_member_scalers=aft_member_scalers,
            expert_ensemble_scaler=_Scaler.from_mapping(
                ensemble_summary["expert_ensemble_scaler"]
            ),
            consensus_mainline_scaler=_Scaler.from_values(
                ensemble_predictions["train_mainline_risk"]
            ),
            consensus_expert_scaler=_Scaler.from_values(
                ensemble_predictions["train_expert_ensemble_risk"]
            ),
            feature_columns=feature_columns,
            imputation_medians=imputation_medians,
            topology_model=topology_model,
            reference_risks=reference_risks,
            artifact_paths=artifact_paths,
            evaluation_sample_ids=evaluation_sample_ids,
            anchor_weighted_degree=anchor_weighted_degree,
            anchor_data=None,
        )
        runtime.anchor_data = self._build_calibration_anchor(runtime)
        return runtime

    def _build_calibration_anchor(self, runtime: _SplitRuntime) -> Data:
        node_degree = max(len(self.node_order) - 1, 1)
        edge_weight = min(runtime.anchor_weighted_degree / float(node_degree), 1.0)
        abundances = {
            name: float(self.topology_input_frame[f"abundance::{name}"].mean())
            for name in self.node_order
        }
        estimate = _TopologyEstimate(
            function_scores={
                name: float(self.function_target_frame[f"function::{name}"].mean())
                for name in self.node_order
            },
            edge_weights={edge_pair: edge_weight for edge_pair in self.edge_pairs},
            out_of_training_range=[],
            out_of_training_range_details=[],
        )
        clinical = {
            str(column): runtime.topology_model.default_for(str(column))
            for column in runtime.config["model"]["clinical_columns"]
        }
        metabolites = {
            str(column): runtime.topology_model.default_for(str(column))
            for column in runtime.config["model"]["metabolite_columns"]
        }
        sample_id = f"WEB_ANCHOR_SPLIT_{runtime.split_seed}"
        graph = self._build_graph_frame(
            sample_id=sample_id,
            scaled_microbes=abundances,
            estimate=estimate,
        )
        return self._build_data(runtime, graph, clinical, metabolites, sample_id)

    def _initialize_reference_distribution(self) -> None:
        runtime_values = list(self.runtimes.values())
        common_ids = set(runtime_values[0].reference_risks)
        for runtime in runtime_values[1:]:
            common_ids &= set(runtime.reference_risks)
        if len(common_ids) != len(self.sample_df):
            raise RuntimeError("Cross-split reference artifacts do not cover the complete research cohort.")

        ordered_ids = sorted(common_ids)
        split_matrix = np.asarray(
            [
                [runtime.reference_risks[sample_id] for sample_id in ordered_ids]
                for runtime in runtime_values
            ],
            dtype=float,
        )
        self.reference_risks = split_matrix.mean(axis=0)
        self.reference_disagreements = np.abs(split_matrix[0] - split_matrix[1])
        self.low_threshold = float(np.quantile(self.reference_risks, 1.0 / 3.0))
        self.medium_threshold = float(np.quantile(self.reference_risks, 2.0 / 3.0))
        self.disagreement_threshold = float(np.quantile(self.reference_disagreements, 0.90))

    def _scale_supported_microbes(self, microbes: Mapping[str, float]) -> dict[str, float]:
        supported = {
            name: max(float(microbes.get(name, 0.0)), 0.0)
            for name in self.node_order
        }
        total = float(sum(supported.values()))
        if total <= 0.0:
            raise ValueError(
                "No supported oral microbes were provided for the temporal-topology model. "
                f"Supported microbes: {', '.join(self.node_order)}."
            )
        if total <= 1.2:
            scale = self.reference_abundance_total / total
            supported = {name: min(1.0, value * scale) for name, value in supported.items()}
        return supported

    def _resolve_split_inputs(
        self,
        runtime: _SplitRuntime,
        scaled_microbes: Mapping[str, float],
        clinical: Mapping[str, float],
        metabolites: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[str]]:
        resolved_clinical: dict[str, float] = {}
        resolved_metabolites: dict[str, float] = {}
        flat_values: dict[str, float] = {
            f"abundance::{name}": float(scaled_microbes[name])
            for name in self.node_order
        }
        defaulted: list[str] = []
        for column in runtime.config["model"]["clinical_columns"]:
            if column in clinical:
                value = float(clinical[column])
            else:
                value = runtime.topology_model.default_for(str(column))
                defaulted.append(f"clinical.{column}")
            resolved_clinical[str(column)] = value
            flat_values[str(column)] = value
        for column in runtime.config["model"]["metabolite_columns"]:
            if column in metabolites:
                value = float(metabolites[column])
            else:
                value = runtime.topology_model.default_for(str(column))
                defaulted.append(f"metabolites.{column}")
            resolved_metabolites[str(column)] = value
            flat_values[str(column)] = value
        if not np.isfinite(np.asarray(list(flat_values.values()), dtype=float)).all():
            raise ValueError("Temporal-topology inference received a non-finite input value.")
        return resolved_clinical, resolved_metabolites, flat_values, defaulted

    def _build_graph_frame(
        self,
        *,
        sample_id: str,
        scaled_microbes: Mapping[str, float],
        estimate: _TopologyEstimate,
    ) -> pd.DataFrame:
        graph = self.graph_template.copy()
        graph.insert(0, "sample_id", sample_id)
        graph["abundance"] = graph["node_name"].map(scaled_microbes).astype(float)
        graph["function_score"] = graph["node_name"].map(estimate.function_scores).astype(float)
        graph["edge_weight"] = [
            estimate.edge_weights[(str(src), str(dst))]
            for src, dst in graph[["src", "dst"]].itertuples(index=False, name=None)
        ]
        return graph

    def _build_data(
        self,
        runtime: _SplitRuntime,
        graph: pd.DataFrame,
        clinical: Mapping[str, float],
        metabolites: Mapping[str, float],
        sample_id: str,
    ) -> Data:
        node_rows = graph.drop_duplicates("node_name").set_index("node_name").loc[self.node_order]
        node_columns = list(runtime.config["model"]["node_feature_columns"])
        x = torch.tensor(node_rows[node_columns].to_numpy(float), dtype=torch.float32)
        node_to_index = {name: index for index, name in enumerate(self.node_order)}
        edges_src: list[int] = []
        edges_dst: list[int] = []
        edge_attr: list[list[float]] = []
        for src, dst, edge_weight in graph[["src", "dst", "edge_weight"]].itertuples(
            index=False, name=None
        ):
            src_index = node_to_index[str(src)]
            dst_index = node_to_index[str(dst)]
            edges_src.extend([src_index, dst_index])
            edges_dst.extend([dst_index, src_index])
            edge_attr.extend([[float(edge_weight)], [float(edge_weight)]])

        clinical_values = [
            float(clinical[column]) for column in runtime.config["model"]["clinical_columns"]
        ]
        metabolite_values = [
            float(metabolites[column])
            for column in runtime.config["model"]["metabolite_columns"]
        ]
        return Data(
            x=x,
            edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            clinical=torch.tensor(clinical_values, dtype=torch.float32),
            metabolites=torch.tensor(metabolite_values, dtype=torch.float32),
            time=torch.tensor(0.0, dtype=torch.float32),
            event=torch.tensor(0.0, dtype=torch.float32),
            sample_id=sample_id,
        )

    def _build_feature_vector(
        self,
        runtime: _SplitRuntime,
        graph: pd.DataFrame,
        clinical: Mapping[str, float],
        metabolites: Mapping[str, float],
        sample_id: str,
    ) -> np.ndarray:
        sample_row: dict[str, Any] = {
            "sample_id": sample_id,
            "time": 0.0,
            "event": 0.0,
            **clinical,
            **metabolites,
        }
        fingerprint, _, _ = build_topology_fingerprint_from_frames(
            graph,
            pd.DataFrame([sample_row]),
            data_summary=self.data_summary,
        )
        row = pd.to_numeric(
            fingerprint.iloc[0].reindex(runtime.feature_columns), errors="coerce"
        ).to_numpy(dtype=float)
        row = np.where(np.isfinite(row), row, runtime.imputation_medians)
        if not np.isfinite(row).all():
            raise ValueError("Non-finite topology features remain after train-only imputation.")
        return row.astype(np.float32)

    def _score_runtime(
        self,
        runtime: _SplitRuntime,
        data: Data,
        feature_vector: np.ndarray,
        *,
        context_items: Sequence[Data] | None = None,
    ) -> dict[str, float]:
        if context_items is None:
            if runtime.anchor_data is None:
                raise RuntimeError(f"Split {runtime.split_seed} calibration anchor is unavailable.")
            context_items = [runtime.anchor_data]
        batch_items = [data, *context_items]
        batch = Batch.from_data_list(batch_items).to(self.device)
        batch_size = len(batch_items)
        member_risks: list[float] = []
        embedding_norms: list[float] = []
        with torch.no_grad():
            for model, refiner in zip(runtime.gnn_models, runtime.refiners):
                output = model(batch, compute_contrastive=False)
                if refiner is None:
                    selected_risk = output["risk"]
                else:
                    selected_risk = refiner(
                        graph_embedding=output["graph_embedding"],
                        clinical=batch.clinical.view(batch_size, -1),
                        metabolites=batch.metabolites.view(batch_size, -1),
                        latent=output["latent"],
                    )
                member_risks.append(float(selected_risk[0].item()))
                embedding_norms.append(float(output["graph_embedding"][0].norm().item()))

        member_array = np.asarray(member_risks, dtype=float)
        standardized_members = (member_array - runtime.member_means) / runtime.member_stds
        mainline_raw = float(np.dot(runtime.member_weights, standardized_members))
        mainline_z = runtime.consensus_mainline_scaler.transform(mainline_raw)

        matrix = xgb.DMatrix(
            feature_vector.reshape(1, -1),
            feature_names=runtime.feature_columns,
        )
        expert_members = []
        for booster, scaler in zip(runtime.aft_models, runtime.aft_member_scalers):
            expert_raw = -float(np.asarray(booster.predict(matrix), dtype=float)[0])
            expert_members.append(scaler.transform(expert_raw))
        expert_member_mean = float(np.mean(expert_members))
        expert_ensemble_z = runtime.expert_ensemble_scaler.transform(expert_member_mean)
        expert_z = runtime.consensus_expert_scaler.transform(expert_ensemble_z)
        selected_risk = (1.0 - self.consensus_alpha) * mainline_z + self.consensus_alpha * expert_z
        return {
            "selected_risk": float(selected_risk),
            "mainline_risk": float(mainline_z),
            "expert_risk": float(expert_z),
            "graph_embedding_norm": float(np.mean(embedding_norms)),
        }

    @staticmethod
    def _average_named_values(
        estimates: Sequence[_TopologyEstimate],
        names: Sequence[str],
        getter: Any,
    ) -> dict[str, float]:
        return {
            name: float(np.mean([getter(estimate, name) for estimate in estimates]))
            for name in names
        }

    def score(
        self,
        microbes: Dict[str, float],
        clinical: Dict[str, float],
        metabolites: Dict[str, float],
    ) -> ResearchModelPrediction:
        scaled_microbes = self._scale_supported_microbes(microbes)
        split_scores: dict[int, dict[str, float]] = {}
        estimates: dict[int, _TopologyEstimate] = {}
        defaulted_inputs: set[str] = set()
        out_of_range: set[str] = set()
        out_of_range_details: dict[str, dict[str, Any]] = {}

        for split_seed, runtime in self.runtimes.items():
            resolved_clinical, resolved_metabolites, flat_values, defaulted = self._resolve_split_inputs(
                runtime,
                scaled_microbes,
                clinical,
                metabolites,
            )
            estimate = runtime.topology_model.predict(flat_values)
            graph = self._build_graph_frame(
                sample_id="WEB_INPUT",
                scaled_microbes=scaled_microbes,
                estimate=estimate,
            )
            data = self._build_data(
                runtime,
                graph,
                resolved_clinical,
                resolved_metabolites,
                "WEB_INPUT",
            )
            features = self._build_feature_vector(
                runtime,
                graph,
                resolved_clinical,
                resolved_metabolites,
                "WEB_INPUT",
            )
            split_scores[split_seed] = self._score_runtime(runtime, data, features)
            estimates[split_seed] = estimate
            defaulted_inputs.update(defaulted)
            out_of_range.update(estimate.out_of_training_range)
            for detail in estimate.out_of_training_range_details:
                field = str(detail["field"])
                existing = out_of_range_details.get(field)
                if existing is None:
                    out_of_range_details[field] = {
                        "field": field,
                        "value": float(detail["value"]),
                        "training_minimum": float(detail["training_minimum"]),
                        "training_maximum": float(detail["training_maximum"]),
                        "affected_split_seeds": [int(split_seed)],
                    }
                    continue
                existing["training_minimum"] = max(
                    float(existing["training_minimum"]),
                    float(detail["training_minimum"]),
                )
                existing["training_maximum"] = min(
                    float(existing["training_maximum"]),
                    float(detail["training_maximum"]),
                )
                existing["affected_split_seeds"].append(int(split_seed))

        ordered_split_scores = [split_scores[int(seed)] for seed in TEMPORAL_TOPOLOGY_SPLIT_SEEDS]
        raw_risk = float(np.mean([row["selected_risk"] for row in ordered_split_scores]))
        disagreement = float(
            abs(ordered_split_scores[0]["selected_risk"] - ordered_split_scores[1]["selected_risk"])
        )
        risk_percentile = float(100.0 * np.mean(self.reference_risks <= raw_risk))
        if raw_risk < self.low_threshold:
            risk_level = "low"
        elif raw_risk < self.medium_threshold:
            risk_level = "medium"
        else:
            risk_level = "high"

        if out_of_range:
            reliability = "caution_out_of_training_range"
        elif defaulted_inputs:
            reliability = "caution_defaulted_inputs"
        elif disagreement > self.disagreement_threshold:
            reliability = "caution_split_disagreement"
        else:
            reliability = "standard"

        estimate_values = [estimates[int(seed)] for seed in TEMPORAL_TOPOLOGY_SPLIT_SEEDS]
        inferred_functions = self._average_named_values(
            estimate_values,
            self.node_order,
            lambda estimate, name: estimate.function_scores[name],
        )
        edge_labels = [f"{src} -> {dst}" for src, dst in self.edge_pairs]
        inferred_edges = self._average_named_values(
            estimate_values,
            edge_labels,
            lambda estimate, name: estimate.edge_weights[tuple(name.split(" -> ", maxsplit=1))],
        )
        unsupported_microbes = sorted(set(microbes).difference(self.node_order))
        backend_name = "temporal_topology_aft_cross_split_consensus"
        risk_result = {
            "risk_score": round(risk_percentile, 2),
            "risk_level": risk_level,
            "risk_percentile": round(risk_percentile, 2),
            "raw_model_risk": round(raw_risk, 6),
            "split_consensus_risks": {
                str(seed): round(split_scores[int(seed)]["selected_risk"], 6)
                for seed in TEMPORAL_TOPOLOGY_SPLIT_SEEDS
            },
            "split_disagreement": round(disagreement, 6),
            "prediction_reliability": reliability,
            "ensemble_size": 16,
            "backend": backend_name,
            "model_release": self.release_name,
        }
        model_features = {
            "backend": backend_name,
            "model_release": self.release_name,
            "model_release_note": self.release_note,
            "device": str(self.device),
            "num_split_branches": 2,
            "num_gnn_models": 6,
            "num_aft_models": 10,
            "consensus_alpha": self.consensus_alpha,
            "topology_source": "inferred_from_web_inputs",
            "topology_inference_method": "split_train_only_standardized_ridge",
            "topology_disclaimer": (
                "Function scores and edge weights are model-inferred from web inputs; "
                "they are not directly measured laboratory topology."
            ),
            "inferred_function_scores": {
                key: round(value, 6) for key, value in inferred_functions.items()
            },
            "inferred_edge_weights": {
                key: round(value, 6) for key, value in inferred_edges.items()
            },
            "topology_inference_metrics": self.topology_inference_metrics,
            "gnn_inference_context": "fixed_median_batch_normalization_anchor",
            "gnn_anchor_weighted_degree": {
                str(seed): round(self.runtimes[int(seed)].anchor_weighted_degree, 6)
                for seed in TEMPORAL_TOPOLOGY_SPLIT_SEEDS
            },
            "gnn_context_note": (
                "A fixed calibration anchor prevents one patient's score from depending on "
                "other concurrent web requests. Saved research metrics used deterministic "
                "eight-sample evaluation batches."
            ),
            "defaulted_inputs": sorted(defaulted_inputs),
            "out_of_training_range_inputs": sorted(out_of_range),
            "out_of_training_range_details": [
                {
                    "field": field,
                    "value": round(float(detail["value"]), 6),
                    "training_minimum": round(float(detail["training_minimum"]), 6),
                    "training_maximum": round(float(detail["training_maximum"]), 6),
                    "affected_split_seeds": sorted(
                        set(int(seed) for seed in detail["affected_split_seeds"])
                    ),
                }
                for field, detail in sorted(out_of_range_details.items())
            ],
            "supported_microbes": self.node_order,
            "supported_microbe_input": {
                key: round(value, 6) for key, value in scaled_microbes.items()
            },
            "unsupported_microbes_ignored": unsupported_microbes,
            "graph_embedding_norm": round(
                float(np.mean([row["graph_embedding_norm"] for row in ordered_split_scores])), 6
            ),
            "reference_low_threshold_raw_risk": round(self.low_threshold, 6),
            "reference_medium_threshold_raw_risk": round(self.medium_threshold, 6),
            "reference_split_disagreement_p90": round(self.disagreement_threshold, 6),
            "reference_population": "topology_v6_complete_research_cohort",
            "reference_abundance_total": round(self.reference_abundance_total, 6),
            "release_metrics": self.release_metrics,
            "artifact_sources": [
                path
                for seed in TEMPORAL_TOPOLOGY_SPLIT_SEEDS
                for path in self.runtimes[int(seed)].artifact_paths
            ],
        }
        return ResearchModelPrediction(risk_result=risk_result, model_features=model_features)

    def score_reference_sample(self, sample_id: str) -> dict[str, Any]:
        """Replay one measured-topology cohort sample against saved branch risks."""
        normalized_id = str(sample_id)
        if normalized_id not in self.sample_index.index:
            raise KeyError(f"Unknown research sample_id: {normalized_id}")
        graph = self.graph_df.loc[self.graph_df["sample_id"] == normalized_id].copy()
        sample = self.sample_index.loc[normalized_id]
        split_rows: dict[str, dict[str, float]] = {}
        for split_seed, runtime in self.runtimes.items():
            clinical = {
                column: float(sample[column])
                for column in runtime.config["model"]["clinical_columns"]
            }
            metabolites = {
                column: float(sample[column])
                for column in runtime.config["model"]["metabolite_columns"]
            }
            data = self._build_data(runtime, graph, clinical, metabolites, normalized_id)
            split_name = next(
                name
                for name, sample_ids in runtime.evaluation_sample_ids.items()
                if normalized_id in sample_ids
            )
            sample_ids = runtime.evaluation_sample_ids[split_name]
            sample_index = sample_ids.index(normalized_id)
            batch_size = int(runtime.config["train"]["batch_size"])
            start = (sample_index // batch_size) * batch_size
            context_ids = [
                value
                for value in sample_ids[start:start + batch_size]
                if value != normalized_id
            ]
            context_items: list[Data] = []
            for context_id in context_ids:
                context_sample = self.sample_index.loc[context_id]
                context_graph = self.graph_df.loc[self.graph_df["sample_id"] == context_id].copy()
                context_clinical = {
                    column: float(context_sample[column])
                    for column in runtime.config["model"]["clinical_columns"]
                }
                context_metabolites = {
                    column: float(context_sample[column])
                    for column in runtime.config["model"]["metabolite_columns"]
                }
                context_items.append(
                    self._build_data(
                        runtime,
                        context_graph,
                        context_clinical,
                        context_metabolites,
                        context_id,
                    )
                )
            feature_row = pd.to_numeric(
                self.fingerprint_index.loc[normalized_id].reindex(runtime.feature_columns),
                errors="coerce",
            ).to_numpy(dtype=float)
            feature_row = np.where(np.isfinite(feature_row), feature_row, runtime.imputation_medians)
            scored = self._score_runtime(
                runtime,
                data,
                feature_row.astype(np.float32),
                context_items=context_items,
            )
            expected = float(runtime.reference_risks[normalized_id])
            split_rows[str(split_seed)] = {
                "actual": float(scored["selected_risk"]),
                "expected": expected,
                "absolute_error": abs(float(scored["selected_risk"]) - expected),
            }
        return {
            "sample_id": normalized_id,
            "splits": split_rows,
            "maximum_absolute_error": max(row["absolute_error"] for row in split_rows.values()),
        }


@lru_cache(maxsize=1)
def get_temporal_topology_model_bridge() -> TemporalTopologyModelBridge:
    return TemporalTopologyModelBridge()
