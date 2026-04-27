from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from config.settings import (
    RESEARCH_MODEL_CHECKPOINT_GLOB,
    RESEARCH_MODEL_CONFIG_PATH,
    RESEARCH_MODEL_FALLBACK_CHECKPOINT,
    RESEARCH_MODEL_MAX_REFERENCE_BATCH,
    RESEARCH_MODEL_REFERENCE_CACHE,
)
from research.data import build_dataset_from_csv, load_research_tables
from research.model_v2 import DeepStructureAwareGATCoxModelV2


@dataclass
class ResearchModelPrediction:
    risk_result: Dict[str, object]
    model_features: Dict[str, object]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class ResearchModelBridge:
    def __init__(self) -> None:
        self.config_path = RESEARCH_MODEL_CONFIG_PATH
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.graph_df, self.clinical_df, self.metabolite_df, self.label_df, _ = load_research_tables(
            graph_csv=self.config["paths"]["graph_csv"],
            clinical_csv=self.config["paths"]["clinical_csv"],
            metabolite_csv=self.config["paths"]["metabolite_csv"],
            label_csv=self.config["paths"]["label_csv"],
        )

        self.node_order = self._infer_node_order()
        self.edge_pairs = self._infer_edge_pairs()
        self.edge_weight_priors = self._fit_edge_weight_priors()
        self.function_score_params = self._fit_function_score_params()
        self.clinical_columns = list(self.config["model"]["clinical_columns"])
        self.metabolite_columns = list(self.config["model"]["metabolite_columns"])
        self.clinical_defaults = self._mean_map(self.clinical_df, self.clinical_columns)
        self.metabolite_defaults = self._mean_map(self.metabolite_df, self.metabolite_columns)
        self.reference_abundance_total = self._reference_abundance_total()

        self.dataset = build_dataset_from_csv(
            graph_csv=self.config["paths"]["graph_csv"],
            clinical_csv=self.config["paths"]["clinical_csv"],
            metabolite_csv=self.config["paths"]["metabolite_csv"],
            label_csv=self.config["paths"]["label_csv"],
            node_feature_columns=self.config["model"]["node_feature_columns"],
            clinical_columns=self.clinical_columns,
            metabolite_columns=self.metabolite_columns,
            seed=int(self.config["seed"]),
            split_seed=self.config["train"].get("split_seed"),
            keep_top_k_edges=self.config.get("graph_preprocess", {}).get("keep_top_k_edges"),
            min_edge_weight=self.config.get("graph_preprocess", {}).get("min_edge_weight"),
            val_ratio=float(self.config["train"]["val_ratio"]),
            test_ratio=float(self.config["train"]["test_ratio"]),
        )

        self.checkpoints = self._resolve_checkpoints()
        self.models = self._load_models()
        self.reference_risks = self._load_or_build_reference_risks()
        self.low_threshold = float(np.quantile(self.reference_risks, 1.0 / 3.0))
        self.medium_threshold = float(np.quantile(self.reference_risks, 2.0 / 3.0))

    def _infer_node_order(self) -> list[str]:
        first_sample_id = str(self.graph_df["sample_id"].iloc[0])
        first_sample = self.graph_df.loc[self.graph_df["sample_id"].astype(str) == first_sample_id]
        return first_sample["node_name"].drop_duplicates().tolist()

    def _infer_edge_pairs(self) -> list[tuple[str, str]]:
        first_sample_id = str(self.graph_df["sample_id"].iloc[0])
        first_sample = self.graph_df.loc[self.graph_df["sample_id"].astype(str) == first_sample_id]
        return list(first_sample[["src", "dst"]].drop_duplicates().itertuples(index=False, name=None))

    def _fit_edge_weight_priors(self) -> dict[tuple[str, str], float]:
        edge_weight_means = (
            self.graph_df.groupby(["src", "dst"])["edge_weight"].mean().to_dict()
        )
        return {
            (src, dst): float(edge_weight_means[(src, dst)])
            for src, dst in self.edge_pairs
        }

    def _fit_function_score_params(self) -> dict[str, tuple[float, float]]:
        params: dict[str, tuple[float, float]] = {}
        dedup = self.graph_df.drop_duplicates(subset=["sample_id", "node_name"]).copy()
        for node_name, node_df in dedup.groupby("node_name"):
            abundance = node_df["abundance"].to_numpy(dtype=float)
            function_score = node_df["function_score"].to_numpy(dtype=float)
            denom = float(np.sum((abundance - abundance.mean()) ** 2))
            slope = 0.0 if denom == 0.0 else float(
                np.sum((abundance - abundance.mean()) * (function_score - function_score.mean())) / denom
            )
            intercept = float(function_score.mean() - slope * abundance.mean())
            params[str(node_name)] = (slope, intercept)
        return params

    def _mean_map(self, df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        return {column: float(df[column].astype(float).mean()) for column in columns}

    def _reference_abundance_total(self) -> float:
        dedup = self.graph_df.drop_duplicates(subset=["sample_id", "node_name"]).copy()
        totals = dedup.groupby("sample_id")["abundance"].sum().to_numpy(dtype=float)
        return float(np.mean(totals))

    def _resolve_checkpoints(self) -> list[Path]:
        project_root = self.config_path.parent
        checkpoints = sorted(project_root.glob(RESEARCH_MODEL_CHECKPOINT_GLOB))
        if checkpoints:
            return checkpoints
        if RESEARCH_MODEL_FALLBACK_CHECKPOINT.exists():
            return [RESEARCH_MODEL_FALLBACK_CHECKPOINT]
        raise FileNotFoundError(
            "No research model checkpoints were found. "
            f"Tried glob: {RESEARCH_MODEL_CHECKPOINT_GLOB}"
        )

    def _build_model(self) -> DeepStructureAwareGATCoxModelV2:
        return DeepStructureAwareGATCoxModelV2(
            node_feature_dim=self.dataset.node_feature_dim,
            clinical_dim=self.dataset.clinical_dim,
            metabolite_dim=self.dataset.metabolite_dim,
            hidden_dim=int(self.config["train"]["hidden_dim"]),
            heads=int(self.config["train"]["heads"]),
            dropout=float(self.config["train"]["dropout"]),
            edge_hidden_dim=int(self.config["train"].get("edge_hidden_dim", 24)),
            num_layers=int(self.config["train"].get("num_layers", 4)),
            layer_attn_heads=int(self.config["train"].get("layer_attn_heads", 4)),
            contrastive_temperature=float(self.config["train"].get("contrastive_temperature", 0.2)),
            survival_head_type=str(self.config["train"].get("survival_head_type", "cox")),
            num_time_bins=int(self.config["train"].get("num_time_bins", 12)),
            use_layer_attention=bool(self.config["train"].get("use_layer_attention", False)),
        ).to(self.device)

    def _load_models(self) -> list[DeepStructureAwareGATCoxModelV2]:
        models: list[DeepStructureAwareGATCoxModelV2] = []
        for checkpoint_path in self.checkpoints:
            model = self._build_model()
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        return models

    def _score_batches(self, loader: DataLoader) -> np.ndarray:
        all_risks: list[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                risk_sum = None
                for model in self.models:
                    output = model(batch, compute_contrastive=False)
                    batch_risk = output["risk"]
                    risk_sum = batch_risk if risk_sum is None else risk_sum + batch_risk
                mean_risk = risk_sum / max(len(self.models), 1)
                all_risks.extend(mean_risk.detach().cpu().numpy().tolist())
        return np.asarray(all_risks, dtype=float)

    def _score_reference_cohort(self) -> np.ndarray:
        all_samples = self.dataset.train_set + self.dataset.val_set + self.dataset.test_set
        loader = DataLoader(
            all_samples,
            batch_size=max(int(self.config["train"]["batch_size"]), RESEARCH_MODEL_MAX_REFERENCE_BATCH),
            shuffle=False,
        )
        return self._score_batches(loader)

    def _reference_cache_payload(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path.as_posix()),
            "checkpoints": [str(path.as_posix()) for path in self.checkpoints],
        }

    def _load_or_build_reference_risks(self) -> np.ndarray:
        cache_path = RESEARCH_MODEL_REFERENCE_CACHE
        expected_meta = self._reference_cache_payload()
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("meta") == expected_meta:
                    return np.asarray(cached["reference_risks"], dtype=float)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        reference_risks = self._score_reference_cohort()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "meta": expected_meta,
                    "reference_risks": reference_risks.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return reference_risks

    def _scale_supported_microbes(self, microbes: Dict[str, float]) -> dict[str, float]:
        supported = {name: max(float(microbes.get(name, 0.0)), 0.0) for name in self.node_order}
        total = float(sum(supported.values()))
        if total > 0.0 and total <= 1.2:
            scale = self.reference_abundance_total / total
            supported = {name: min(1.0, value * scale) for name, value in supported.items()}
        return supported

    def _predict_function_score(self, node_name: str, abundance: float) -> float:
        slope, intercept = self.function_score_params[node_name]
        return _clip01(slope * abundance + intercept)

    def _build_single_sample(self, microbes: Dict[str, float], clinical: Dict[str, float], metabolites: Dict[str, float]) -> Data:
        scaled_microbes = self._scale_supported_microbes(microbes)
        if sum(scaled_microbes.values()) <= 0.0:
            raise ValueError(
                "No supported oral microbes were provided for the research GNN model. "
                f"Supported microbes: {', '.join(self.node_order)}."
            )

        x_rows = []
        for node_name in self.node_order:
            abundance = float(scaled_microbes.get(node_name, 0.0))
            function_score = self._predict_function_score(node_name, abundance)
            x_rows.append([abundance, function_score])
        x = torch.tensor(x_rows, dtype=torch.float32)

        node_to_idx = {name: idx for idx, name in enumerate(self.node_order)}
        edges_src: list[int] = []
        edges_dst: list[int] = []
        edge_attr: list[list[float]] = []
        for src, dst in self.edge_pairs:
            weight = float(self.edge_weight_priors[(src, dst)])
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            edges_src.extend([src_idx, dst_idx])
            edges_dst.extend([dst_idx, src_idx])
            edge_attr.extend([[weight], [weight]])

        clinical_values = [
            float(clinical.get(column, self.clinical_defaults[column]))
            for column in self.clinical_columns
        ]
        metabolite_values = [
            float(metabolites.get(column, self.metabolite_defaults[column]))
            for column in self.metabolite_columns
        ]

        unsupported_microbes = sorted(set(microbes).difference(self.node_order))
        data = Data(
            x=x,
            edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            clinical=torch.tensor(clinical_values, dtype=torch.float32),
            metabolites=torch.tensor(metabolite_values, dtype=torch.float32),
            time=torch.tensor(0.0, dtype=torch.float32),
            event=torch.tensor(0.0, dtype=torch.float32),
            sample_id="WEB_INPUT",
        )
        data.unsupported_microbes = unsupported_microbes
        return data

    def score(
        self,
        microbes: Dict[str, float],
        clinical: Dict[str, float],
        metabolites: Dict[str, float],
    ) -> ResearchModelPrediction:
        sample = self._build_single_sample(microbes, clinical, metabolites)
        batch = Batch.from_data_list([sample]).to(self.device)

        raw_risks = []
        graph_embedding_norms = []
        with torch.no_grad():
            for model in self.models:
                output = model(batch, compute_contrastive=False)
                raw_risks.append(float(output["risk"].item()))
                graph_embedding_norms.append(float(output["graph_embedding"].norm(dim=1).mean().item()))

        raw_model_risk = float(np.mean(raw_risks))
        risk_percentile = float(100.0 * np.mean(self.reference_risks <= raw_model_risk))
        if raw_model_risk < self.low_threshold:
            risk_level = "low"
        elif raw_model_risk < self.medium_threshold:
            risk_level = "medium"
        else:
            risk_level = "high"

        supported_microbes = {name: float(microbes.get(name, 0.0)) for name in self.node_order}
        risk_result = {
            "risk_score": round(risk_percentile, 2),
            "risk_level": risk_level,
            "risk_percentile": round(risk_percentile, 2),
            "raw_model_risk": round(raw_model_risk, 6),
            "ensemble_size": len(self.models),
            "backend": "research_gnn_cox_ensemble" if len(self.models) > 1 else "research_gnn_cox_single",
        }
        model_features = {
            "backend": risk_result["backend"],
            "ensemble_size": len(self.models),
            "supported_microbes": self.node_order,
            "supported_microbe_input": supported_microbes,
            "unsupported_microbes_ignored": list(sample.unsupported_microbes),
            "graph_embedding_norm": round(float(np.mean(graph_embedding_norms)), 6),
            "reference_low_threshold_raw_risk": round(self.low_threshold, 6),
            "reference_medium_threshold_raw_risk": round(self.medium_threshold, 6),
            "reference_abundance_total": round(self.reference_abundance_total, 6),
            "checkpoint_sources": [str(path.as_posix()) for path in self.checkpoints],
        }
        return ResearchModelPrediction(risk_result=risk_result, model_features=model_features)


@lru_cache(maxsize=1)
def get_research_model_bridge() -> ResearchModelBridge:
    return ResearchModelBridge()
