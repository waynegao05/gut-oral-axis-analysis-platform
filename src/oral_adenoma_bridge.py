from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ALLOWED_SAMPLE_TYPES = {"oral", "oral_swab", "buccal_swab", "saliva"}
FORBIDDEN_SAMPLE_TOKENS = {
    "stool",
    "fecal",
    "faecal",
    "gut",
    "intestinal",
    "blood",
    "serum",
    "plasma",
    "tissue",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_sample_type(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in normalized for token in FORBIDDEN_SAMPLE_TOKENS):
        raise ValueError("仅允许口腔拭子或唾液数据，禁止粪便、血液和组织数据。")
    if normalized not in ALLOWED_SAMPLE_TYPES:
        raise ValueError("sample_type 只能是 oral、oral_swab、buccal_swab 或 saliva。")
    return normalized


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True)
class OralAdenomaPrediction:
    probability: float
    threshold: float
    screen_positive: bool
    sample_type: str
    release_name: str
    selected_taxonomies: tuple[str, ...]
    formal_internal_metrics: Mapping[str, Any]
    claim_boundary: str

    def as_dict(self) -> dict[str, Any]:
        sensitivity = self.formal_internal_metrics["adenoma_sensitivity"]
        false_positive_rate = self.formal_internal_metrics["false_positive_rate"]
        specificity = self.formal_internal_metrics["specificity"]
        roc_auc = self.formal_internal_metrics["roc_auc"]
        return {
            "prediction_available": True,
            "endpoint": "oral_microbiome_adenoma_screening_research",
            "model_release": self.release_name,
            "sample_type": self.sample_type,
            "adenoma_probability": self.probability,
            "operating_threshold": self.threshold,
            "screen_positive": self.screen_positive,
            "result_label": "内部研究筛查阳性" if self.screen_positive else "内部研究筛查阴性",
            "formal_internal_metrics": {
                "adenoma_sensitivity": sensitivity,
                "false_positive_rate": false_positive_rate,
                "specificity": specificity,
                "roc_auc": roc_auc,
            },
            "selected_taxonomies": list(self.selected_taxonomies),
            "research_only": True,
            "not_diagnostic": True,
            "verified_diminutive_adenoma_le_5mm": False,
            "claim_boundary": self.claim_boundary,
        }


class OralAdenomaBridge:
    def __init__(self, artifact_path: Path, expected_sha256: str) -> None:
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Oral adenoma release artifact not found: {artifact_path}")
        actual_sha256 = _sha256(artifact_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Oral adenoma release SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("research_only") is not True:
            raise ValueError("Oral adenoma release must remain research-only.")
        if artifact.get("release_name") != "oral_adenoma_internal_v3":
            raise ValueError("Unexpected oral adenoma release name.")

        self.artifact = artifact
        self.release_name = str(artifact["release_name"])
        self.feature_ids = tuple(str(value) for value in artifact["feature_ids"])
        self.taxonomies = tuple(str(value) for value in artifact["taxonomies"])
        if len(self.feature_ids) != len(self.taxonomies):
            raise ValueError("Oral adenoma release has inconsistent feature metadata.")
        self.taxonomy_to_index = {
            taxonomy: index for index, taxonomy in enumerate(self.taxonomies)
        }
        if len(self.taxonomy_to_index) != len(self.taxonomies):
            raise ValueError("Oral adenoma release taxonomies must be unique.")

        preprocessing = artifact["preprocessing"]
        self.pseudocount = float(preprocessing["pseudocount_percent"])
        self.sum_min, self.sum_max = [
            float(value) for value in preprocessing["required_sum_range_percent"]
        ]
        self.scaler_mean = np.asarray(preprocessing["scaler_mean"], dtype=float)
        self.scaler_scale = np.asarray(preprocessing["scaler_scale"], dtype=float)
        self.selected_indices = np.asarray(preprocessing["selected_indices"], dtype=int)
        self.coefficient = np.asarray(artifact["model"]["coefficient"], dtype=float)
        self.intercept = float(artifact["model"]["intercept"])
        self.threshold = float(artifact["operating_threshold"])

        feature_count = len(self.feature_ids)
        if self.scaler_mean.shape != (feature_count,) or self.scaler_scale.shape != (
            feature_count,
        ):
            raise ValueError("Oral adenoma scaler dimensions do not match the feature list.")
        if self.coefficient.shape != self.selected_indices.shape:
            raise ValueError("Oral adenoma coefficient dimensions do not match selection.")
        arrays = (self.scaler_mean, self.scaler_scale, self.coefficient)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("Oral adenoma release contains non-finite weights.")
        if (self.scaler_scale <= 0.0).any():
            raise ValueError("Oral adenoma release contains an invalid scaler value.")

    @staticmethod
    def _coerce_abundance(value: object, field: str) -> float:
        if isinstance(value, bool) or value is None or value == "":
            raise ValueError(f"oral_abundances.{field} 必须是有效数字。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"oral_abundances.{field} 必须是有效数字。") from exc
        if not math.isfinite(number):
            raise ValueError(f"oral_abundances.{field} 不能是 NaN 或 Infinity。")
        if number < 0.0 or number > 100.0:
            raise ValueError(f"oral_abundances.{field} 必须是 0 到 100 之间的百分比。")
        return number

    def _vectorize(self, abundances: Mapping[str, object]) -> np.ndarray:
        if not isinstance(abundances, Mapping):
            raise ValueError("oral_abundances 必须是菌属到百分比的 JSON 对象。")
        supplied = {str(key).strip(): value for key, value in abundances.items()}
        unknown = sorted(set(supplied).difference(self.taxonomy_to_index))
        missing = sorted(set(self.taxonomies).difference(supplied))
        if unknown:
            raise ValueError(f"包含模型不认识的口腔菌属：{unknown[:5]}。")
        if missing:
            raise ValueError(
                f"口腔腺瘤模型需要完整的 {len(self.taxonomies)} 个菌属，当前缺少 "
                f"{len(missing)} 个，例如：{missing[:5]}。"
            )
        values = np.asarray(
            [self._coerce_abundance(supplied[taxonomy], taxonomy) for taxonomy in self.taxonomies],
            dtype=float,
        )
        total = float(values.sum())
        if total < self.sum_min or total > self.sum_max:
            raise ValueError(
                "口腔菌属百分比之和必须约为 100%，"
                f"当前合计为 {total:.6g}%。"
            )
        return values

    def score(
        self,
        abundances: Mapping[str, object],
        *,
        sample_type: object,
    ) -> OralAdenomaPrediction:
        normalized_sample_type = _normalize_sample_type(sample_type)
        values = self._vectorize(abundances)
        logged = np.log(values + self.pseudocount)
        clr = logged - float(logged.mean())
        scaled = (clr - self.scaler_mean) / self.scaler_scale
        selected = scaled[self.selected_indices]
        decision = float(np.dot(selected, self.coefficient) + self.intercept)
        probability = _sigmoid(decision)
        return OralAdenomaPrediction(
            probability=probability,
            threshold=self.threshold,
            screen_positive=probability >= self.threshold,
            sample_type=normalized_sample_type,
            release_name=self.release_name,
            selected_taxonomies=tuple(self.artifact["selected_taxonomies"]),
            formal_internal_metrics=self.artifact["formal_internal_metrics"],
            claim_boundary=str(self.artifact["claim_boundary"]),
        )


@lru_cache(maxsize=1)
def get_oral_adenoma_bridge() -> OralAdenomaBridge:
    from config.settings import (
        ORAL_ADENOMA_ARTIFACT_PATH,
        ORAL_ADENOMA_ARTIFACT_SHA256,
    )

    return OralAdenomaBridge(
        ORAL_ADENOMA_ARTIFACT_PATH,
        ORAL_ADENOMA_ARTIFACT_SHA256,
    )
