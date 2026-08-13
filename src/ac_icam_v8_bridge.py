from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from config.settings import (
    AC_ICAM_V8_ARTIFACT_SHA256,
    AC_ICAM_V8_ARTIFACT_PATH,
    AC_ICAM_V8_RELEASE_NAME,
    AC_ICAM_V8_RELEASE_NOTE,
    TEMPORAL_TOPOLOGY_RELEASE_NAME,
)


@dataclass(frozen=True)
class V8ModelPrediction:
    risk_result: dict[str, object]
    model_features: dict[str, object]
    general_risk_result: dict[str, object] | None = None
    general_risk_features: dict[str, object] | None = None


WEB_REQUIRED_CLINICAL_FIELDS = ("age", "sex")
PFS_REQUIRED_ONCOLOGY_FIELDS = (
    "stage",
    "path_t",
    "path_n",
    "path_m",
    "tumor_location",
    "tumor_morphology",
)
PFS_REQUIRED_CLINICAL_FIELDS = (
    *WEB_REQUIRED_CLINICAL_FIELDS,
    *PFS_REQUIRED_ONCOLOGY_FIELDS,
)
GENERAL_RISK_REQUIRED_MICROBES = (
    "Fusobacterium",
    "Porphyromonas",
    "Prevotella",
    "Streptococcus",
    "Lactobacillus",
)


SEX_ALIASES = {
    "female": "Female",
    "f": "Female",
    "女": "Female",
    "male": "Male",
    "m": "Male",
    "男": "Male",
}

LOCATION_ALIASES = {
    "ceceum": "Ceceum",
    "cecum": "Ceceum",
    "盲肠": "Ceceum",
    "colon ascendens": "Colon Ascendens",
    "ascending colon": "Colon Ascendens",
    "升结肠": "Colon Ascendens",
    "colon descendens": "Colon Descendens",
    "descending colon": "Colon Descendens",
    "降结肠": "Colon Descendens",
    "colon sigmoideum": "Colon Sigmoideum",
    "sigmoid colon": "Colon Sigmoideum",
    "乙状结肠": "Colon Sigmoideum",
    "colon transversum": "Colon Transversum",
    "transverse colon": "Colon Transversum",
    "横结肠": "Colon Transversum",
    "flexura hepatica": "Flexura Hepatica",
    "hepatic flexure": "Flexura Hepatica",
    "肝曲": "Flexura Hepatica",
    "flexura lienalis": "Flexura Lienalis",
    "splenic flexure": "Flexura Lienalis",
    "脾曲": "Flexura Lienalis",
    "rectosigmoideum": "Rectosigmoideum",
    "rectosigmoid": "Rectosigmoideum",
    "直乙交界": "Rectosigmoideum",
}

MORPHOLOGY_ALIASES = {
    "adenocarcinoma": "Adenocarcinoma",
    "腺癌": "Adenocarcinoma",
    "adenocarcinoma in villeus adenoma": "Adenocarcinoma In Villeus Adenoma",
    "villous adenoma with adenocarcinoma": "Adenocarcinoma In Villeus Adenoma",
    "绒毛状腺瘤伴腺癌": "Adenocarcinoma In Villeus Adenoma",
    "adenocarcinoma intestinal type": "Adenocarcinoma Intestinal Type",
    "intestinal type adenocarcinoma": "Adenocarcinoma Intestinal Type",
    "肠型腺癌": "Adenocarcinoma Intestinal Type",
    "adenocarcinoma with mixed subtypes": "Adenocarcinoma With Mixed Subtypes",
    "mixed adenocarcinoma": "Adenocarcinoma With Mixed Subtypes",
    "混合亚型腺癌": "Adenocarcinoma With Mixed Subtypes",
    "cribriform carcinoma": "Cribriform Carcinoma",
    "筛状癌": "Cribriform Carcinoma",
    "mucineus adenocarcinoma": "Mucineus Adenocarcinoma",
    "mucinous adenocarcinoma": "Mucineus Adenocarcinoma",
    "黏液腺癌": "Mucineus Adenocarcinoma",
    "signet ring cell carcinoma": "Signet Ring Cell Carcinoma",
    "印戒细胞癌": "Signet Ring Cell Carcinoma",
}


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"clinical.{field} 必须是有效数字。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"clinical.{field} 必须是有效数字。") from exc
    if not math.isfinite(number):
        raise ValueError(f"clinical.{field} 不能是 NaN 或 Infinity。")
    return number


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _normalize_category(
    value: Any,
    *,
    field: str,
    aliases: Mapping[str, str],
    allowed: set[str],
) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"clinical.{field} 不能为空。")
    normalized = aliases.get(text.casefold(), text)
    if normalized not in allowed:
        choices = "、".join(sorted(allowed))
        raise ValueError(
            f"clinical.{field} 不在 V8 训练类别中。可选值：{choices}。"
        )
    return normalized


class ACICAMV8ModelBridge:
    def __init__(self, artifact_path: Path = AC_ICAM_V8_ARTIFACT_PATH) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                "AC-ICAM V8 deployment artifact is missing. Run "
                "`python -m experiments.ac_icam_real_outcome_v8.deployment`."
            )
        artifact_bytes = self.artifact_path.read_bytes()
        self.artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if self.artifact_sha256 != AC_ICAM_V8_ARTIFACT_SHA256:
            raise RuntimeError(
                "AC-ICAM V8 deployment artifact SHA-256 does not match the "
                "locked release in config.settings."
            )
        self.artifact = json.loads(
            artifact_bytes.decode("utf-8")
        )
        if self.artifact.get("schema_version") != 1:
            raise RuntimeError("Unsupported AC-ICAM V8 artifact schema.")
        if self.artifact.get("release_name") != AC_ICAM_V8_RELEASE_NAME:
            raise RuntimeError(
                "AC-ICAM V8 artifact release does not match config.settings."
            )

    @staticmethod
    def _variant_levels(variant: Mapping[str, Any]) -> dict[str, set[str]]:
        first = variant["members"][0]["transformer"]
        return {
            str(column): set(str(value) for value in levels)
            for column, levels in zip(
                first["categorical_columns"],
                first["category_levels"],
            )
        }

    def _missing_oncology_prediction(
        self,
        *,
        microbes: Mapping[str, float],
        clinical: Mapping[str, Any],
        metabolites: Mapping[str, float],
        missing_fields: list[str],
    ) -> V8ModelPrediction:
        core_variant = self.artifact["variants"]["clinical_core"]
        levels = self._variant_levels(core_variant)
        used_clinical_inputs = {
            "age": _finite_float(clinical["age"], field="age"),
            "sex": _normalize_category(
                clinical["sex"],
                field="sex",
                aliases=SEX_ALIASES,
                allowed=levels["sex"],
            ),
        }
        backend_name = str(self.artifact["backend"])
        missing_paths = [f"clinical.{field}" for field in missing_fields]
        intended_use = (
            "未提供完整结直肠癌病理资料，本次不计算 PFS。年龄、性别、菌群和"
            "用药信息仍可用于非 PFS 的展示与辅助核对。"
        )
        risk_result: dict[str, object] = {
            "risk_score": None,
            "risk_level": "not_available",
            "risk_percentile": None,
            "raw_model_risk": None,
            "prediction_reliability": "not_applicable_missing_oncology",
            "prediction_available": False,
            "not_available_reason": "missing_oncology_fields",
            "missing_oncology_fields": missing_paths,
            "ensemble_size": 0,
            "backend": backend_name,
            "model_release": AC_ICAM_V8_RELEASE_NAME,
            "model_variant": "not_calculated",
            "endpoint": "PFS",
            "pfs_probability": {"36": None, "60": None},
            "progression_probability": {"36": None, "60": None},
            "time_horizon_unit": "months",
            "intended_use": intended_use,
            "research_use_only": True,
        }
        model_features: dict[str, object] = {
            "backend": backend_name,
            "model_release": AC_ICAM_V8_RELEASE_NAME,
            "model_release_note": AC_ICAM_V8_RELEASE_NOTE,
            "model_variant": "not_calculated",
            "endpoint": "PFS",
            "reference_population": self.artifact["scope"],
            "training_cohort": self.artifact["training_cohort"],
            "required_web_fields": list(WEB_REQUIRED_CLINICAL_FIELDS),
            "required_pfs_fields": list(PFS_REQUIRED_CLINICAL_FIELDS),
            "required_clinical_fields": list(
                PFS_REQUIRED_CLINICAL_FIELDS
            ),
            "artifact_required_clinical_fields": self.artifact[
                "required_clinical_fields"
            ],
            "optional_clinical_fields": ["icr_score"],
            "artifact_optional_clinical_fields": self.artifact[
                "optional_clinical_fields"
            ],
            "pfs_model_eligible": False,
            "missing_oncology_fields": missing_paths,
            "used_clinical_inputs": used_clinical_inputs,
            "defaulted_inputs": [],
            "out_of_training_range_inputs": [],
            "out_of_training_range_details": [],
            "member_standardized_risks": [],
            "member_disagreement": None,
            "member_disagreement_p90": None,
            "microbiome_used_for_risk": False,
            "microbiome_role": (
                "Submitted microbes remain available for descriptive and pharmacy "
                "support modules; no V8 PFS score was calculated."
            ),
            "submitted_microbe_count": int(len(microbes)),
            "treatment_used_for_risk": False,
            "icr_used_for_risk": False,
            "formal_metrics": None,
            "deployment_policy": self.artifact["deployment_policy"],
            "limitations": self.artifact["limitations"],
            "artifact_source": f"config/releases/{self.artifact_path.name}",
            "artifact_sha256": self.artifact_sha256,
        }
        general_risk_result, general_risk_features = (
            self._general_risk_prediction(
                microbes=microbes,
                clinical=clinical,
                metabolites=metabolites,
            )
        )
        return V8ModelPrediction(
            risk_result=risk_result,
            model_features=model_features,
            general_risk_result=general_risk_result,
            general_risk_features=general_risk_features,
        )

    @staticmethod
    def _general_risk_unavailable(
        *,
        reason: str,
        missing_microbes: list[str],
    ) -> tuple[dict[str, object], dict[str, object]]:
        missing_paths = [f"microbes.{name}" for name in missing_microbes]
        result: dict[str, object] = {
            "risk_score": None,
            "risk_level": "not_available",
            "risk_percentile": None,
            "raw_model_risk": None,
            "prediction_reliability": "not_applicable_incomplete_microbiome",
            "prediction_available": False,
            "not_available_reason": reason,
            "backend": "temporal_topology_aft_cross_split_consensus",
            "model_release": TEMPORAL_TOPOLOGY_RELEASE_NAME,
            "model_variant": "temporal_topology_research_percentile",
            "endpoint": "research_risk_index",
            "display_name": "菌群-临床研究风险指数",
            "risk_kind": "research_cohort_percentile",
            "score_unit": "reference_percentile_0_100",
            "required_microbes": list(GENERAL_RISK_REQUIRED_MICROBES),
            "missing_microbe_fields": missing_paths,
            "absolute_cancer_probability": False,
            "screening_result": False,
            "pfs_calculated": False,
            "dataset_version": "topology_v6",
            "dataset_is_synthetic_noisy_augmented": True,
            "intended_use": (
                "用于展示菌群与可用健康信息在研究参考队列中的相对位置；"
                "不是结直肠癌患病概率、筛查结果、诊断或 PFS 预测。"
            ),
            "research_use_only": True,
        }
        features: dict[str, object] = {
            "backend": result["backend"],
            "model_release": result["model_release"],
            "model_variant": result["model_variant"],
            "endpoint": result["endpoint"],
            "general_risk_model_eligible": False,
            "prediction_available": False,
            "not_available_reason": reason,
            "required_microbes": list(GENERAL_RISK_REQUIRED_MICROBES),
            "missing_microbe_fields": missing_paths,
            "submitted_microbe_count": int(
                len(GENERAL_RISK_REQUIRED_MICROBES) - len(missing_microbes)
            ),
            "defaulted_inputs": [],
            "out_of_training_range_inputs": [],
            "out_of_training_range_details": [],
            "sex_used_for_risk": False,
            "dataset_version": "topology_v6",
            "dataset_is_synthetic_noisy_augmented": True,
        }
        return result, features

    def _general_risk_prediction(
        self,
        *,
        microbes: Mapping[str, float],
        clinical: Mapping[str, Any],
        metabolites: Mapping[str, float],
    ) -> tuple[dict[str, object], dict[str, object]]:
        missing_microbes = [
            name
            for name in GENERAL_RISK_REQUIRED_MICROBES
            if name not in microbes
        ]
        if missing_microbes:
            return self._general_risk_unavailable(
                reason="incomplete_microbiome_panel",
                missing_microbes=missing_microbes,
            )

        panel_total = sum(
            max(float(microbes[name]), 0.0)
            for name in GENERAL_RISK_REQUIRED_MICROBES
        )
        if panel_total <= 0.0:
            return self._general_risk_unavailable(
                reason="invalid_microbiome_panel",
                missing_microbes=[],
            )

        from src.temporal_topology_bridge import (
            get_temporal_topology_model_bridge,
        )

        prediction = get_temporal_topology_model_bridge().score(
            dict(microbes),
            dict(clinical),
            dict(metabolites),
        )
        result = dict(prediction.risk_result)
        features = dict(prediction.model_features)
        out_of_range = list(features.get("out_of_training_range_inputs", []))
        prediction_available = not bool(out_of_range)
        if not prediction_available:
            result["risk_score"] = None
            result["risk_level"] = "not_available"
            result["risk_percentile"] = None
            result["raw_model_risk"] = None
            result["not_available_reason"] = "out_of_training_range"

        result.update(
            {
                "prediction_available": prediction_available,
                "model_variant": "temporal_topology_research_percentile",
                "endpoint": "research_risk_index",
                "display_name": "菌群-临床研究风险指数",
                "risk_kind": "research_cohort_percentile",
                "score_unit": "reference_percentile_0_100",
                "required_microbes": list(GENERAL_RISK_REQUIRED_MICROBES),
                "missing_microbe_fields": [],
                "absolute_cancer_probability": False,
                "screening_result": False,
                "pfs_calculated": False,
                "dataset_version": "topology_v6",
                "dataset_is_synthetic_noisy_augmented": True,
                "intended_use": (
                    "用于展示菌群与可用健康信息在研究参考队列中的相对位置；"
                    "不是结直肠癌患病概率、筛查结果、诊断或 PFS 预测。"
                ),
                "research_use_only": True,
            }
        )
        features.update(
            {
                "model_variant": result["model_variant"],
                "endpoint": result["endpoint"],
                "general_risk_model_eligible": prediction_available,
                "prediction_available": prediction_available,
                "required_microbes": list(GENERAL_RISK_REQUIRED_MICROBES),
                "missing_microbe_fields": [],
                "sex_used_for_risk": False,
                "dataset_version": "topology_v6",
                "dataset_is_synthetic_noisy_augmented": True,
            }
        )
        if not prediction_available:
            features["not_available_reason"] = "out_of_training_range"
        return result, features

    def _prepare_clinical(
        self,
        clinical: Mapping[str, Any],
        *,
        variant_name: str,
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        required = list(self.artifact["required_clinical_fields"])
        missing = [
            field
            for field in required
            if field not in clinical or _is_missing(clinical[field])
        ]
        if missing:
            raise ValueError(
                "V8 PFS 模型缺少必要字段："
                + "、".join(f"clinical.{field}" for field in missing)
            )

        variant = self.artifact["variants"][variant_name]
        levels = self._variant_levels(variant)
        prepared: dict[str, Any] = {}
        numeric_fields = (
            "age",
            "stage",
            "path_t",
            "path_n",
            "path_m",
            "icr_score",
        )
        for field in numeric_fields:
            if field in clinical and not _is_missing(clinical[field]):
                prepared[field] = _finite_float(clinical[field], field=field)

        prepared["sex"] = _normalize_category(
            clinical["sex"],
            field="sex",
            aliases=SEX_ALIASES,
            allowed=levels["sex"],
        )
        prepared["tumor_location"] = _normalize_category(
            clinical["tumor_location"],
            field="tumor_location",
            aliases=LOCATION_ALIASES,
            allowed=levels["tumor_location"],
        )
        prepared["tumor_morphology"] = _normalize_category(
            clinical["tumor_morphology"],
            field="tumor_morphology",
            aliases=MORPHOLOGY_ALIASES,
            allowed=levels["tumor_morphology"],
        )

        defaulted: list[str] = []
        if "path_m" not in prepared:
            defaulted.append("clinical.path_m")

        out_of_range: list[str] = []
        details: list[dict[str, Any]] = []
        ranges = self.artifact["numeric_training_ranges"]
        for field, value in prepared.items():
            if field not in ranges or not isinstance(value, float):
                continue
            minimum = float(ranges[field]["minimum"])
            maximum = float(ranges[field]["maximum"])
            if value < minimum or value > maximum:
                path = f"clinical.{field}"
                out_of_range.append(path)
                details.append(
                    {
                        "field": path,
                        "value": float(value),
                        "training_minimum": minimum,
                        "training_maximum": maximum,
                    }
                )
        return prepared, defaulted, out_of_range, details

    @staticmethod
    def _member_input(
        member: Mapping[str, Any],
        clinical: Mapping[str, Any],
        defaulted: list[str],
    ) -> np.ndarray:
        transformer = member["transformer"]
        numeric_values: list[float] = []
        for column, median in zip(
            transformer["numeric_columns"],
            transformer["numeric_medians"],
        ):
            if column in clinical:
                numeric_values.append(float(clinical[column]))
            else:
                numeric_values.append(float(median))
                path = f"clinical.{column}"
                if path not in defaulted:
                    defaulted.append(path)
        values = list(numeric_values)
        for column, levels in zip(
            transformer["categorical_columns"],
            transformer["category_levels"],
        ):
            observed = str(clinical[column])
            values.extend(1.0 if observed == level else 0.0 for level in levels)
        return np.asarray(values, dtype=float)

    @staticmethod
    def _score_member(
        member: Mapping[str, Any],
        values: np.ndarray,
    ) -> tuple[float, float, dict[str, float]]:
        model = member["cox_model"]
        mean = np.asarray(model["mean"], dtype=float)
        scale = np.asarray(model["scale"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        raw = float(((values - mean) / scale) @ coefficients)
        calibration = member["risk_calibration"]
        standardized = (
            raw - float(calibration["mean"])
        ) / float(calibration["scale"])
        survival = {
            horizon: float(
                np.exp(
                    -float(cumulative_hazard)
                    * np.exp(np.clip(raw, -30.0, 30.0))
                )
            )
            for horizon, cumulative_hazard in member[
                "breslow_cumulative_hazard"
            ].items()
        }
        return raw, float(standardized), survival

    def score(
        self,
        microbes: Mapping[str, float],
        clinical: Mapping[str, Any],
        metabolites: Mapping[str, float],
    ) -> V8ModelPrediction:
        missing_web_fields = [
            field
            for field in WEB_REQUIRED_CLINICAL_FIELDS
            if field not in clinical or _is_missing(clinical[field])
        ]
        if missing_web_fields:
            raise ValueError(
                "网页分析缺少必填字段："
                + "、".join(
                    f"clinical.{field}" for field in missing_web_fields
                )
            )

        missing_oncology_fields = [
            field
            for field in PFS_REQUIRED_ONCOLOGY_FIELDS
            if field not in clinical or _is_missing(clinical[field])
        ]
        if missing_oncology_fields:
            return self._missing_oncology_prediction(
                microbes=microbes,
                clinical=clinical,
                metabolites=metabolites,
                missing_fields=missing_oncology_fields,
            )

        has_icr = (
            "icr_score" in clinical
            and not _is_missing(clinical["icr_score"])
        )
        variant_name = "clinical_icr" if has_icr else "clinical_core"
        variant = self.artifact["variants"][variant_name]
        prepared, defaulted, out_of_range, range_details = self._prepare_clinical(
            clinical,
            variant_name=variant_name,
        )

        standardized_risks: list[float] = []
        survival_rows: list[dict[str, float]] = []
        for member in variant["members"]:
            values = self._member_input(member, prepared, defaulted)
            _, standardized, survival = self._score_member(member, values)
            standardized_risks.append(standardized)
            survival_rows.append(survival)

        deployment_risk = float(np.mean(standardized_risks))
        mapping = variant["deployment_to_oof_calibration"]
        calibrated_risk = (
            (
                deployment_risk - float(mapping["source_mean"])
            )
            / float(mapping["source_scale"])
            * float(mapping["target_scale"])
            + float(mapping["target_mean"])
        )
        reference = np.asarray(variant["reference_oof_risks"], dtype=float)
        risk_percentile = float(100.0 * np.mean(reference <= calibrated_risk))
        thresholds = variant["risk_thresholds"]
        if calibrated_risk < float(thresholds["low_upper"]):
            risk_level = "low"
        elif calibrated_risk < float(thresholds["medium_upper"]):
            risk_level = "medium"
        else:
            risk_level = "high"

        disagreement = float(np.std(standardized_risks))
        if out_of_range:
            reliability = "caution_out_of_training_range"
        elif defaulted:
            reliability = "caution_defaulted_inputs"
        elif disagreement > float(variant["member_disagreement_p90"]):
            reliability = "caution_split_disagreement"
        else:
            reliability = "standard"

        pfs_probability = {
            horizon: float(
                np.mean([row[horizon] for row in survival_rows])
            )
            for horizon in ("36", "60")
        }
        backend_name = str(self.artifact["backend"])
        risk_result: dict[str, object] = {
            "risk_score": round(risk_percentile, 2),
            "risk_level": risk_level,
            "risk_percentile": round(risk_percentile, 2),
            "raw_model_risk": round(calibrated_risk, 6),
            "prediction_reliability": reliability,
            "prediction_available": not bool(out_of_range),
            "ensemble_size": int(len(variant["members"])),
            "backend": backend_name,
            "model_release": AC_ICAM_V8_RELEASE_NAME,
            "model_variant": variant_name,
            "endpoint": "PFS",
            "pfs_probability": {
                horizon: round(probability, 6)
                for horizon, probability in pfs_probability.items()
            },
            "progression_probability": {
                horizon: round(1.0 - probability, 6)
                for horizon, probability in pfs_probability.items()
            },
            "time_horizon_unit": "months",
            "intended_use": (
                "适用于已确诊 AJCC I-IV 期结直肠癌患者的相对 PFS 风险估计。"
            ),
            "research_use_only": True,
        }
        model_features: dict[str, object] = {
            "backend": backend_name,
            "model_release": AC_ICAM_V8_RELEASE_NAME,
            "model_release_note": AC_ICAM_V8_RELEASE_NOTE,
            "model_variant": variant_name,
            "endpoint": "PFS",
            "reference_population": self.artifact["scope"],
            "training_cohort": self.artifact["training_cohort"],
            "required_web_fields": list(WEB_REQUIRED_CLINICAL_FIELDS),
            "required_pfs_fields": list(PFS_REQUIRED_CLINICAL_FIELDS),
            "required_clinical_fields": list(
                PFS_REQUIRED_CLINICAL_FIELDS
            ),
            "artifact_required_clinical_fields": self.artifact[
                "required_clinical_fields"
            ],
            "optional_clinical_fields": ["icr_score"],
            "artifact_optional_clinical_fields": self.artifact[
                "optional_clinical_fields"
            ],
            "pfs_model_eligible": True,
            "missing_oncology_fields": [],
            "used_clinical_inputs": prepared,
            "defaulted_inputs": sorted(defaulted),
            "out_of_training_range_inputs": sorted(out_of_range),
            "out_of_training_range_details": range_details,
            "member_standardized_risks": [
                round(value, 6) for value in standardized_risks
            ],
            "member_disagreement": round(disagreement, 6),
            "member_disagreement_p90": round(
                float(variant["member_disagreement_p90"]),
                6,
            ),
            "microbiome_used_for_risk": False,
            "microbiome_role": (
                "Submitted microbes remain available for descriptive and pharmacy "
                "support modules but do not change the V8 PFS score."
            ),
            "submitted_microbe_count": int(len(microbes)),
            "treatment_used_for_risk": False,
            "icr_used_for_risk": bool(has_icr),
            "formal_metrics": variant["formal_metrics"],
            "deployment_policy": self.artifact["deployment_policy"],
            "limitations": self.artifact["limitations"],
            "artifact_source": f"config/releases/{self.artifact_path.name}",
            "artifact_sha256": self.artifact_sha256,
        }
        return V8ModelPrediction(
            risk_result=risk_result,
            model_features=model_features,
        )


@lru_cache(maxsize=1)
def get_ac_icam_v8_model_bridge() -> ACICAMV8ModelBridge:
    return ACICAMV8ModelBridge()
