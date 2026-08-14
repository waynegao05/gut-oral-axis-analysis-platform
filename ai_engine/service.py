from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

from config.settings import (
    AC_ICAM_V8_ARTIFACT_PATH,
    AC_ICAM_V8_RELEASE_NAME,
    ENABLE_INTERNAL_ORAL_ADENOMA,
    ORAL_ADENOMA_ARTIFACT_PATH,
    ORAL_ADENOMA_RELEASE_NAME,
    RESEARCH_MODEL_CONFIG_PATH,
    RESEARCH_MODEL_RELEASE_NAME,
    TEMPORAL_TOPOLOGY_FULL_RISK_ROOT,
    TEMPORAL_TOPOLOGY_RELEASE_NAME,
    TEMPORAL_TOPOLOGY_ROOT,
    WEB_MODEL_BACKEND,
)
from src.clinical_standardizer import standardize_raw_payload
from src.logging_utils import get_logger
from src.pharmacy_engine import load_pharmacy_knowledge_base
from src.pipeline import run_pipeline
from src.validators import (
    REQUIRED_TOP_LEVEL_KEYS,
    V8_WEB_REQUIRED_CLINICAL_FIELDS,
    validate_payload,
)

from ai_engine.errors import (
    ArtifactIntegrityError,
    CapabilityUnavailableError,
    EngineError,
    FeatureDisabledError,
    InvalidInputError,
    ModelInferenceError,
    ModelNotLoadedError,
)


PipelineRunner = Callable[[dict[str, Any]], dict[str, object]]
ModelLoader = Callable[[], object]


@dataclass(frozen=True)
class StandardizationResult:
    source_format: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CapabilityStatus:
    enabled: bool
    available: bool
    loaded: bool
    reason: str | None = None
    missing_requirements: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "loaded": self.loaded,
            "reason": self.reason,
            "missing_requirements": list(self.missing_requirements),
        }


def _active_release_name(backend: str) -> str:
    return {
        "ac_icam_v8": AC_ICAM_V8_RELEASE_NAME,
        "temporal_topology": TEMPORAL_TOPOLOGY_RELEASE_NAME,
        "legacy_cox": RESEARCH_MODEL_RELEASE_NAME,
    }.get(backend, backend)


def _load_model_for_backend(backend: str) -> object:
    if backend == "ac_icam_v8":
        from src.ac_icam_v8_bridge import get_ac_icam_v8_model_bridge

        return get_ac_icam_v8_model_bridge()
    if backend == "temporal_topology":
        from src.temporal_topology_bridge import get_temporal_topology_model_bridge

        return get_temporal_topology_model_bridge()
    if backend == "legacy_cox":
        from archive.legacy_web_backends.cox_ensemble_v1 import (
            get_research_model_bridge,
        )

        return get_research_model_bridge()
    raise RuntimeError(f"Unsupported GOA_MODEL_BACKEND: {backend!r}.")


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class AIService:
    """Stable service boundary around the existing model and pharmacy pipeline."""

    def __init__(
        self,
        *,
        backend: str = WEB_MODEL_BACKEND,
        pipeline_runner: PipelineRunner = run_pipeline,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.backend = backend
        self._pipeline_runner = pipeline_runner
        self._model_loader = model_loader or (
            lambda: _load_model_for_backend(self.backend)
        )
        self._state_lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        self._initialization_attempted = False
        self._model_loaded = False
        self._general_risk_loaded = False
        self._oral_adenoma_loaded = False
        self._initialization_error: str | None = None
        self._model: object | None = None
        self._logger = get_logger("gut_oral_axis.ai_engine")

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_release(self) -> str:
        return _active_release_name(self.backend)

    def initialize(self) -> bool:
        with self._state_lock:
            if self._initialization_attempted:
                return self._model_loaded
            self._initialization_attempted = True
            try:
                self._model = self._model_loader()
            except Exception as exc:
                self._initialization_error = type(exc).__name__
                self._logger.exception("Primary model initialization failed.")
                return False
            self._model_loaded = True
            self._initialization_error = None
            return True

    def standardize(self, payload: object) -> StandardizationResult:
        if not isinstance(payload, dict):
            raise InvalidInputError.from_messages(["输入必须是 JSON 对象。"])

        candidate = copy.deepcopy(payload)
        if self._is_canonical_payload(candidate):
            standardized = candidate
            source_format = "canonical"
        else:
            try:
                standardized = standardize_raw_payload(candidate)
            except (TypeError, ValueError) as exc:
                raise InvalidInputError.from_messages([str(exc)]) from exc
            source_format = "raw_standardized"

        valid, errors = validate_payload(
            standardized,
            require_positive_microbes=self.backend != "ac_icam_v8",
            required_clinical_fields=(
                V8_WEB_REQUIRED_CLINICAL_FIELDS if self.backend == "ac_icam_v8" else ()
            ),
        )
        if not valid:
            raise InvalidInputError.from_messages(errors)
        return StandardizationResult(source_format, standardized)

    def analyze(self, payload: object) -> dict[str, object]:
        standardized = self.standardize(payload)
        self._ensure_ready()
        try:
            # Serial execution avoids concurrent loading of the heavyweight fallback
            # and gives the future desktop host a predictable memory ceiling.
            with self._analysis_lock:
                report = self._pipeline_runner(standardized.payload)
        except EngineError:
            raise
        except ModuleNotFoundError as exc:
            raise CapabilityUnavailableError(
                details=[{"capability": "general_risk"}]
            ) from exc
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError() from exc
        except ValueError as exc:
            raise InvalidInputError.from_messages([str(exc)]) from exc
        except Exception as exc:
            raise ModelInferenceError() from exc

        if not isinstance(report, dict):
            raise ModelInferenceError("模型返回了无法识别的结果结构。")
        self._record_loaded_capabilities(report)
        return self._analysis_payload(standardized, report)

    def predict(self, payload: object) -> dict[str, object]:
        analysis = self.analyze(payload)
        report = analysis["report"]
        assert isinstance(report, dict)
        return {
            "source_format": analysis["source_format"],
            "standardized_payload": analysis["standardized_payload"],
            "risk_result": analysis["risk_result"],
            "general_risk_result": analysis["general_risk_result"],
            "graph_features": report.get("gnn_features", {}),
            "recommendations": analysis["recommendations"],
            "pharmacy_assessment": analysis["pharmacy_assessment"],
            "model_release": self.model_release,
        }

    def oral_adenoma_schema(self) -> dict[str, object]:
        bridge = self._get_oral_adenoma_bridge()
        with self._state_lock:
            self._oral_adenoma_loaded = True
        return {
            "model_release": bridge.release_name,
            "research_only": True,
            "input_unit": "percent",
            "required_sum_range_percent": [bridge.sum_min, bridge.sum_max],
            "feature_count": len(bridge.feature_ids),
            "feature_ids": list(bridge.feature_ids),
            "taxonomies": list(bridge.taxonomies),
            "accepted_sample_types": sorted(bridge.artifact["allowed_sample_types"]),
            "claim_boundary": bridge.artifact["claim_boundary"],
        }

    def analyze_oral_adenoma(self, payload: Mapping[str, Any]) -> dict[str, object]:
        bridge = self._get_oral_adenoma_bridge()
        with self._state_lock:
            self._oral_adenoma_loaded = True
        try:
            prediction = bridge.score(
                payload.get("oral_abundances"),
                sample_type=payload.get("sample_type"),
            )
        except ValueError as exc:
            raise InvalidInputError.from_messages([str(exc)]) from exc
        except Exception as exc:
            raise ModelInferenceError("内部口腔腺瘤模型运行失败。") from exc
        return {"oral_adenoma_result": prediction.as_dict()}

    def health(self) -> dict[str, object]:
        capabilities = self._capability_statuses()
        return {
            "engine_ready": self._model_loaded,
            "model_loaded": self._model_loaded,
            "active_backend": self.backend,
            "model_versions": {
                "primary": self.model_release,
                "pfs": AC_ICAM_V8_RELEASE_NAME,
                "general_risk": TEMPORAL_TOPOLOGY_RELEASE_NAME,
                "oral_adenoma": ORAL_ADENOMA_RELEASE_NAME,
                "pharmacy": self._pharmacy_version(),
            },
            "capabilities": {
                name: status.as_dict() for name, status in capabilities.items()
            },
            "initialization_error": self._initialization_error,
        }

    @staticmethod
    def _is_canonical_payload(payload: Mapping[str, Any]) -> bool:
        return all(
            isinstance(payload.get(key), dict) for key in REQUIRED_TOP_LEVEL_KEYS
        )

    def _ensure_ready(self) -> None:
        if not self._initialization_attempted:
            self.initialize()
        if not self._model_loaded:
            raise ModelNotLoadedError()

    @staticmethod
    def _analysis_payload(
        standardized: StandardizationResult,
        report: dict[str, object],
    ) -> dict[str, object]:
        return {
            "source_format": standardized.source_format,
            "standardized_payload": standardized.payload,
            "report": report,
            "risk_result": report.get("risk_result", {}),
            "general_risk_result": report.get("general_risk_result", {}),
            "recommendations": report.get("recommendations", []),
            "pharmacy_assessment": report.get("pharmacy_assessment", {}),
            "top_microbes": report.get("top_microbes", []),
        }

    def _record_loaded_capabilities(self, report: Mapping[str, object]) -> None:
        general_risk = report.get("general_risk_result")
        if not isinstance(general_risk, Mapping):
            return
        if general_risk.get("prediction_available") is True:
            with self._state_lock:
                self._general_risk_loaded = True

    @staticmethod
    def _get_oral_adenoma_bridge() -> Any:
        if not ENABLE_INTERNAL_ORAL_ADENOMA:
            raise FeatureDisabledError("内部口腔腺瘤研究功能当前未启用。")
        try:
            from src.oral_adenoma_bridge import get_oral_adenoma_bridge

            return get_oral_adenoma_bridge()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError() from exc
        except ValueError as exc:
            raise ArtifactIntegrityError() from exc

    def _capability_statuses(self) -> dict[str, CapabilityStatus]:
        general_requirements = ("torch", "torch_geometric", "xgboost")
        missing_modules = tuple(
            name for name in general_requirements if not _module_available(name)
        )
        required_general_paths = {
            "research_model_config": Path(RESEARCH_MODEL_CONFIG_PATH),
            "temporal_topology_artifacts": Path(TEMPORAL_TOPOLOGY_ROOT),
            "full_risk_artifacts": Path(TEMPORAL_TOPOLOGY_FULL_RISK_ROOT),
        }
        missing_paths = tuple(
            name for name, path in required_general_paths.items() if not path.exists()
        )
        general_available = self._general_risk_loaded or (
            not missing_modules and not missing_paths
        )
        general_missing = (*missing_modules, *missing_paths)

        pharmacy_version = self._pharmacy_version()
        pharmacy_available = pharmacy_version != "unavailable"
        oral_available = (
            ENABLE_INTERNAL_ORAL_ADENOMA and Path(ORAL_ADENOMA_ARTIFACT_PATH).exists()
        )

        return {
            "primary_model": CapabilityStatus(
                enabled=True,
                available=self._model_loaded,
                loaded=self._model_loaded,
                reason=None if self._model_loaded else "initialization_failed",
            ),
            "pfs": CapabilityStatus(
                enabled=self.backend == "ac_icam_v8",
                available=self.backend == "ac_icam_v8"
                and Path(AC_ICAM_V8_ARTIFACT_PATH).exists(),
                loaded=self.backend == "ac_icam_v8" and self._model_loaded,
                reason=None if self.backend == "ac_icam_v8" else "inactive_backend",
            ),
            "general_risk": CapabilityStatus(
                enabled=True,
                available=general_available,
                loaded=self._general_risk_loaded,
                reason=None if general_available else "dependency_or_artifact_missing",
                missing_requirements=general_missing,
            ),
            "pharmacy": CapabilityStatus(
                enabled=True,
                available=pharmacy_available,
                loaded=pharmacy_available,
                reason=None if pharmacy_available else "knowledge_base_unavailable",
            ),
            "oral_adenoma": CapabilityStatus(
                enabled=ENABLE_INTERNAL_ORAL_ADENOMA,
                available=oral_available or self._oral_adenoma_loaded,
                loaded=self._oral_adenoma_loaded,
                reason=(
                    None
                    if oral_available
                    else (
                        "feature_disabled"
                        if not ENABLE_INTERNAL_ORAL_ADENOMA
                        else "artifact_missing"
                    )
                ),
            ),
        }

    @staticmethod
    def _pharmacy_version() -> str:
        try:
            return str(load_pharmacy_knowledge_base()["engine_version"])
        except Exception:
            return "unavailable"
