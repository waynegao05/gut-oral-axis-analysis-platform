from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str | None = None
    capability: str | None = None


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class CapabilityResponse(BaseModel):
    enabled: bool
    available: bool
    loaded: bool
    reason: str | None = None
    missing_requirements: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    request_id: str
    engine_ready: bool
    model_loaded: bool
    ai_engine_version: str
    started_at: str
    uptime_seconds: float
    active_backend: str
    model_versions: dict[str, str]
    capabilities: dict[str, CapabilityResponse]
    initialization_error: str | None = None


class StandardizeResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    source_format: str
    standardized_payload: dict[str, Any]


class PredictResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    source_format: str
    standardized_payload: dict[str, Any]
    risk_result: dict[str, Any]
    general_risk_result: dict[str, Any]
    graph_features: dict[str, Any]
    recommendations: list[dict[str, Any]]
    pharmacy_assessment: dict[str, Any]
    model_release: str


class AnalyzeResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    source_format: str
    standardized_payload: dict[str, Any]
    report: dict[str, Any]
    risk_result: dict[str, Any]
    general_risk_result: dict[str, Any]
    recommendations: list[dict[str, Any]]
    pharmacy_assessment: dict[str, Any]
    top_microbes: list[tuple[str, float]]


class OralAdenomaSchemaResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    model_release: str
    research_only: bool
    input_unit: str
    required_sum_range_percent: list[float]
    feature_count: int
    feature_ids: list[str]
    taxonomies: list[str]
    accepted_sample_types: list[str]
    claim_boundary: str


class OralAdenomaAnalysisResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    oral_adenoma_result: dict[str, Any]
