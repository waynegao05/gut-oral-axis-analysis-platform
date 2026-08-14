from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from typing import Any
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ai_engine.errors import (
    AuthenticationError,
    EngineError,
    RequestTooLargeError,
)
from ai_engine.runtime import EngineRuntime
from ai_engine.schemas.requests import JsonObjectRequest, OralAdenomaRequest
from ai_engine.schemas.responses import (
    AnalyzeResponse,
    HealthResponse,
    OralAdenomaAnalysisResponse,
    OralAdenomaSchemaResponse,
    PredictResponse,
    StandardizeResponse,
)
from ai_engine.service import AIService
from src.logging_utils import get_logger


DEFAULT_MAX_REQUEST_BYTES = 2 * 1024 * 1024


def create_app(
    *,
    service: AIService | None = None,
    engine_token: str | None = None,
    require_token: bool = True,
    initialize_on_startup: bool = True,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> FastAPI:
    token = engine_token if engine_token is not None else os.getenv("GOA_ENGINE_TOKEN")
    if require_token and (not token or len(token) < 32):
        raise RuntimeError(
            "GOA_ENGINE_TOKEN must contain at least 32 characters for the local API."
        )
    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive.")

    runtime = EngineRuntime(service)
    logger = get_logger("gut_oral_axis.ai_engine.api")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if initialize_on_startup:
            runtime.initialize()
        yield

    app = FastAPI(
        title="Gut-Oral Axis Local AI Engine",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    def request_id(request: Request) -> str:
        return getattr(request.state, "request_id", uuid4().hex)

    def error_response(
        request: Request,
        *,
        status_code: int,
        error_code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "error",
                "error_code": error_code,
                "message": message,
                "request_id": request_id(request),
                "details": details or [],
            },
        )

    @app.middleware("http")
    async def request_boundary(request: Request, call_next):
        request.state.request_id = uuid4().hex
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = 0
            if content_length > max_request_bytes:
                exc = RequestTooLargeError()
                return error_response(
                    request,
                    status_code=exc.http_status,
                    error_code=exc.error_code,
                    message=exc.public_message,
                )
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > max_request_bytes:
                exc = RequestTooLargeError()
                return error_response(
                    request,
                    status_code=exc.http_status,
                    error_code=exc.error_code,
                    message=exc.public_message,
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(EngineError)
    async def engine_error_handler(request: Request, exc: EngineError):
        return error_response(
            request,
            status_code=exc.http_status,
            error_code=exc.error_code,
            message=exc.public_message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        details = []
        for item in exc.errors():
            location = [str(part) for part in item.get("loc", ()) if part != "body"]
            details.append(
                {
                    "field": ".".join(location) or None,
                    "message": str(item.get("msg", "输入格式不正确。")),
                }
            )
        return error_response(
            request,
            status_code=400,
            error_code="INVALID_INPUT",
            message="请检查输入后重新提交。",
            details=details,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        return error_response(
            request,
            status_code=exc.status_code,
            error_code="HTTP_ERROR",
            message="请求的本地分析接口不存在或不可用。",
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception(
            "Unhandled AI Engine API error. request_id=%s",
            request_id(request),
        )
        return error_response(
            request,
            status_code=500,
            error_code="ENGINE_ERROR",
            message="本地分析引擎发生未处理错误，请查看技术日志。",
        )

    async def authorize_engine(
        submitted_token: str | None = Header(
            default=None,
            alias="X-GOA-Engine-Token",
        ),
    ) -> None:
        if not require_token:
            return
        assert token is not None
        if submitted_token is None or not hmac.compare_digest(
            submitted_token,
            token,
        ):
            raise AuthenticationError()

    protected = [Depends(authorize_engine)]

    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        dependencies=protected,
    )
    def health(request: Request) -> dict[str, object]:
        return {
            "status": "ok",
            "request_id": request_id(request),
            **runtime.health(),
        }

    @app.post(
        "/api/v1/standardize",
        response_model=StandardizeResponse,
        dependencies=protected,
    )
    def standardize(
        request: Request,
        payload: JsonObjectRequest = Body(...),
    ) -> dict[str, object]:
        result = runtime.service.standardize(payload.root)
        return {
            "status": "success",
            "request_id": request_id(request),
            "source_format": result.source_format,
            "standardized_payload": result.payload,
        }

    @app.post(
        "/api/v1/predict",
        response_model=PredictResponse,
        dependencies=protected,
    )
    def predict(
        request: Request,
        payload: JsonObjectRequest = Body(...),
    ) -> dict[str, object]:
        return {
            "status": "success",
            "request_id": request_id(request),
            **runtime.service.predict(payload.root),
        }

    @app.post(
        "/api/v1/analyze",
        response_model=AnalyzeResponse,
        dependencies=protected,
    )
    def analyze(
        request: Request,
        payload: JsonObjectRequest = Body(...),
    ) -> dict[str, object]:
        return {
            "status": "success",
            "request_id": request_id(request),
            **runtime.service.analyze(payload.root),
        }

    @app.get(
        "/api/v1/oral-adenoma/schema",
        response_model=OralAdenomaSchemaResponse,
        dependencies=protected,
    )
    def oral_adenoma_schema(request: Request) -> dict[str, object]:
        return {
            "status": "success",
            "request_id": request_id(request),
            **runtime.service.oral_adenoma_schema(),
        }

    @app.post(
        "/api/v1/oral-adenoma/analyze",
        response_model=OralAdenomaAnalysisResponse,
        dependencies=protected,
    )
    def analyze_oral_adenoma(
        request: Request,
        payload: OralAdenomaRequest,
    ) -> dict[str, object]:
        return {
            "status": "success",
            "request_id": request_id(request),
            **runtime.service.analyze_oral_adenoma(payload.model_dump()),
        }

    return app
