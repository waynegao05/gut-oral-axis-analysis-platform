from __future__ import annotations

from typing import Any, Mapping, Sequence


class EngineError(Exception):
    error_code = "ENGINE_ERROR"
    http_status = 500
    default_message = "分析引擎暂时无法完成请求。"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.public_message = message or self.default_message
        self.details = [dict(item) for item in (details or ())]
        super().__init__(self.public_message)


class InvalidInputError(EngineError):
    error_code = "INVALID_INPUT"
    http_status = 400
    default_message = "请检查输入后重新提交。"

    @classmethod
    def from_messages(cls, messages: Sequence[str]) -> "InvalidInputError":
        return cls(
            details=[{"message": str(message)} for message in messages if message]
        )


class AuthenticationError(EngineError):
    error_code = "AUTHENTICATION_FAILED"
    http_status = 401
    default_message = "本地分析引擎拒绝了未经授权的请求。"


class FeatureDisabledError(EngineError):
    error_code = "FEATURE_DISABLED"
    http_status = 404
    default_message = "该分析功能当前未启用。"


class ModelNotLoadedError(EngineError):
    error_code = "MODEL_NOT_LOADED"
    http_status = 503
    default_message = "模型尚未完成加载，请稍后重试。"


class CapabilityUnavailableError(EngineError):
    error_code = "CAPABILITY_UNAVAILABLE"
    http_status = 503
    default_message = "当前请求所需的模型能力尚未安装完整。"


class ArtifactIntegrityError(EngineError):
    error_code = "ARTIFACT_INTEGRITY_ERROR"
    http_status = 503
    default_message = "模型工件缺失或完整性校验失败。"


class ModelInferenceError(EngineError):
    error_code = "MODEL_INFERENCE_FAILED"
    http_status = 500
    default_message = "模型运行失败，请查看本地技术日志。"


class RequestTooLargeError(EngineError):
    error_code = "REQUEST_TOO_LARGE"
    http_status = 413
    default_message = "请求数据超过本地分析引擎允许的大小。"
