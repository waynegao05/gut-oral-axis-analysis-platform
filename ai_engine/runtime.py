from __future__ import annotations

from datetime import UTC, datetime
import ipaddress
import time

from ai_engine import AI_ENGINE_VERSION
from ai_engine.service import AIService


def validate_bind_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return normalized
    candidate = normalized.strip("[]")
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise ValueError("AI Engine host must be a loopback address.") from exc
    if not address.is_loopback:
        raise ValueError("AI Engine must only listen on a loopback address.")
    return normalized


class EngineRuntime:
    def __init__(self, service: AIService | None = None) -> None:
        self.service = service or AIService()
        self.started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()

    def initialize(self) -> bool:
        return self.service.initialize()

    def health(self) -> dict[str, object]:
        return {
            "ai_engine_version": AI_ENGINE_VERSION,
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": round(time.monotonic() - self._started_monotonic, 3),
            **self.service.health(),
        }
