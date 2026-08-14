from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel


class JsonObjectRequest(RootModel[dict[str, Any]]):
    """A JSON object that may be canonical input or a supported raw payload."""


class OralAdenomaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_type: str
    oral_abundances: dict[str, float]
