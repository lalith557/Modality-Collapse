"""Response models."""
from __future__ import annotations

from pydantic import BaseModel

from inference.prediction_schema import PredictionResponse  # noqa: F401  (re-export)


class ErrorBody(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    device: str
    models: dict
    uptime_s: float
    research_prototype: bool = True
