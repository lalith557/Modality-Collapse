"""Response schema shared by the inference engine, the FastAPI backend and the frontend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["compatible", "potential_distribution_shift", "low_quality", "unsupported"]
Severity = Literal["unsupported", "low_quality", "info"]


class ClassProb(BaseModel):
    index: int
    code: str
    name: str | None
    probability: float


class Prediction(BaseModel):
    source: Literal["joint_head", "frozen_encoder_probe"]
    modalities_observed: list[str]
    description: str
    predicted_index: int
    predicted_code: str
    predicted_name: str | None
    probabilities: list[ClassProb]
    max_probability: float
    entropy_nats: float
    normalized_entropy: float
    reference_accuracy: dict = Field(description="held-out accuracy of THIS predictor (validation; test evaluation-only)")


class Issue(BaseModel):
    code: str
    severity: Severity
    modality: str
    message: str
    measured_false_flag_rate_test: float | None = None


class ShiftAssessment(BaseModel):
    detector: str
    score: float
    threshold: float
    exceeds: bool
    ratio_to_threshold: float
    validation_flag_rate: float
    test_flag_rate_evaluation_only: float
    used_for_status: bool


class TraceLayer(BaseModel):
    name: str
    type: str
    output_shape: list[int]
    mean: float
    std: float
    min: float
    max: float


class Step(BaseModel):
    name: str
    duration_ms: float
    detail: str = ""


class PredictionResponse(BaseModel):
    request_id: str
    dataset: Literal["cremad", "utd_mhad"]
    mode: str
    status: Status
    status_reasons: list[str]
    primary: Prediction | None
    unimodal: dict[str, Prediction]
    issues: list[Issue]
    shift: list[ShiftAssessment]
    input_report: dict
    preprocessed: dict
    trace: dict
    steps: list[Step]
    total_ms: float
    model: dict
    limitations: list[str]
    disclaimer: str = ("Research prototype. Predictions come from a small model trained from scratch on one "
                       "dataset; they are not a reliable emotion or action recognizer and must not be used "
                       "for decisions about people.")
