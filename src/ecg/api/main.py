"""Research-only FastAPI service for the frozen PTB-XL CNN Baseline v1.

The API never fits normalization, tunes thresholds, or loads test predictions.
The frozen predictor verifies model and normalization artifact integrity at startup.

For local use, launch with:
    ECG_ARTIFACTS_DIR=/path/to/artifacts uvicorn src.ecg.api.main:app --host 127.0.0.1

Do NOT expose this research API to the public internet or use it for clinical care.
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.ecg.inference.predictor import (
    CLASS_ORDER,
    LEAD_ORDER,
    MODEL_NAME,
    RESEARCH_NOTICE,
    FrozenCNNPredictor,
)

_MAX_FLOAT32 = 3.4028234663852886e38


class ECGRequest(BaseModel):
    """Exactly one 10-second, 12-lead, 100 Hz ECG in millivolts."""

    waveform: list[list[float]] = Field(
        description="12 x 1000 array, one row per lead in lead_order (mV)."
    )
    lead_order: list[str] = Field(
        description="Exact PTB-XL lead sequence; no automatic reordering."
    )
    sampling_rate_hz: Literal[100]
    units: Literal["mV"]

    @field_validator("lead_order")
    @classmethod
    def validate_leads(cls, value: list[str]) -> list[str]:
        if tuple(value) != LEAD_ORDER:
            raise ValueError("lead_order must match the exact PTB-XL 12-lead sequence")
        return value

    @field_validator("waveform", mode="before")
    @classmethod
    def validate_waveform(cls, value: object) -> object:
        # Pre-conversion validation prevents strings/bools becoming floats.
        if not isinstance(value, list) or len(value) != 12:
            raise ValueError("waveform must contain exactly 12 leads")
        for i, lead in enumerate(value):
            if not isinstance(lead, list) or len(lead) != 1000:
                raise ValueError(f"waveform lead {i} must contain exactly 1000 samples")
            for x in lead:
                if type(x) not in (int, float):
                    raise ValueError("waveform samples must be numeric JSON values")
                if not math.isfinite(x) or abs(x) > _MAX_FLOAT32:
                    raise ValueError("waveform samples must be finite float32-compatible values")
        return value


class ClassPrediction(BaseModel):
    class_name: str = Field(alias="class")
    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted: bool


class ECGResponse(BaseModel):
    model: str
    checkpoint_epoch: int
    checkpoint_sha256: str
    normalization_sha256: str
    predictions: list[ClassPrediction]
    predicted_classes: list[str]
    status: Literal["classifications_generated", "no_class_above_threshold"]
    research_notice: str


class ServiceStatus(BaseModel):
    status: Literal["ready"]
    model: str
    research_notice: str


def create_app(
    *,
    predictor: FrozenCNNPredictor | None = None,
    artifacts_dir: str | Path | None = None,
    device: str | None = None,
) -> FastAPI:
    """Construct app. A supplied predictor is only for isolated tests.

    Production use loads verified frozen artifacts exactly once during lifespan.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        selected = predictor
        if selected is None:
            configured_dir = artifacts_dir or os.environ.get("ECG_ARTIFACTS_DIR")
            if not configured_dir:
                raise RuntimeError(
                    "ECG_ARTIFACTS_DIR must point to the frozen experiment's artifacts directory"
                )
            selected_device = device or os.environ.get("ECG_DEVICE", "cpu")
            if selected_device not in ("cpu", "cuda"):
                raise RuntimeError("ECG_DEVICE must be cpu or cuda")
            selected = FrozenCNNPredictor.from_artifacts(
                configured_dir, device=selected_device
            )
        app.state.predictor = selected
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(
        title="PTB-XL Frozen CNN Research API",
        version="0.1.0",
        description=(
            "Research-only 12-lead ECG multi-label inference. "
            "Not a clinical diagnostic service."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=ServiceStatus, tags=["Service"])
    def health() -> ServiceStatus:
        return ServiceStatus(
            status="ready", model=MODEL_NAME, research_notice=RESEARCH_NOTICE
        )

    @app.post("/v1/predict", response_model=ECGResponse, tags=["Inference"])
    def predict_one(request: ECGRequest) -> dict:
        try:
            return app.state.predictor.predict(
                request.waveform,
                lead_order=request.lead_order,
                sampling_rate_hz=request.sampling_rate_hz,
                units=request.units,
            )
        except ValueError as exc:
            # Reject invalid ECGs rather than returning unsupported predictions.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
