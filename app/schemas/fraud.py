from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    """Lowercase public API schema for one transaction."""

    model_config = ConfigDict(extra="forbid")

    time: float = Field(..., ge=0, description="Seconds elapsed in dataset")
    amount: float = Field(..., ge=0, description="Transaction amount")
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    include_top_risk_factors: bool = Field(
        default=False,
        description="Return lightweight top factor signals when available",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[TransactionFeatures] = Field(..., min_length=1, max_length=500)


RiskLevel = Literal["low", "medium", "high"]
Decision = Literal["approve", "monitor", "review", "block"]


class TopRiskFactor(BaseModel):
    feature: str
    value: float
    direction: Literal["elevates_risk", "reduces_risk", "unknown"]


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    decision: Decision
    threshold: float = Field(..., ge=0, le=1)
    model_version: str
    top_risk_factors: list[TopRiskFactor] | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str
    threshold: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    app_name: str
    app_version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    threshold: float | None = None
    feature_count: int | None = None
    metrics: dict | None = None
    message: str | None = None
