from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.schemas.fraud import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionFeatures,
)
from app.services.risk_service import RiskService

router = APIRouter(tags=["fraud-risk"])


def _get_risk_service(request: Request) -> RiskService:
    service = getattr(request.app.state, "risk_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifacts are not loaded. Train the model first with "
                "uv run python -m src.training.train --input data/raw/creditcard.csv"
            ),
        )
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings: Settings = get_settings()
    model_loaded = getattr(request.app.state, "risk_service", None) is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_loaded=model_loaded,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = getattr(request.app.state, "risk_service", None)
    if service is None:
        return ModelInfoResponse(
            model_loaded=False,
            message="Model artifacts are not loaded yet.",
        )

    artifacts = service.artifacts
    return ModelInfoResponse(
        model_loaded=True,
        model_name=artifacts.model_name,
        model_version=artifacts.model_version,
        threshold=round(artifacts.threshold, 6),
        feature_count=len(artifacts.feature_names),
        metrics=artifacts.metrics,
    )


@router.post("/v1/predict", response_model=PredictionResponse)
def predict(
    transaction: TransactionFeatures,
    request: Request,
) -> PredictionResponse:
    service = _get_risk_service(request)
    return service.predict(transaction)


@router.post("/v1/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    service = _get_risk_service(request)
    return service.batch_predict(payload.transactions)
