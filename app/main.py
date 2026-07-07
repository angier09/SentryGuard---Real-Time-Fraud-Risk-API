from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.models.model_loader import (
    ModelArtifactNotFoundError,
    ThresholdArtifactNotFoundError,
    load_model_artifacts,
)
from app.services.risk_service import RiskService

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        try:
            artifacts = load_model_artifacts(app_settings)
            fastapi_app.state.risk_service = RiskService(
                artifacts,
                low_cutoff=app_settings.risk_low_cutoff,
            )
            logger.info("Loaded model artifact: %s", artifacts.model_version)
        except (ModelArtifactNotFoundError, ThresholdArtifactNotFoundError) as exc:
            fastapi_app.state.risk_service = None
            logger.warning("Starting without model artifacts: %s", exc)

        yield

    fastapi_app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="Real-time fraud risk scoring API.",
        lifespan=lifespan,
    )
    fastapi_app.include_router(router)
    Instrumentator().instrument(fastapi_app).expose(fastapi_app)
    return fastapi_app


app = create_app()
