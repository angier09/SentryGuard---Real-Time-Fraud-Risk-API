from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    The defaults are relative paths so the project remains portable across
    local development, CI, Docker, and hosted deployment environments.
    """

    app_name: str = "SentryGuard"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: str = "INFO"

    model_path: Path = Path("artifacts/model.joblib")
    threshold_path: Path = Path("artifacts/threshold.json")
    metrics_path: Path = Path("artifacts/metrics.json")

    risk_low_cutoff: float = 0.30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SENTRYGUARD_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
