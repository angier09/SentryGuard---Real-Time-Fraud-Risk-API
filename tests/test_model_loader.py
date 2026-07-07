from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.model_loader import (
    ModelArtifactNotFoundError,
    ThresholdArtifactNotFoundError,
    load_model_artifacts,
)


def test_model_loader_missing_artifact_error(tmp_path: Path) -> None:
    settings = Settings(
        model_path=tmp_path / "missing.joblib",
        threshold_path=tmp_path / "threshold.json",
        metrics_path=tmp_path / "metrics.json",
    )

    with pytest.raises(ModelArtifactNotFoundError, match="Model artifact not found"):
        load_model_artifacts(settings)


def test_model_loader_missing_threshold_error(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    settings = Settings(
        model_path=model_path,
        threshold_path=tmp_path / "missing-threshold.json",
        metrics_path=tmp_path / "metrics.json",
    )

    with pytest.raises(
        ThresholdArtifactNotFoundError,
        match="Threshold artifact not found",
    ):
        load_model_artifacts(settings)
