from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import Settings
from src.features.build_features import get_model_feature_columns


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact is unavailable."""


class ThresholdArtifactNotFoundError(FileNotFoundError):
    """Raised when the decision threshold artifact is unavailable."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    feature_names: list[str]
    threshold: float
    metadata: dict[str, Any]
    metrics: dict[str, Any] | None

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(features.columns))
        if missing:
            raise ValueError("Prediction frame missing columns: " + ", ".join(missing))

        ordered = features[self.feature_names]
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(ordered)
            return np.asarray(probabilities)[:, 1]

        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(ordered))
            return 1 / (1 + np.exp(-scores))

        raise TypeError("Loaded model must expose predict_proba or decision_function.")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_artifacts(settings: Settings) -> ModelArtifacts:
    if not settings.model_path.exists():
        raise ModelArtifactNotFoundError(
            f"Model artifact not found at {settings.model_path}. "
            "Run: uv run python -m src.training.train --input data/raw/creditcard.csv"
        )
    if not settings.threshold_path.exists():
        raise ThresholdArtifactNotFoundError(
            f"Threshold artifact not found at {settings.threshold_path}. "
            "Run training to generate artifacts/threshold.json."
        )

    payload = joblib.load(settings.model_path)
    threshold_payload = _load_json(settings.threshold_path)
    metrics = (
        _load_json(settings.metrics_path) if settings.metrics_path.exists() else None
    )

    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_names = list(payload.get("feature_names", get_model_feature_columns()))
        metadata = dict(payload.get("metadata", {}))
    else:
        model = payload
        feature_names = get_model_feature_columns()
        metadata = {"model_name": "unknown", "model_version": "unknown"}

    return ModelArtifacts(
        model=model,
        feature_names=feature_names,
        threshold=float(threshold_payload["threshold"]),
        metadata=metadata,
        metrics=metrics,
    )
