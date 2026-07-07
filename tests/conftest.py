from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.models.model_loader import ModelArtifacts
from app.schemas.fraud import TransactionFeatures
from app.services.risk_service import RiskService
from src.features.build_features import get_model_feature_columns


class FakeFraudModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        amount_signal = np.clip(features["Amount"].to_numpy() / 1_000, 0, 1)
        probabilities = np.maximum(amount_signal, 0.42)
        return np.column_stack([1 - probabilities, probabilities])


@pytest.fixture
def fake_artifacts() -> ModelArtifacts:
    return ModelArtifacts(
        model=FakeFraudModel(),
        feature_names=get_model_feature_columns(),
        threshold=0.72,
        metadata={
            "model_name": "xgboost",
            "model_version": "xgb_v1.0.0",
        },
        metrics={"test": {"pr_auc_average_precision": 0.9}},
    )


@pytest.fixture
def risk_service(fake_artifacts: ModelArtifacts) -> RiskService:
    return RiskService(fake_artifacts)


@pytest.fixture
def sample_transaction_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {"time": 10.0, "amount": 125.50}
    payload.update({f"v{i}": 0.0 for i in range(1, 29)})
    return payload


@pytest.fixture
def sample_transaction(
    sample_transaction_payload: dict[str, Any],
) -> TransactionFeatures:
    return TransactionFeatures(**sample_transaction_payload)


@pytest.fixture
def synthetic_creditcard_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    non_fraud_count = 64
    fraud_count = 16
    labels = np.array([0] * non_fraud_count + [1] * fraud_count)

    data: dict[str, Any] = {
        "Time": np.arange(len(labels), dtype=float),
        "Amount": rng.normal(80, 15, size=len(labels)),
    }
    for index in range(1, 29):
        data[f"V{index}"] = rng.normal(0, 1, size=len(labels))

    frame = pd.DataFrame(data)
    frame.loc[labels == 1, "Amount"] += 120
    frame.loc[labels == 1, "V1"] += 3.0
    frame.loc[labels == 1, "V2"] -= 2.0
    frame["Class"] = labels
    return frame
