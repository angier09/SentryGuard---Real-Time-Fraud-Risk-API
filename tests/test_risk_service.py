from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.models.model_loader import ModelArtifacts
from app.schemas.fraud import TransactionFeatures
from app.services.risk_service import RiskService
from src.features.build_features import get_model_feature_columns


def test_risk_mapping_low_medium_high(
    risk_service: RiskService,
) -> None:
    assert risk_service.risk_level_and_decision(0.10) == ("low", "approve")
    assert risk_service.risk_level_and_decision(0.50) == ("medium", "monitor")
    assert risk_service.risk_level_and_decision(0.80) == ("high", "review")
    assert risk_service.risk_level_and_decision(0.95) == ("high", "block")


def test_predict_returns_valid_response(
    risk_service: RiskService,
    sample_transaction: TransactionFeatures,
) -> None:
    response = risk_service.predict(sample_transaction)

    assert 0 <= response.fraud_probability <= 1
    assert response.threshold == 0.72
    assert response.model_version == "xgb_v1.0.0"
    assert response.risk_level in {"low", "medium", "high"}


def test_top_risk_factors_are_lightweight_signals(
    risk_service: RiskService,
    sample_transaction_payload: dict,
) -> None:
    sample_transaction_payload["include_top_risk_factors"] = True
    sample_transaction_payload["time"] = 0.1
    sample_transaction_payload["amount"] = 1.0
    sample_transaction_payload["v1"] = -3.0
    sample_transaction_payload["v2"] = 2.5
    transaction = TransactionFeatures(**sample_transaction_payload)

    response = risk_service.predict(transaction)

    assert response.top_risk_factors is not None
    assert response.top_risk_factors[0].feature == "v1"
    assert response.top_risk_factors[0].direction == "unknown"


def test_top_risk_factors_use_signed_shap_values_for_xgboost() -> None:
    feature_names = get_model_feature_columns()
    rows = 20
    labels = np.array([0] * 10 + [1] * 10)
    features = pd.DataFrame(0.0, index=range(rows), columns=feature_names)
    features["Time"] = np.arange(rows, dtype=float)
    features["Amount"] = np.where(labels == 1, 250.0, 20.0)
    features["V1"] = np.where(labels == 1, 4.0, -2.0)
    features["V2"] = np.where(labels == 1, -3.0, 1.0)

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=8,
        max_depth=2,
        random_state=42,
    )
    classifier.fit(features, labels)
    service = RiskService(
        ModelArtifacts(
            model=Pipeline(steps=[("classifier", classifier)]),
            feature_names=feature_names,
            threshold=0.5,
            metadata={"model_name": "xgboost", "model_version": "xgb_test"},
            metrics=None,
        )
    )
    payload: dict[str, Any] = {
        "time": 19.0,
        "amount": 250.0,
        "include_top_risk_factors": True,
    }
    payload.update({f"v{index}": 0.0 for index in range(1, 29)})
    payload["v1"] = 4.0
    payload["v2"] = -3.0

    response = service.predict(TransactionFeatures(**payload))

    assert response.top_risk_factors is not None
    assert len(response.top_risk_factors) == 3
    assert {factor.direction for factor in response.top_risk_factors} <= {
        "elevates_risk",
        "reduces_risk",
        "unknown",
    }
    assert any(
        factor.direction in {"elevates_risk", "reduces_risk"}
        for factor in response.top_risk_factors
    )
