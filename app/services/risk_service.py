from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.models.model_loader import ModelArtifacts
from app.schemas.fraud import (
    BatchPredictionResponse,
    Decision,
    PredictionResponse,
    RiskLevel,
    TopRiskFactor,
    TransactionFeatures,
)
from src.features.build_features import (
    API_TO_MODEL_FEATURE_MAP,
    api_payload_to_model_frame,
    api_payloads_to_model_frame,
)

MODEL_TO_API_FEATURE_MAP = {
    model_name: api_name for api_name, model_name in API_TO_MODEL_FEATURE_MAP.items()
}


class RiskService:
    def __init__(self, artifacts: ModelArtifacts, low_cutoff: float = 0.30) -> None:
        self.artifacts = artifacts
        self.low_cutoff = low_cutoff
        self._shap_explainer: Any | None = None

    def predict(self, transaction: TransactionFeatures) -> PredictionResponse:
        payload = transaction.model_dump()
        features = api_payload_to_model_frame(payload)
        probability = float(self.artifacts.predict_proba(features)[0])
        risk_level, decision = self.risk_level_and_decision(probability)

        top_factors = (
            self._top_risk_factors(transaction)
            if transaction.include_top_risk_factors
            else None
        )

        return PredictionResponse(
            fraud_probability=round(probability, 6),
            risk_level=risk_level,
            decision=decision,
            threshold=round(self.artifacts.threshold, 6),
            model_version=self.artifacts.model_version,
            top_risk_factors=top_factors,
        )

    def batch_predict(
        self,
        transactions: list[TransactionFeatures],
    ) -> BatchPredictionResponse:
        payloads = [transaction.model_dump() for transaction in transactions]
        features = api_payloads_to_model_frame(payloads)
        probabilities = self.artifacts.predict_proba(features)

        predictions = []
        for transaction, probability in zip(transactions, probabilities, strict=True):
            risk_level, decision = self.risk_level_and_decision(float(probability))
            top_factors = (
                self._top_risk_factors(transaction)
                if transaction.include_top_risk_factors
                else None
            )
            predictions.append(
                PredictionResponse(
                    fraud_probability=round(float(probability), 6),
                    risk_level=risk_level,
                    decision=decision,
                    threshold=round(self.artifacts.threshold, 6),
                    model_version=self.artifacts.model_version,
                    top_risk_factors=top_factors,
                )
            )

        return BatchPredictionResponse(
            predictions=predictions,
            model_version=self.artifacts.model_version,
            threshold=round(self.artifacts.threshold, 6),
        )

    def risk_level_and_decision(
        self,
        fraud_probability: float,
    ) -> tuple[RiskLevel, Decision]:
        if fraud_probability < self.low_cutoff:
            return "low", "approve"
        if fraud_probability < self.artifacts.threshold:
            return "medium", "monitor"
        if fraud_probability >= 0.90:
            return "high", "block"
        return "high", "review"

    def _top_risk_factors(
        self,
        transaction: TransactionFeatures,
        limit: int = 3,
    ) -> list[TopRiskFactor]:
        payload = transaction.model_dump()
        features = api_payload_to_model_frame(payload)
        shap_values = self._xgboost_shap_values(features)

        if shap_values is None:
            candidates = [
                (
                    feature,
                    float(features.iloc[0][feature]),
                    0.0,
                    abs(float(features.iloc[0][feature])),
                )
                for feature in self.artifacts.feature_names
            ]
        else:
            candidates = [
                (
                    feature,
                    float(features.iloc[0][feature]),
                    float(shap_value),
                    abs(float(shap_value)),
                )
                for feature, shap_value in zip(
                    self.artifacts.feature_names,
                    shap_values,
                    strict=True,
                )
            ]

        top = sorted(candidates, key=lambda item: item[3], reverse=True)[:limit]
        return [
            TopRiskFactor(
                feature=MODEL_TO_API_FEATURE_MAP.get(field, field),
                value=value,
                direction=self._direction_from_shap_value(shap_value),
            )
            for field, value, shap_value, _ in top
        ]

    def _xgboost_shap_values(self, features: pd.DataFrame) -> np.ndarray | None:
        estimator, explanation_features = self._extract_xgboost_estimator(features)
        if estimator is None:
            return None

        if self._shap_explainer is None:
            import shap

            self._shap_explainer = shap.TreeExplainer(estimator)

        raw_values = self._shap_explainer.shap_values(explanation_features)
        values = self._normalise_shap_values(raw_values)
        return values[0]

    def _extract_xgboost_estimator(
        self,
        features: pd.DataFrame,
    ) -> tuple[Any | None, pd.DataFrame | np.ndarray]:
        model = self.artifacts.model
        if isinstance(model, Pipeline):
            transformed: pd.DataFrame | np.ndarray = features
            for _, step in model.steps[:-1]:
                transformed = step.transform(transformed)
            estimator = model.steps[-1][1]
        else:
            transformed = features
            estimator = model

        if estimator.__class__.__name__ != "XGBClassifier":
            return None, transformed
        return estimator, transformed

    @staticmethod
    def _normalise_shap_values(raw_values: Any) -> np.ndarray:
        values = raw_values.values if hasattr(raw_values, "values") else raw_values
        if isinstance(values, list):
            values = values[-1]

        array = np.asarray(values)
        if array.ndim == 3:
            array = array[:, :, -1]
        return array

    @staticmethod
    def _direction_from_shap_value(
        shap_value: float,
    ) -> Literal["elevates_risk", "reduces_risk", "unknown"]:
        if shap_value > 0:
            return "elevates_risk"
        if shap_value < 0:
            return "reduces_risk"
        return "unknown"
