from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FEATURE_COLUMNS: list[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

API_FEATURE_FIELDS: list[str] = [
    "time",
    "amount",
    *[f"v{i}" for i in range(1, 29)],
]

API_TO_MODEL_FEATURE_MAP: dict[str, str] = {
    "time": "Time",
    "amount": "Amount",
    **{f"v{i}": f"V{i}" for i in range(1, 29)},
}


def get_model_feature_columns() -> list[str]:
    """Return the canonical training/serving feature order."""

    return MODEL_FEATURE_COLUMNS.copy()


def get_api_feature_fields() -> list[str]:
    """Return the public lowercase API field names."""

    return API_FEATURE_FIELDS.copy()


def build_feature_scaling_pipeline() -> Pipeline:
    """Build preprocessing used by linear models without fitting on holdout data."""

    return Pipeline(steps=[("scaler", StandardScaler())])


def api_payload_to_model_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert one lowercase API payload into Kaggle feature columns.

    The Kaggle dataset uses capitalized columns such as ``Time`` and ``V1``.
    The API intentionally exposes lowercase JSON fields. This function is the
    single boundary where that naming difference is reconciled.
    """

    row = {
        model_name: payload[api_name]
        for api_name, model_name in API_TO_MODEL_FEATURE_MAP.items()
    }
    return pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)


def api_payloads_to_model_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            model_name: payload[api_name]
            for api_name, model_name in API_TO_MODEL_FEATURE_MAP.items()
        }
        for payload in payloads
    ]
    return pd.DataFrame(rows, columns=MODEL_FEATURE_COLUMNS)
