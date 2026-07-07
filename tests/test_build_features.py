from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import (
    api_payload_to_model_frame,
    api_payloads_to_model_frame,
    build_feature_scaling_pipeline,
    get_model_feature_columns,
)


def test_api_payloads_are_mapped_to_model_feature_order() -> None:
    payload = {"time": 1.0, "amount": 99.0}
    payload.update({f"v{index}": float(index) for index in range(1, 29)})

    frame = api_payload_to_model_frame(payload)
    batch_frame = api_payloads_to_model_frame([payload, payload])

    assert frame.shape == (1, 30)
    assert batch_frame.shape == (2, 30)
    assert list(frame.columns) == get_model_feature_columns()
    assert frame.loc[0, "Time"] == 1.0
    assert frame.loc[0, "V28"] == 28.0
    assert frame.loc[0, "Amount"] == 99.0


def test_scaling_pipeline_fits_only_on_training_data() -> None:
    columns = get_model_feature_columns()
    train = pd.DataFrame([[1.0] * 30, [3.0] * 30, [5.0] * 30], columns=columns)
    test = pd.DataFrame([[101.0] * 30, [103.0] * 30], columns=columns)

    pipeline = build_feature_scaling_pipeline()
    transformed_train = pipeline.fit_transform(train)
    transformed_test = pipeline.transform(test)
    scaler = pipeline.named_steps["scaler"]

    assert transformed_train.shape == train.shape
    assert transformed_test.shape == test.shape
    assert scaler.mean_[0] == pytest.approx(train["Time"].mean())
    assert scaler.mean_[0] != pytest.approx(pd.concat([train, test])["Time"].mean())
    assert transformed_test[:, 0].min() > transformed_train[:, 0].max()

