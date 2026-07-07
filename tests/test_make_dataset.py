from __future__ import annotations

import pandas as pd
import pytest

from src.data.make_dataset import (
    DatasetValidationError,
    split_features_target,
    stratified_train_test_split,
    validate_creditcard_dataset,
)
from src.features.build_features import get_model_feature_columns


def test_validate_creditcard_dataset_raises_clear_error_on_missing_columns(
    synthetic_creditcard_frame: pd.DataFrame,
) -> None:
    frame = synthetic_creditcard_frame.drop(columns=["V14", "Amount"])

    with pytest.raises(DatasetValidationError, match="Amount, V14"):
        validate_creditcard_dataset(frame)


def test_validate_creditcard_dataset_rejects_unknown_class_values(
    synthetic_creditcard_frame: pd.DataFrame,
) -> None:
    frame = synthetic_creditcard_frame.copy()
    frame.loc[0, "Class"] = 2

    with pytest.raises(DatasetValidationError, match="only 0 and 1"):
        validate_creditcard_dataset(frame)


def test_stratified_train_test_split_preserves_class_balance(
    synthetic_creditcard_frame: pd.DataFrame,
) -> None:
    features, target = split_features_target(synthetic_creditcard_frame)

    x_train, x_test, y_train, y_test = stratified_train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=7,
    )

    assert list(x_train.columns) == get_model_feature_columns()
    assert list(x_test.columns) == get_model_feature_columns()
    assert len(x_train) == 60
    assert len(x_test) == 20
    assert y_train.mean() == pytest.approx(target.mean())
    assert y_test.mean() == pytest.approx(target.mean())

