from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import get_model_feature_columns

TARGET_COLUMN = "Class"


class DatasetValidationError(ValueError):
    """Raised when the source CSV does not match the expected schema."""


def load_creditcard_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}. Download it with: "
            "uv run kaggle datasets download -d mlg-ulb/creditcardfraud "
            "-p data/raw --unzip"
        )

    df = pd.read_csv(input_path)
    validate_creditcard_dataset(df)
    return df


def validate_creditcard_dataset(df: pd.DataFrame) -> None:
    required_columns = [*get_model_feature_columns(), TARGET_COLUMN]
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise DatasetValidationError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    unknown_targets = sorted(set(df[TARGET_COLUMN].dropna().unique()) - {0, 1})
    if unknown_targets:
        raise DatasetValidationError(
            f"Target column Class must contain only 0 and 1. Found: {unknown_targets}"
        )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = get_model_feature_columns()
    return df[feature_columns].copy(), df[TARGET_COLUMN].astype(int).copy()


def stratified_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target while preserving the fraud class ratio."""

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
