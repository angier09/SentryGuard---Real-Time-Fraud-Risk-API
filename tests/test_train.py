from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import get_model_feature_columns
from src.training.train import run_training


def test_run_training_end_to_end_on_small_dataset(
    synthetic_creditcard_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "creditcard.csv"
    artifacts_dir = tmp_path / "artifacts"
    synthetic_creditcard_frame.to_csv(input_path, index=False)

    summary = run_training(
        input_path=input_path,
        artifacts_dir=artifacts_dir,
        test_size=0.25,
        validation_size=0.25,
    )

    model_payload = joblib.load(artifacts_dir / "model.joblib")
    threshold_payload = json.loads((artifacts_dir / "threshold.json").read_text())
    metrics_payload = json.loads((artifacts_dir / "metrics.json").read_text())
    sample_features = synthetic_creditcard_frame[get_model_feature_columns()].head(3)

    assert summary["selected_model"] in {
        "logistic_regression_baseline",
        "xgboost",
    }
    assert 0 <= threshold_payload["threshold"] <= 1
    assert metrics_payload["dataset"]["rows"] == len(synthetic_creditcard_frame)
    assert hasattr(model_payload["model"], "predict_proba")
    assert model_payload["model"].predict_proba(sample_features).shape == (3, 2)
