from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.make_dataset import (
    load_creditcard_dataset,
    split_features_target,
    stratified_train_test_split,
)
from src.evaluation.evaluate import evaluate_binary_classifier, find_best_threshold
from src.features.build_features import (
    build_feature_scaling_pipeline,
    get_model_feature_columns,
)

RANDOM_STATE = 42
MODEL_VERSION = "xgb_v1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fraud detection models for SentryGuard."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/creditcard.csv"),
        help="Path to Kaggle creditcard.csv.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for model.joblib, metrics.json, and threshold.json.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Held-out test size.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Validation size taken from the training split.",
    )
    return parser.parse_args()


def build_logistic_regression_pipeline() -> Pipeline:
    preprocessing = build_feature_scaling_pipeline()
    return Pipeline(
        steps=[
            *preprocessing.steps,
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(y_train: pd.Series) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not be imported. On macOS Apple Silicon, install "
            "the OpenMP runtime with: brew install libomp"
        ) from exc

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("classifier", classifier)])


def predict_positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features))[:, 1]


def train_candidate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    candidates = {
        "logistic_regression_baseline": build_logistic_regression_pipeline(),
        "xgboost": build_xgboost_pipeline(y_train),
    }
    for model in candidates.values():
        model.fit(x_train, y_train)
    return candidates


def select_best_model(
    candidates: dict[str, Pipeline],
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[str, Pipeline, dict[str, Any]]:
    selection_report: dict[str, Any] = {}
    best_name = ""
    best_score = -1.0

    y_validation_array = y_validation.to_numpy()
    for name, model in candidates.items():
        probabilities = predict_positive_probability(model, x_validation)
        threshold_result = find_best_threshold(y_validation_array, probabilities)
        validation_metrics = evaluate_binary_classifier(
            y_validation_array,
            probabilities,
            threshold_result["threshold"],
        )
        validation_ap = float(
            average_precision_score(y_validation_array, probabilities)
        )
        selection_report[name] = {
            "validation": validation_metrics,
            "threshold_search": threshold_result,
        }
        if validation_ap > best_score:
            best_name = name
            best_score = validation_ap

    return best_name, candidates[best_name], selection_report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def save_artifacts(
    *,
    model: Pipeline,
    model_name: str,
    threshold: float,
    metrics: dict[str, Any],
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_version = MODEL_VERSION if model_name == "xgboost" else "logreg_v1.0.0"
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "feature_schema": "kaggle_creditcard_v1",
    }
    payload = {
        "model": model,
        "feature_names": get_model_feature_columns(),
        "metadata": metadata,
    }
    joblib.dump(payload, artifacts_dir / "model.joblib")
    write_json(
        artifacts_dir / "threshold.json",
        {
            "threshold": round(float(threshold), 6),
            "selection_metric": "validation_f1",
            "selected_at_utc": metadata["trained_at_utc"],
        },
    )
    write_json(artifacts_dir / "metrics.json", metrics)


def run_training(
    *,
    input_path: Path,
    artifacts_dir: Path,
    test_size: float = 0.20,
    validation_size: float = 0.20,
) -> dict[str, Any]:
    df = load_creditcard_dataset(input_path)
    x, y = split_features_target(df)

    (
        x_train_validation,
        x_test,
        y_train_validation,
        y_test,
    ) = stratified_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=y_train_validation,
    )

    candidates = train_candidate_models(x_train, y_train)
    best_name, best_model, selection_report = select_best_model(
        candidates,
        x_validation,
        y_validation,
    )

    validation_threshold = selection_report[best_name]["threshold_search"]["threshold"]
    y_test_array = y_test.to_numpy()
    test_probabilities = predict_positive_probability(best_model, x_test)
    test_metrics = evaluate_binary_classifier(
        y_test_array,
        test_probabilities,
        validation_threshold,
    )

    best_model.fit(x_train_validation, y_train_validation)

    metrics = {
        "dataset": {
            "input_path": str(input_path),
            "rows": int(len(df)),
            "features": len(get_model_feature_columns()),
            "fraud_rate": round(float(y.mean()), 8),
            "test_size": test_size,
            "validation_size_from_training": validation_size,
        },
        "selected_model": best_name,
        "model_selection": selection_report,
        "test": test_metrics,
        "notes": (
            "Threshold selected on validation data. Test metrics are computed "
            "once on the held-out test split; no SMOTE or resampling is applied "
            "to validation or test data."
        ),
    }

    save_artifacts(
        model=best_model,
        model_name=best_name,
        threshold=validation_threshold,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
    )

    return {
        "selected_model": best_name,
        "threshold": validation_threshold,
        "test": test_metrics,
        "artifacts_dir": str(artifacts_dir),
    }


def main() -> None:
    args = parse_args()
    summary = run_training(
        input_path=args.input,
        artifacts_dir=args.artifacts_dir,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
