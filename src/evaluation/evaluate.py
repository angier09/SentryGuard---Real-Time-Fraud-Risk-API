from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_predictions(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = threshold_predictions(probabilities, threshold)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "precision": round(
            float(precision_score(y_true, predictions, zero_division=0)), 6
        ),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "pr_auc_average_precision": round(
            float(average_precision_score(y_true, probabilities)), 6
        ),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
) -> dict[str, float]:
    """Select a threshold by maximizing F1 on validation data.

    Fraud systems usually cannot optimize accuracy because the majority class
    dominates. F1 is a practical default when precision and recall both matter.
    """

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(min_threshold, max_threshold, 99),
                np.quantile(probabilities, np.linspace(0.01, 0.99, 99)),
            ]
        )
    )
    candidate_thresholds = candidate_thresholds[
        (candidate_thresholds >= min_threshold)
        & (candidate_thresholds <= max_threshold)
    ]

    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in candidate_thresholds:
        predictions = threshold_predictions(probabilities, float(threshold))
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        if (f1, recall, precision) > (
            best["f1"],
            best["recall"],
            best["precision"],
        ):
            best = {
                "threshold": float(threshold),
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

    return {key: round(value, 6) for key, value in best.items()}
