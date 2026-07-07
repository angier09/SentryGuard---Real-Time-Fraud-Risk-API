from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.evaluate import evaluate_binary_classifier, find_best_threshold


def test_evaluate_binary_classifier_matches_hand_calculated_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = evaluate_binary_classifier(y_true, probabilities, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(0.666667)
    assert metrics["pr_auc_average_precision"] == pytest.approx(0.833333)
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_find_best_threshold_picks_expected_controlled_threshold() -> None:
    y_true = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.4, 0.8, 0.9])

    result = find_best_threshold(
        y_true,
        probabilities,
        min_threshold=0.4,
        max_threshold=0.8,
    )

    assert result == {
        "threshold": 0.4,
        "f1": 0.8,
        "precision": 0.666667,
        "recall": 1.0,
    }

