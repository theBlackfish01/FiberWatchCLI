from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.metrics import aggregate_session_predictions, classification_metrics


def test_session_aggregation_uses_mean_probability() -> None:
    y = np.asarray([0, 0, 1, 1])
    sessions = np.asarray(["a", "a", "b", "b"])
    probabilities = np.zeros((4, 6), dtype=float)
    probabilities[0, :2] = (0.8, 0.2)
    probabilities[1, :2] = (0.4, 0.6)
    probabilities[2, :2] = (0.2, 0.8)
    probabilities[3, :2] = (0.1, 0.9)
    true, predicted, ordered = aggregate_session_predictions(y, sessions, probabilities=probabilities)
    assert ordered == ["a", "b"]
    np.testing.assert_array_equal(true, [0, 1])
    np.testing.assert_array_equal(predicted, [0, 1])


def test_classification_metrics_report_worst_class() -> None:
    y_true = np.repeat(np.arange(6), 2)
    y_pred = y_true.copy()
    y_pred[-2:] = 0
    result = classification_metrics(y_true, y_pred)
    assert result["accuracy"] == 10 / 12
    assert result["worst_class_recall"] == 0.0
    assert result["per_class_recall"]["walking"] == 0.0
