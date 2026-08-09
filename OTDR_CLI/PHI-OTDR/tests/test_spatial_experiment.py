from __future__ import annotations

import numpy as np

from phi_research.spatial_experiment import _ece, _metrics, _select_temperature, _temperature


def test_temperature_probabilities_and_selection() -> None:
    probs = np.asarray([[0.8, 0.2, 0, 0, 0, 0], [0.3, 0.7, 0, 0, 0, 0]], dtype=float)
    labels = np.asarray([0, 1])
    value = _select_temperature(labels, probs)
    calibrated = _temperature(probs, value)
    assert calibrated.shape == probs.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert value <= 1.0


def test_metrics_are_session_level_and_finite() -> None:
    labels = np.arange(6)
    probs = np.eye(6) * 0.9 + np.ones((6, 6)) * (0.1 / 6)
    probs /= probs.sum(axis=1, keepdims=True)
    result = _metrics(labels, probs)
    assert result["session_count"] == 6
    assert result["macro_f1"] == 1.0
    assert result["worst_class_recall"] == 1.0
    assert np.isfinite(result["negative_log_likelihood"])
    assert 0.0 <= _ece(labels, probs) <= 1.0
