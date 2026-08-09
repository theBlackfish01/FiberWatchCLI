from __future__ import annotations

import numpy as np

from phi_research.confirmatory_supervised import calibration_metrics, risk_coverage_metrics, session_probabilities


def test_calibration_metrics_perfect_probabilities() -> None:
    y = np.asarray([0, 1])
    probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = calibration_metrics(y, probabilities)
    assert result["expected_calibration_error_15bin"] == 0.0
    assert result["multiclass_brier"] == 0.0


def test_session_probabilities_average_windows() -> None:
    y = np.asarray([0, 0, 1])
    sessions = np.asarray(["a", "a", "b"])
    probabilities = np.asarray([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    session_true, session_prob, session_ids = session_probabilities(y, sessions, probabilities)
    assert session_ids.tolist() == ["a", "b"]
    assert session_true.tolist() == [0, 1]
    np.testing.assert_allclose(session_prob, [[0.8, 0.2], [0.2, 0.8]])


def test_risk_coverage_ranks_correct_high_confidence_first() -> None:
    y = np.asarray([0, 1, 1, 0])
    probabilities = np.asarray([[0.99, 0.01], [0.1, 0.9], [0.4, 0.6], [0.45, 0.55]])
    result = risk_coverage_metrics(y, probabilities)
    assert result["curve"][2] == {"coverage": 0.75, "risk": 0.0}
    assert result["risk_at_100pct_coverage"] == 0.25
