from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.report_visuals_v3 import _reliability, _risk_curve, _summary


def test_uncertainty_metrics_are_perfect_for_one_hot_predictions() -> None:
    labels = np.arange(6)
    probs = np.eye(6)
    summary = _summary(labels, probs)
    assert summary["macro_f1"] == 1.0
    assert summary["ece_10"] == 0.0
    assert summary["aurc"] == 0.0
    assert sum(row["count"] for row in _reliability(labels, probs)) == 6
    _, risk, aurc = _risk_curve(labels, probs)
    assert np.all(risk == 0)
    assert aurc == 0.0
