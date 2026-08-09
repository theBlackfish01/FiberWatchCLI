from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.metrics import calibrate_rejection_threshold, open_set_metrics


def test_calibration_separates_known_and_unknown_scores() -> None:
    result = calibrate_rejection_threshold(np.asarray([0.8, 0.9, 1.0]), np.asarray([0.1, 0.2, 0.3]))
    assert 0.3 < result["balanced_threshold"] <= 0.8
    assert result["balanced_h"] == 1.0


def test_open_set_metrics_are_perfect_for_separated_scores() -> None:
    result = open_set_metrics(
        np.asarray([0.9, 0.8, 0.2, 0.1]),
        np.asarray([True, True, False, False]),
        np.asarray([True, True, False, False]),
        threshold=0.5,
    )
    assert result["known_acceptance"] == 1.0
    assert result["unknown_recall"] == 1.0
    assert result["unknown_auroc"] == 1.0
