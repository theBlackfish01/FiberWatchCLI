from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.development_summary import _cluster_ci, _paired


def test_cluster_interval_is_deterministic_and_contains_mean() -> None:
    values = np.asarray([0.2, 0.4, 0.6, 0.8])
    first = _cluster_ci(values, seed=7, draws=1000)
    second = _cluster_ci(values, seed=7, draws=1000)
    assert first == second
    assert first["lower_95"] <= first["mean"] <= first["upper_95"]


def test_paired_comparison_reports_direction() -> None:
    result = _paired(np.asarray([0.8, 0.7, 0.9]), np.asarray([0.5, 0.6, 0.7]))
    assert result["mean_difference"] > 0
    assert result["win_fraction"] == 1.0
