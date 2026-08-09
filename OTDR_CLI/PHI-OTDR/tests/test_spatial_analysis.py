from __future__ import annotations

import numpy as np

from phi_research.spatial_analysis import _bh_qvalues, _stratified_bootstrap


def test_bh_qvalues_are_bounded_and_monotone_by_pvalue() -> None:
    p = [0.01, 0.04, 0.03]
    q = _bh_qvalues(p)
    assert all(0 <= value <= 1 for value in q)
    assert q[0] <= q[2] <= q[1]


def test_paired_bootstrap_detects_perfect_improvement() -> None:
    labels = np.repeat(np.arange(6), 4)
    primary = labels.copy()
    control = np.zeros_like(labels)
    result = _stratified_bootstrap(labels, primary, control, draws=200, seed=4)
    assert result["observed_delta_macro_f1"] > 0.7
    assert result["ci95_low"] > 0.0
    assert result["bootstrap_probability_delta_le_zero"] == 0.0
