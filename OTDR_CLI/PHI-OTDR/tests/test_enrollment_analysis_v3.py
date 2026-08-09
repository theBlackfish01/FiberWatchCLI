from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.enrollment_analysis_v3 import (
    _direction_key,
    bh_qvalues,
    exact_sign_flip_p,
    paired_cluster_comparison,
)


def test_direction_key_normalizes_manifest_mapping() -> None:
    assert (
        _direction_key({"source": "january", "target": "april_may"})
        == "january_to_april_may"
    )


def test_exact_sign_flip_has_expected_six_cluster_resolution() -> None:
    assert exact_sign_flip_p([1, 1, 1, 1, 1, 1]) == 2 / 64
    assert exact_sign_flip_p([1, -1, 1, -1, 1, -1]) == 1.0


def test_bh_qvalues_preserve_order_and_bounds() -> None:
    qvalues = bh_qvalues([0.01, 0.04, 0.03, 0.8])
    assert all(0 <= value <= 1 for value in qvalues)
    assert qvalues[0] <= qvalues[2] <= qvalues[1] <= qvalues[3]


def test_paired_cluster_comparison_uses_identical_support() -> None:
    left = {}
    right = {}
    for class_index, heldout in enumerate(("background", "digging", "knocking", "watering", "shaking", "walking")):
        for draw in range(30):
            key = ("january_to_april_may", heldout, 1, draw)
            common = {
                "direction": key[0],
                "heldout_class": heldout,
                "shot": 1,
                "draw": draw,
                "support_sessions": [f"support-{class_index}-{draw}"],
            }
            left[key] = {**common, "enrollment_h": 0.6}
            right[key] = {**common, "enrollment_h": 0.4}
    result, class_rows = paired_cluster_comparison(
        left,
        right,
        direction="january_to_april_may",
        shot=1,
        metric="enrollment_h",
        seed=7,
        bootstrap_draws=1000,
    )
    assert np.isclose(result["mean_effect"], 0.2)
    assert result["paired_random_draws"] == 180
    assert len(class_rows) == 6
    assert result["exact_sign_flip_two_sided_p"] == 2 / 64
