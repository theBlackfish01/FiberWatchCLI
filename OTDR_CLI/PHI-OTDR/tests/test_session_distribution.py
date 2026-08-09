from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.session_distribution import (
    SessionWindows,
    class_scores,
    select_support,
    session_descriptor,
)


def _session(values: np.ndarray, window_ids: np.ndarray | None = None) -> SessionWindows:
    return SessionWindows(
        session_id="s",
        label=0,
        partition="source_train",
        window_ids=np.arange(1, len(values) + 1) if window_ids is None else window_ids,
        values=np.asarray(values, dtype=np.float32),
    )


def test_ordered_descriptor_changes_when_trajectory_reverses() -> None:
    values = np.asarray([[0.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
    forward = session_descriptor(_session(values), "ordered_trajectory")
    reverse = session_descriptor(_session(values[::-1]), "ordered_trajectory")
    robust_forward = session_descriptor(_session(values), "robust_quantiles")
    robust_reverse = session_descriptor(_session(values[::-1]), "robust_quantiles")
    assert not np.allclose(forward, reverse)
    assert np.allclose(robust_forward, robust_reverse)


def test_one_window_trajectory_is_finite() -> None:
    descriptor = session_descriptor(_session(np.asarray([[1.0, 2.0]])), "ordered_trajectory")
    assert np.isfinite(descriptor).all()


def test_class_scores_use_nearest_sessions_per_class() -> None:
    gallery = np.asarray([[0.0], [0.2], [10.0], [10.2]])
    labels = np.asarray([0, 0, 1, 1])
    scores = class_scores(np.asarray([[0.1], [10.1]]), gallery, labels, [0, 1], neighbors=2)
    assert np.argmax(scores, axis=1).tolist() == [0, 1]


def test_support_strategies_are_unique_deterministic_and_bounded() -> None:
    candidates = np.asarray([[0.0], [0.1], [0.2], [4.0], [5.0], [6.0], [7.0]])
    gallery = np.asarray([[-2.0], [-1.0]])
    for strategy in (
        "random",
        "medoid",
        "farthest_first",
        "facility_location",
        "uncertainty_diversity",
    ):
        first = select_support(candidates, gallery, strategy=strategy, shot=5, seed=7)
        second = select_support(candidates, gallery, strategy=strategy, shot=5, seed=7)
        assert np.array_equal(first, second)
        assert len(first) == len(set(first.tolist())) == 5
        assert np.min(first) >= 0 and np.max(first) < len(candidates)
