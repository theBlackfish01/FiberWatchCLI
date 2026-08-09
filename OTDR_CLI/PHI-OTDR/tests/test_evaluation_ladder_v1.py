from __future__ import annotations

import numpy as np

from phi_research.evaluation_ladder_v1 import (
    aggregate_session_probabilities,
    classification_metrics,
    eligible_date_class_cells,
    overlap_diagnostics,
    random_session_masks,
    random_window_masks,
)


def test_random_window_split_intentionally_overlaps_sessions() -> None:
    labels = np.repeat(np.arange(6), 10)
    sessions = np.asarray([f"s{class_id}_{item // 5}" for class_id in range(6) for item in range(10)])
    dates = np.asarray(["d1"] * len(labels))
    eras = np.asarray(["e1"] * len(labels))
    train, test = random_window_masks(labels, test_fraction=0.2, seed=7)
    diagnostics = overlap_diagnostics(
        train,
        test,
        sessions=sessions,
        dates=dates,
        labels=labels,
        eras=eras,
    )
    assert diagnostics["train_test_session_overlap"] > 0
    assert diagnostics["train_test_cell_overlap"] == 6


def test_random_session_split_is_group_safe_and_class_stratified() -> None:
    sessions = np.asarray(
        [f"s{class_id}_{session}" for class_id in range(6) for session in range(5) for _ in range(3)]
    )
    labels = np.asarray(
        [class_id for class_id in range(6) for _session in range(5) for _ in range(3)]
    )
    train, test = random_session_masks(
        sessions, labels, test_fraction=0.2, seed=20260808
    )
    assert not set(sessions[train]) & set(sessions[test])
    assert set(labels[train]) == set(range(6))
    assert set(labels[test]) == set(range(6))


def test_eligible_cells_require_both_date_and_class_anchors() -> None:
    rows = {
        "a0": {"date": "d1", "label": 0},
        "a1": {"date": "d1", "label": 0},
        "b0": {"date": "d1", "label": 1},
        "b1": {"date": "d1", "label": 1},
        "c0": {"date": "d2", "label": 0},
        "c1": {"date": "d2", "label": 0},
        "d0": {"date": "d3", "label": 2},
        "d1": {"date": "d3", "label": 2},
    }
    assert eligible_date_class_cells(rows, min_sessions=2) == [("d1", 0)]


def test_session_probability_aggregation_and_metrics() -> None:
    labels = np.asarray([0, 0, 1, 1])
    sessions = np.asarray(["a", "a", "b", "b"])
    probabilities = np.asarray(
        [
            [0.8, 0.2, 0, 0, 0, 0],
            [0.6, 0.4, 0, 0, 0, 0],
            [0.2, 0.8, 0, 0, 0, 0],
            [0.1, 0.9, 0, 0, 0, 0],
        ]
    )
    session_ids, session_labels, session_probs, counts = aggregate_session_probabilities(
        labels, sessions, probabilities
    )
    assert session_ids.tolist() == ["a", "b"]
    assert session_labels.tolist() == [0, 1]
    assert counts.tolist() == [2, 2]
    metrics = classification_metrics(session_labels, session_probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy_observed_classes"] == 1.0
    assert metrics["per_class_recall"]["shaking"] is None
