from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from phi_research.frozen_neural_inference import cohort_summary, load_inventory_paths


def test_load_inventory_paths_normalizes_separators(tmp_path: Path) -> None:
    path = tmp_path / "inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rel_path", "readable"])
        writer.writeheader()
        writer.writerow({"rel_path": r"train\01_background\sample.mat", "readable": "True"})
    assert load_inventory_paths(path) == {"train/01_background/sample.mat"}


def test_cohort_summary_reaggregates_only_selected_windows() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([0, 1, 1, 1])
    probabilities = np.asarray([
        [0.9, 0.1, 0, 0, 0, 0],
        [0.4, 0.6, 0, 0, 0, 0],
        [0.1, 0.9, 0, 0, 0, 0],
        [0.2, 0.8, 0, 0, 0, 0],
    ])
    sessions = np.asarray(["a", "a", "b", "b"])
    paths = np.asarray(["a/1", "a/2", "b/1", "b/2"])
    mask = np.asarray([True, False, True, True])

    summary, arrays = cohort_summary(
        y_true, predictions, probabilities, sessions, paths, mask
    )

    assert summary["window_count"] == 3
    assert summary["session_count"] == 2
    assert summary["window_metrics"]["accuracy"] == 1.0
    assert arrays["session_ids"].tolist() == ["a", "b"]
    assert arrays["session_pred"].tolist() == [0, 1]
