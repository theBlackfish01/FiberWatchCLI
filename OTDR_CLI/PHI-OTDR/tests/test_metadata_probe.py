from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.metadata_probe import categorical_to_class_probe, inventory_confounding


def test_categorical_probe_uses_session_rows_and_handles_unseen_values() -> None:
    table = {
        "label": np.asarray([0, 0, 1, 1, 0, 1]),
        "date_token": np.asarray(["d0", "d0", "d1", "d1", "d0", "unseen"]),
        "source_token": np.asarray(["a", "a", "b", "b", "a", "b"]),
    }
    train = np.asarray([True, True, True, True, False, False])
    validation = ~train
    result = categorical_to_class_probe(table, train, validation, ("date_token",))
    assert result["validation"]["samples"] == 2
    assert result["unseen_validation_values"]["date_token"] == ["unseen"]


def test_inventory_confounding_counts_single_class_groups() -> None:
    table = {
        "label": np.asarray([0, 0, 1, 1]),
        "date_token": np.asarray(["d0", "d0", "d1", "d2"]),
        "source_token": np.asarray(["a", "a", "b", "b"]),
        "date_source": np.asarray(["d0|a", "d0|a", "d1|b", "d2|b"]),
    }
    result = inventory_confounding(table)
    assert result["date_token"]["group_count"] == 3
    assert result["date_token"]["single_class_group_count"] == 3
