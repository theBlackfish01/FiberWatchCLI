from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.morphology_attributes_v3 import (
    ATTRIBUTE_NAMES,
    _risk_coverage,
    aggregate_attribute_sessions,
    derive_window_attributes,
)
from phi_research.morphology_features import extract_morphology


def test_attributes_are_finite_and_physics_readable() -> None:
    time = np.linspace(0.0, 12.0 * np.pi, 10_000)
    array = np.column_stack(
        [8000.0 + 100.0 * np.sin(time + channel / 3.0) for channel in range(12)]
    ).astype(np.uint16)
    morphology, names = extract_morphology(array)
    attributes, attribute_names = derive_window_attributes(morphology[None, :], names)
    assert attribute_names == ATTRIBUTE_NAMES
    assert attributes.shape == (1, len(ATTRIBUTE_NAMES))
    assert np.isfinite(attributes).all()
    assert attributes[0, attribute_names.index("spatial_width")] >= 0


def test_session_aggregation_is_deterministic_and_preserves_sessions() -> None:
    attributes = np.arange(4 * len(ATTRIBUTE_NAMES), dtype=np.float32).reshape(4, -1)
    sessions = np.asarray(["b", "a", "b", "a"])
    window_ids = np.asarray([2, 2, 1, 1])
    first = aggregate_attribute_sessions(attributes, sessions, window_ids)
    second = aggregate_attribute_sessions(attributes, sessions, window_ids)
    assert np.array_equal(first[0], second[0])
    assert first[1].tolist() == ["a", "b"]
    assert len(first[2]) == len(ATTRIBUTE_NAMES) * 6


def test_risk_coverage_is_zero_for_perfect_predictions() -> None:
    labels = np.arange(6)
    probs = np.eye(6)
    result = _risk_coverage(labels, probs)
    assert result["aurc"] == 0.0
    assert result["points"]["coverage_1.00"]["accuracy"] == 1.0
