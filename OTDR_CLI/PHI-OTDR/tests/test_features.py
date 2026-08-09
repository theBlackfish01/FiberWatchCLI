from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.features import extract_features


def test_feature_schema_is_finite_and_deterministic() -> None:
    time = np.linspace(0, 8 * np.pi, 10000, dtype=np.float64)
    array = np.stack([8000 + 20 * np.sin(time + channel / 4) for channel in range(12)], axis=1)
    first = extract_features(array)
    second = extract_features(array)
    assert len(first.names) == len(set(first.names))
    assert first.values.shape == (339,)
    assert np.isfinite(first.values).all()
    np.testing.assert_array_equal(first.values, second.values)


def test_feature_extractor_rejects_wrong_shape() -> None:
    try:
        extract_features(np.zeros((100, 12)))
    except ValueError as exc:
        assert "Expected" in str(exc)
    else:
        raise AssertionError("Wrong-shaped array was accepted")
