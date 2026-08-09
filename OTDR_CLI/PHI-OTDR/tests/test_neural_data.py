from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.neural_data import normalize_window


def test_global_minmax_is_local_and_bounded() -> None:
    array = np.arange(120, dtype=np.uint16).reshape(10, 12)
    normalized = normalize_window(array, "global_minmax")
    assert normalized.dtype == np.float32
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0


def test_channel_zscore_normalizes_each_channel() -> None:
    array = np.arange(1200, dtype=np.float32).reshape(100, 12)
    normalized = normalize_window(array, "channel_zscore")
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-5)
