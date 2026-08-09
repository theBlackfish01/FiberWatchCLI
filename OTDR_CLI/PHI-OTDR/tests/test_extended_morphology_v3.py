from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.extended_morphology_v3 import WINDOW_FEATURE_NAMES, extract_wavelet_rank


def test_wavelet_rank_is_finite_bounded_and_energy_normalized() -> None:
    rng = np.random.default_rng(20260808)
    array = rng.integers(0, 65535, size=(10000, 12), dtype=np.uint16)
    result = extract_wavelet_rank(array)
    assert result.shape == (len(WINDOW_FEATURE_NAMES),)
    assert np.isfinite(result).all()
    assert np.isclose(np.sum(result[:5]), 1.0, atol=1e-6)
    assert 0.0 <= result[5] <= 1.0
    assert 0.0 <= result[6] <= 1.0


def test_wavelet_rank_rejects_wrong_shape() -> None:
    with np.testing.assert_raises_regex(ValueError, "Expected"):
        extract_wavelet_rank(np.zeros((100, 12), dtype=np.uint16))
