from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.confirmatory_open import _bootstrap_class_ci


def test_class_bootstrap_is_deterministic_and_contains_mean() -> None:
    result = _bootstrap_class_ci([0.2, 0.4, 0.6, 0.8], np.random.default_rng(3), draws=2000)
    assert result["mean"] == 0.5
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]
    assert result["worst_holdout"] == 0.2
