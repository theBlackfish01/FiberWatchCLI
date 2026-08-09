from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.morphology_attribute_analysis_v3 import stratified_paired_bootstrap


def test_stratified_paired_bootstrap_is_deterministic_and_detects_improvement() -> None:
    labels = np.repeat(np.arange(6), 4)
    left = labels.copy()
    right = np.roll(labels, 1)
    first = stratified_paired_bootstrap(
        labels, left, right, metric="macro_f1", seed=7, draws=200
    )
    second = stratified_paired_bootstrap(
        labels, left, right, metric="macro_f1", seed=7, draws=200
    )
    assert first == second
    assert first["observed_delta"] > 0
    assert first["ci95_low"] > 0
