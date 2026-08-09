from __future__ import annotations

import numpy as np

from phi_research.distributional_enrollment_v3 import (
    _aggregate_descriptor,
    _classification_metrics,
    select_support,
)


def test_support_selectors_are_unique_deterministic_and_query_free() -> None:
    candidates = np.asarray([[0.0], [1.0], [2.0], [9.0], [10.0]])
    for selector in ("random", "medoid", "k_center", "pool_coverage"):
        first = select_support(candidates, selector=selector, shot=3, seed=12)
        second = select_support(candidates, selector=selector, shot=3, seed=12)
        assert np.array_equal(first, second)
        assert len(first) == len(set(first.tolist())) == 3
        assert np.all((first >= 0) & (first < len(candidates)))


def test_enrollment_metrics_use_session_predictions() -> None:
    labels = np.asarray([0, 1, 2, 3, 4, 5])
    result = _classification_metrics(labels, labels.copy(), holdout=5)
    assert result["session_macro_f1"] == 1.0
    assert result["base_class_accuracy"] == 1.0
    assert result["enrolled_class_recall"] == 1.0
    assert result["enrollment_h"] == 1.0


def test_hash_sized_seed_is_accepted_by_pca() -> None:
    values = np.arange(120, dtype=float).reshape(20, 6)
    fit = np.arange(20) < 15
    result = _aggregate_descriptor(values, fit, components=3, seed=2**63 + 17)
    assert result.shape == (20, 3)
    assert np.isfinite(result).all()
