import numpy as np

from phi_research.factorization_v1 import (
    _covariance_map,
    _nuisance_projection,
    _source_date_residual,
    transform_fold,
)


def test_source_date_residual_uses_training_dates_only():
    train = np.asarray([[0.0, 1.0], [2.0, 3.0], [10.0, 11.0], [12.0, 13.0]])
    test = np.asarray([[4.0, 5.0], [20.0, 21.0]])
    train_dates = np.asarray(["a", "a", "b", "b"])
    test_dates = np.asarray(["a", "unseen"])
    transformed_train, transformed_test = _source_date_residual(
        train, test, train_dates, test_dates
    )
    assert np.allclose(np.mean(transformed_train[train_dates == "a"], axis=0), np.mean(train, axis=0))
    assert np.allclose(transformed_test[1], test[1])


def test_nuisance_projection_and_coral_are_finite():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(30, 8))
    test = rng.normal(loc=1.0, size=(12, 8))
    dates = np.asarray([f"d{i // 6}" for i in range(30)])
    projected_train, projected_test, singular = _nuisance_projection(
        train, test, dates, rank=2, ridge=1.0
    )
    assert projected_train.shape == train.shape
    assert projected_test.shape == test.shape
    assert np.isfinite(projected_train).all()
    assert len(singular) >= 2
    mapping = _covariance_map(train, test, floor=1e-3, shrinkage=0.1)
    assert mapping.shape == (8, 8)
    assert np.isfinite(mapping).all()


def test_transductive_transform_does_not_accept_query_labels():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(20, 5))
    train = np.zeros(20, dtype=bool)
    train[:12] = True
    test = ~train
    transformed_train, transformed_test, diagnostics = transform_fold(
        "target_unlabelled_mean_alignment",
        features,
        train,
        test,
        test,
        np.asarray(["a"] * 12 + ["b"] * 8),
        {"target_access": "unlabelled evaluation-group features"},
    )
    assert transformed_train.shape == (12, 5)
    assert transformed_test.shape == (8, 5)
    assert diagnostics["alignment_shift_norm"] >= 0
