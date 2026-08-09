from __future__ import annotations

import numpy as np

from phi_research.morphology_features import aggregate_sessions, extract_morphology, transform_view


def _signal(channel: int = 8) -> np.ndarray:
    rng = np.random.default_rng(4)
    x = rng.normal(8000.0, 2.0, size=(10000, 12))
    x[:, channel] += np.sin(np.linspace(0, 1000, len(x))) * 100.0
    return x


def test_morphology_is_finite_and_named() -> None:
    values, names = extract_morphology(_signal())
    assert len(values) == len(names)
    assert len(values) > 70
    assert np.isfinite(values).all()
    assert len(set(names)) == len(names)


def test_all_spatial_views_and_ablations_are_finite() -> None:
    base, names = extract_morphology(_signal())
    for view in ("absolute", "invariant", "registered", "registered_position", "dual"):
        for ablation in ("amplitude", "dynamics", "fused"):
            values, view_names = transform_view(
                base,
                names,
                view=view,
                estimator="multi_estimator_consensus",
                ablation=ablation,
            )
            assert len(values) == len(view_names)
            assert len(values) > 3
            assert np.isfinite(values).all()


def test_registered_profile_moves_peak_and_position_view_retains_location() -> None:
    base, names = extract_morphology(_signal(channel=9))
    registered, registered_names = transform_view(
        base, names, view="registered", estimator="temporal_difference_energy"
    )
    channels = np.asarray(
        [value for value, name in zip(registered, registered_names) if name.startswith("registered_difference_ch")]
    )
    assert np.argmax(channels) in {5, 6}
    positioned, positioned_names = transform_view(
        base, names, view="registered_position", estimator="temporal_difference_energy"
    )
    center = positioned[positioned_names.index("position_center")]
    assert center > 8.0


def test_forced_shift_augmentation_moves_absolute_profile_without_wrap() -> None:
    base, names = extract_morphology(_signal(channel=9))
    original, original_names = transform_view(
        base, names, view="absolute", estimator="multi_estimator_consensus", ablation="dynamics"
    )
    shifted, shifted_names = transform_view(
        base,
        names,
        view="absolute",
        estimator="multi_estimator_consensus",
        ablation="dynamics",
        forced_shift=-2.0,
    )
    assert original_names == shifted_names
    key = [i for i, name in enumerate(original_names) if name.startswith("absolute_difference_ch")]
    assert np.argmax(shifted[key]) == np.argmax(original[key]) - 2


def test_session_aggregation_preserves_sessions_and_sequence_slope() -> None:
    x = np.asarray([[0.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    features, sessions, names = aggregate_sessions(x, ["a", "a", "b"], np.asarray([1, 2, 1]))
    assert sessions.tolist() == ["a", "b"]
    assert features.shape == (2, 12)
    assert len(names) == 12
    assert features[0, -2] > 0
    assert features[1, -2] == 0
