from __future__ import annotations

import numpy as np

from phi_research.spatial_registration import (
    activity_profile,
    profile_center,
    register_array,
    shift_channels,
    shift_profile,
)


def _pulse(channel: int) -> np.ndarray:
    x = np.full((200, 12), 100.0)
    x[:, channel] += np.sin(np.linspace(0, 20, len(x))) * 30.0
    return x


def test_estimators_find_localized_activity() -> None:
    x = _pulse(9)
    for estimator in (
        "temporal_difference_energy",
        "robust_variance",
        "spectral_energy",
        "multi_estimator_consensus",
    ):
        profile = activity_profile(x, estimator, temporal_stride=1)
        center, confidence = profile_center(profile)
        assert np.isclose(profile.sum(), 1.0)
        assert center > 8.5
        assert confidence > 0.1


def test_registration_moves_activity_to_center_without_wrap() -> None:
    result = register_array(_pulse(9), min_confidence=0.0, temporal_stride=1)
    registered = activity_profile(result.values, "temporal_difference_energy", temporal_stride=1)
    center, _ = profile_center(registered)
    assert abs(center - 5.5) < 0.75
    assert result.applied_shift < 0
    assert result.clipped_channel_fraction > 0
    assert result.retained_activity_fraction > 0.99


def test_shift_profile_uses_padding_not_circular_wrap() -> None:
    profile = np.zeros(12)
    profile[-1] = 1.0
    shifted = shift_profile(profile, 2.0)
    assert shifted.sum() == 0.0
    assert shifted[1] == 0.0


def test_baseline_padding_does_not_create_zero_raw_edges() -> None:
    x = np.full((20, 12), 8000.0)
    shifted, _ = shift_channels(x, -3.0, pad_mode="baseline")
    assert np.allclose(shifted, 8000.0)


def test_uniform_background_is_not_shifted_by_confidence_gate() -> None:
    result = register_array(np.full((20, 12), 50.0), min_confidence=0.02)
    assert result.confidence == 0.0
    assert result.applied_shift == 0.0
    assert result.retained_activity_fraction == 1.0
