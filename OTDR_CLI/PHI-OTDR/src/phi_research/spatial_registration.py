"""Physics-motivated spatial activity estimation and non-circular registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Estimator = Literal[
    "temporal_difference_energy",
    "robust_variance",
    "spectral_energy",
    "multi_estimator_consensus",
]


@dataclass(frozen=True)
class RegistrationResult:
    values: np.ndarray
    activity_profile: np.ndarray
    estimated_center: float
    applied_shift: float
    confidence: float
    retained_activity_fraction: float
    clipped_channel_fraction: float
    estimator: str


def _validate(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 4:
        raise ValueError(f"Expected a time-by-channel matrix, received {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Spatial registration requires finite numeric data")
    return values.astype(np.float64, copy=False)


def _unit_profile(profile: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= np.finfo(np.float64).eps:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return values / total


def activity_profile(
    array: np.ndarray,
    estimator: Estimator,
    *,
    temporal_stride: int = 5,
) -> np.ndarray:
    """Return a normalized non-negative channel activity profile.

    The spectral estimator is mathematically distinct but uses Parseval-scale
    non-DC power.  Consensus averages the ranks of three normalized profiles,
    preventing a single estimator's physical units from dominating.
    """
    x = _validate(array)
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    sampled = x[::temporal_stride]
    if estimator == "temporal_difference_energy":
        profile = np.mean(np.diff(sampled, axis=0) ** 2, axis=0)
    elif estimator == "robust_variance":
        median = np.median(sampled, axis=0, keepdims=True)
        mad = np.median(np.abs(sampled - median), axis=0)
        profile = mad**2
    elif estimator == "spectral_energy":
        centered = sampled - np.mean(sampled, axis=0, keepdims=True)
        spectrum = np.fft.rfft(centered, axis=0)
        profile = np.mean(np.abs(spectrum[1:]) ** 2, axis=0)
    elif estimator == "multi_estimator_consensus":
        components = np.stack(
            [
                activity_profile(x, name, temporal_stride=temporal_stride)
                for name in (
                    "temporal_difference_energy",
                    "robust_variance",
                    "spectral_energy",
                )
            ]
        )
        # Each component already has unit mass, so averaging is scale-neutral
        # and, unlike ordinal argsort ranks, does not invent structure in ties.
        profile = np.mean(components, axis=0)
    else:
        raise ValueError(f"Unknown activity estimator: {estimator}")
    return _unit_profile(profile)


def profile_center(profile: np.ndarray) -> tuple[float, float]:
    """Estimate active-region center and concentration confidence.

    Removing the profile floor makes the center respond to localized excess
    energy instead of uniform sensor noise.  Confidence is normalized entropy;
    an exactly uniform profile has zero confidence.
    """
    unit = _unit_profile(profile)
    excess = np.maximum(unit - np.min(unit), 0.0)
    if float(excess.sum()) <= np.finfo(np.float64).eps:
        return (len(unit) - 1) / 2.0, 0.0
    weights = excess / excess.sum()
    positions = np.arange(len(unit), dtype=np.float64)
    center = float(np.sum(weights * positions))
    entropy = float(-np.sum(weights * np.log(weights + 1e-15)))
    confidence = float(np.clip(1.0 - entropy / np.log(len(unit)), 0.0, 1.0))
    return center, confidence


def shift_channels(
    array: np.ndarray,
    shift: float,
    *,
    pad_mode: Literal["baseline", "zero"] = "baseline",
) -> tuple[np.ndarray, float]:
    """Translate along channels with linear interpolation and no wraparound.

    Positive shifts move activity to larger channel indices.  Baseline padding
    subtracts each time row's median before zero padding, then restores it; this
    avoids introducing an artificial zero-intensity edge into raw uint16 traces.
    """
    values = np.asarray(array)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 1:
        raise ValueError(f"Expected a row-by-channel matrix, received {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Channel shifting requires finite numeric data")
    x = values.astype(np.float64, copy=False)
    channels = x.shape[1]
    if pad_mode == "baseline":
        baseline = np.median(x, axis=1, keepdims=True)
        working = x - baseline
    elif pad_mode == "zero":
        baseline = np.zeros((x.shape[0], 1), dtype=np.float64)
        working = x
    else:
        raise ValueError(f"Unknown padding mode: {pad_mode}")
    destination = np.arange(channels, dtype=np.float64)
    source = destination - float(shift)
    left = np.floor(source).astype(np.int64)
    right = left + 1
    alpha = source - left
    valid_left = (left >= 0) & (left < channels)
    valid_right = (right >= 0) & (right < channels)
    result = np.zeros_like(working)
    if np.any(valid_left):
        result[:, valid_left] += working[:, left[valid_left]] * (1.0 - alpha[valid_left])
    if np.any(valid_right):
        result[:, valid_right] += working[:, right[valid_right]] * alpha[valid_right]
    clipped = float(np.mean((source < 0.0) | (source > channels - 1)))
    return (result + baseline).astype(np.float32), clipped


def register_array(
    array: np.ndarray,
    estimator: Estimator = "multi_estimator_consensus",
    *,
    target_center: float | None = None,
    min_confidence: float = 0.02,
    temporal_stride: int = 5,
) -> RegistrationResult:
    """Center the active region while preserving an explicit position record."""
    x = _validate(array)
    profile = activity_profile(x, estimator, temporal_stride=temporal_stride)
    center, confidence = profile_center(profile)
    target = (x.shape[1] - 1) / 2.0 if target_center is None else float(target_center)
    requested_shift = target - center
    applied_shift = requested_shift if confidence >= min_confidence else 0.0
    shifted, clipped = shift_channels(x, applied_shift, pad_mode="baseline")
    shifted_profile = activity_profile(
        shifted, "temporal_difference_energy", temporal_stride=temporal_stride
    )
    original_dynamic = activity_profile(
        x, "temporal_difference_energy", temporal_stride=temporal_stride
    )
    # Compare mass in channels that remain addressable after translation.  The
    # activity profiles are normalized, so this is a dimensionless retention.
    positions = np.arange(x.shape[1], dtype=np.float64)
    valid = (positions + applied_shift >= 0.0) & (positions + applied_shift <= x.shape[1] - 1)
    retained = float(np.sum(original_dynamic[valid]))
    if abs(applied_shift) < 1e-12:
        retained = 1.0
    if not np.isfinite(shifted_profile).all():
        raise AssertionError("Registered activity profile is non-finite")
    return RegistrationResult(
        values=shifted,
        activity_profile=profile.astype(np.float32),
        estimated_center=center,
        applied_shift=float(applied_shift),
        confidence=confidence,
        retained_activity_fraction=float(np.clip(retained, 0.0, 1.0)),
        clipped_channel_fraction=clipped,
        estimator=estimator,
    )


def shift_profile(profile: np.ndarray, shift: float) -> np.ndarray:
    """Translate a 1-D activity profile with zero padding and no wraparound."""
    values = np.asarray(profile, dtype=np.float64)
    shifted, _ = shift_channels(values[None, :], shift, pad_mode="zero")
    return shifted[0]
