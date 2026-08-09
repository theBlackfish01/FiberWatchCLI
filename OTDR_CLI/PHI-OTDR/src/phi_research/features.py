"""Deterministic, physically interpretable window features for Phi-OTDR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureVector:
    values: np.ndarray
    names: tuple[str, ...]


def _append_channel_features(
    values: list[float], names: list[str], prefix: str, matrix: np.ndarray
) -> None:
    for channel in range(matrix.shape[1]):
        for statistic, vector in (
            ("mean", np.mean(matrix, axis=0)),
            ("std", np.std(matrix, axis=0)),
            ("range", np.ptp(matrix, axis=0)),
        ):
            values.append(float(vector[channel]))
            names.append(f"{prefix}_ch{channel:02d}_{statistic}")


def extract_features(array: np.ndarray, *, temporal_stride: int = 5) -> FeatureVector:
    """Extract amplitude, dynamics, spectral, spatial, and correlation features.

    Frequencies are normalized because the dataset contains no authoritative
    sample-rate metadata.  No learned or cross-sample normalization occurs.
    """
    if array.shape != (10000, 12):
        raise ValueError(f"Expected (10000, 12), received {array.shape}")
    x = array.astype(np.float64, copy=False)
    sampled = x[::temporal_stride]
    channel_count = sampled.shape[1]
    values: list[float] = []
    names: list[str] = []

    _append_channel_features(values, names, "raw", sampled)
    quantiles = np.quantile(sampled, (0.05, 0.25, 0.50, 0.75, 0.95), axis=0)
    for q_index, q_name in enumerate(("q05", "q25", "q50", "q75", "q95")):
        for channel in range(channel_count):
            values.append(float(quantiles[q_index, channel]))
            names.append(f"raw_ch{channel:02d}_{q_name}")

    delta = np.diff(sampled, axis=0)
    abs_delta = np.abs(delta)
    delta_statistics = (
        ("mean_abs", np.mean(abs_delta, axis=0)),
        ("std", np.std(delta, axis=0)),
        ("p95_abs", np.quantile(abs_delta, 0.95, axis=0)),
    )
    for statistic, vector in delta_statistics:
        for channel in range(channel_count):
            values.append(float(vector[channel]))
            names.append(f"delta_ch{channel:02d}_{statistic}")

    centered = sampled - np.mean(sampled, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    spectrum = spectrum[1:]
    total_power = np.sum(spectrum, axis=0) + 1e-12
    normalized = spectrum / total_power
    bin_count = normalized.shape[0]
    band_edges = (0, max(1, int(0.04 * bin_count)), max(2, int(0.16 * bin_count)), max(3, int(0.50 * bin_count)), bin_count)
    for band in range(4):
        band_power = np.sum(normalized[band_edges[band] : band_edges[band + 1]], axis=0)
        for channel in range(channel_count):
            values.append(float(band_power[channel]))
            names.append(f"spectrum_ch{channel:02d}_band{band}")
    entropy = -np.sum(normalized * np.log(normalized + 1e-12), axis=0) / np.log(max(bin_count, 2))
    dominant = np.argmax(normalized, axis=0) / max(bin_count - 1, 1)
    for statistic, vector in (("entropy", entropy), ("dominant_frequency", dominant)):
        for channel in range(channel_count):
            values.append(float(vector[channel]))
            names.append(f"spectrum_ch{channel:02d}_{statistic}")

    blocks = np.array_split(sampled, 20, axis=0)
    block_dynamic = np.stack(
        [np.std(block - np.mean(block, axis=0, keepdims=True), axis=0) for block in blocks]
    )
    block_axis = np.linspace(-1.0, 1.0, block_dynamic.shape[0])
    block_slope = np.sum(block_dynamic * block_axis[:, None], axis=0) / np.sum(block_axis**2)
    for statistic, vector in (
        ("mean", np.mean(block_dynamic, axis=0)),
        ("std", np.std(block_dynamic, axis=0)),
        ("max", np.max(block_dynamic, axis=0)),
        ("slope", block_slope),
    ):
        for channel in range(channel_count):
            values.append(float(vector[channel]))
            names.append(f"block_dynamic_ch{channel:02d}_{statistic}")

    correlation = np.nan_to_num(np.corrcoef(sampled, rowvar=False), nan=0.0)
    for left in range(channel_count):
        for right in range(left + 1, channel_count):
            values.append(float(correlation[left, right]))
            names.append(f"correlation_ch{left:02d}_ch{right:02d}")

    for channel in range(channel_count - 1):
        values.append(float(correlation[channel, channel + 1]))
        names.append(f"neighbor_correlation_ch{channel:02d}_ch{channel + 1:02d}")

    channel_positions = np.arange(channel_count, dtype=np.float64)
    energy = block_dynamic**2 + 1e-12
    centroids = np.sum(energy * channel_positions[None, :], axis=1) / np.sum(energy, axis=1)
    centroid_slope = float(np.sum((centroids - centroids.mean()) * block_axis) / np.sum(block_axis**2))
    for statistic, value in (
        ("mean", np.mean(centroids)),
        ("std", np.std(centroids)),
        ("range", np.ptp(centroids)),
        ("slope", centroid_slope),
    ):
        values.append(float(value))
        names.append(f"spatial_energy_centroid_{statistic}")

    global_stats = (
        ("mean", np.mean(sampled)),
        ("std", np.std(sampled)),
        ("range", np.ptp(sampled)),
        ("delta_mean_abs", np.mean(abs_delta)),
        ("delta_std", np.std(delta)),
        ("mean_neighbor_correlation", np.mean(np.diag(correlation, k=1))),
    )
    for statistic, value in global_stats:
        values.append(float(value))
        names.append(f"global_{statistic}")

    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Feature extraction produced non-finite values")
    return FeatureVector(values=vector, names=tuple(names))
