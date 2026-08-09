"""Compact interpretable Phi-OTDR morphology descriptors and spatial views."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from .data_contract import canonical_json_hash
from .dataset import SampleRef, build_sample_index, load_array, load_manifest
from .spatial_registration import activity_profile, profile_center, shift_profile


View = Literal["absolute", "invariant", "registered", "registered_position", "dual"]
Ablation = Literal["amplitude", "dynamics", "fused"]
PROFILE_NAMES = ("amplitude", "difference", "robust", "spectral", "consensus")
ESTIMATOR_TO_PROFILE = {
    "temporal_difference_energy": "difference",
    "robust_variance": "robust",
    "spectral_energy": "spectral",
    "multi_estimator_consensus": "consensus",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(values: np.ndarray) -> tuple[np.ndarray, float]:
    nonnegative = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    scale = float(np.mean(nonnegative))
    total = float(nonnegative.sum())
    if total <= 1e-12:
        return np.full(len(nonnegative), 1.0 / len(nonnegative)), -27.6310211
    return nonnegative / total, float(np.log(scale + 1e-12))


def extract_morphology(array: np.ndarray, *, temporal_stride: int = 5) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract window-level temporal, spectral, correlation, and spatial morphology."""
    if array.shape != (10000, 12):
        raise ValueError(f"Expected (10000, 12), received {array.shape}")
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")
    x = array[::temporal_stride].astype(np.float64, copy=False)
    delta = np.diff(x, axis=0)
    abs_delta = np.abs(delta)
    centered = x - np.mean(x, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    spectrum = spectrum[1:]
    total_spectrum = np.sum(spectrum, axis=0) + 1e-12
    frequencies = np.linspace(0.0, 1.0, spectrum.shape[0], dtype=np.float64)[:, None]
    low_spectrum = np.sum(spectrum[: max(1, spectrum.shape[0] // 8)], axis=0)
    spectral_centroid = np.sum(spectrum * frequencies, axis=0) / total_spectrum

    values: list[float] = []
    names: list[str] = []

    def scalar(name: str, value: float) -> None:
        values.append(float(value))
        names.append(name)

    scalar("amplitude_global_mean", np.mean(x))
    scalar("amplitude_global_std", np.std(x))
    scalar("amplitude_global_range", np.ptp(x))
    scalar("dynamics_delta_rms", np.sqrt(np.mean(delta**2)))
    scalar("dynamics_delta_mean_abs", np.mean(abs_delta))
    scalar("dynamics_delta_p95_abs", np.quantile(abs_delta, 0.95))
    scalar(
        "dynamics_delta_burstiness",
        np.quantile(abs_delta, 0.95) / (np.quantile(abs_delta, 0.50) + 1e-6),
    )
    global_power = np.sum(spectrum, axis=1)
    global_power /= np.sum(global_power) + 1e-12
    edges = (0, max(1, len(global_power) // 32), max(2, len(global_power) // 8), max(3, len(global_power) // 2), len(global_power))
    for band in range(4):
        scalar(f"dynamics_spectrum_band{band}", np.sum(global_power[edges[band] : edges[band + 1]]))
    scalar(
        "dynamics_spectrum_entropy",
        -np.sum(global_power * np.log(global_power + 1e-12)) / np.log(max(len(global_power), 2)),
    )
    scalar("dynamics_spectrum_centroid", np.sum(global_power * np.linspace(0.0, 1.0, len(global_power))))

    correlation = np.nan_to_num(np.corrcoef(delta, rowvar=False), nan=0.0)
    off_diagonal = correlation[np.triu_indices(correlation.shape[0], k=1)]
    neighbors = np.diag(correlation, k=1)
    for statistic, value in (
        ("mean", np.mean(off_diagonal)),
        ("std", np.std(off_diagonal)),
        ("q10", np.quantile(off_diagonal, 0.10)),
        ("q90", np.quantile(off_diagonal, 0.90)),
        ("neighbor_mean", np.mean(neighbors)),
    ):
        scalar(f"dynamics_correlation_{statistic}", value)

    blocks = np.array_split(delta, 20, axis=0)
    block_energy = np.asarray([np.sqrt(np.mean(block**2)) for block in blocks])
    block_axis = np.linspace(-1.0, 1.0, len(block_energy))
    for statistic, value in (
        ("mean", np.mean(block_energy)),
        ("std", np.std(block_energy)),
        ("max", np.max(block_energy)),
        ("slope", np.sum((block_energy - block_energy.mean()) * block_axis) / np.sum(block_axis**2)),
    ):
        scalar(f"dynamics_temporal_block_{statistic}", value)

    raw_profiles = {
        "amplitude": np.std(x, axis=0),
        "difference": np.mean(delta**2, axis=0),
        "robust": np.median(np.abs(x - np.median(x, axis=0, keepdims=True)), axis=0) ** 2,
        "spectral": low_spectrum,
    }
    normalized = {name: _unit(profile) for name, profile in raw_profiles.items()}
    normalized["consensus"] = (
        np.mean(np.stack([normalized[name][0] for name in ("difference", "robust", "spectral")]), axis=0),
        float(np.mean([normalized[name][1] for name in ("difference", "robust", "spectral")])),
    )
    for profile_name in PROFILE_NAMES:
        profile, log_scale = normalized[profile_name]
        for channel, value in enumerate(profile):
            scalar(f"{profile_name}_profile_ch{channel:02d}", value)
        scalar(f"{profile_name}_log_scale", log_scale)
        center, confidence = profile_center(profile)
        scalar(f"{profile_name}_center", center)
        scalar(f"{profile_name}_confidence", confidence)
        scalar(f"{profile_name}_spectral_centroid_mean", np.mean(spectral_centroid))

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Morphology extraction produced non-finite values")
    return result, tuple(names)


def _profile_indices(names: Sequence[str], profile_name: str) -> np.ndarray:
    return np.asarray(
        [i for i, name in enumerate(names) if name.startswith(f"{profile_name}_profile_ch")],
        dtype=np.int64,
    )


def _invariant_profile(profile: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    profile = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
    profile /= profile.sum() + 1e-12
    center, _ = profile_center(profile)
    position = np.arange(len(profile), dtype=np.float64) - center
    width = np.sqrt(np.sum(profile * position**2))
    entropy = -np.sum(profile * np.log(profile + 1e-12)) / np.log(len(profile))
    sorted_values = np.sort(profile)[::-1]
    fft_magnitude = np.abs(np.fft.rfft(profile))
    values = [*sorted_values.tolist(), *fft_magnitude[1:].tolist(), float(width), float(entropy), float(np.max(profile))]
    names = (
        [f"{prefix}_sorted_{i:02d}" for i in range(len(sorted_values))]
        + [f"{prefix}_spatial_fft_{i:02d}" for i in range(1, len(fft_magnitude))]
        + [f"{prefix}_width", f"{prefix}_entropy", f"{prefix}_peak_share"]
    )
    return values, names


def transform_view(
    base: np.ndarray,
    feature_names: Sequence[str],
    *,
    view: View,
    estimator: str,
    ablation: Ablation = "fused",
    forced_shift: float = 0.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Transform base descriptors into one frozen spatial representation."""
    values = np.asarray(base, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(feature_names):
        raise ValueError("Base descriptor and feature names disagree")
    profile_key = ESTIMATOR_TO_PROFILE.get(estimator)
    if profile_key is None:
        raise ValueError(f"Unknown registration estimator: {estimator}")
    include_amplitude = ablation in {"amplitude", "fused"}
    include_dynamics = ablation in {"dynamics", "fused"}
    scalar_indices = [
        i
        for i, name in enumerate(feature_names)
        if (include_amplitude and name.startswith("amplitude_global_"))
        or (include_dynamics and name.startswith("dynamics_"))
        or (
            (include_amplitude and name.startswith("amplitude_log_scale"))
            or (include_dynamics and any(name.startswith(f"{p}_log_scale") for p in PROFILE_NAMES if p != "amplitude"))
        )
    ]
    out = values[scalar_indices].tolist()
    out_names = [str(feature_names[i]) for i in scalar_indices]
    chosen_profiles = []
    if include_amplitude:
        chosen_profiles.append("amplitude")
    if include_dynamics:
        chosen_profiles.extend(("difference", "robust", "spectral", "consensus"))
    center_index = feature_names.index(f"{profile_key}_center")
    confidence_index = feature_names.index(f"{profile_key}_confidence")
    center = float(values[center_index])
    confidence = float(values[confidence_index])
    shift = 5.5 - center if confidence >= 0.02 else 0.0

    def append_absolute(prefix: str = "absolute") -> None:
        for name in chosen_profiles:
            profile = values[_profile_indices(feature_names, name)]
            if abs(forced_shift) > 1e-12:
                profile = shift_profile(profile, forced_shift)
            out.extend(profile.tolist())
            out_names.extend(f"{prefix}_{name}_ch{i:02d}" for i in range(len(profile)))

    def append_registered(prefix: str = "registered") -> None:
        for name in chosen_profiles:
            profile = shift_profile(values[_profile_indices(feature_names, name)], shift)
            out.extend(profile.tolist())
            out_names.extend(f"{prefix}_{name}_ch{i:02d}" for i in range(len(profile)))

    if view == "absolute":
        append_absolute()
    elif view == "invariant":
        for name in chosen_profiles:
            invariant, invariant_names = _invariant_profile(
                values[_profile_indices(feature_names, name)], f"invariant_{name}"
            )
            out.extend(invariant)
            out_names.extend(invariant_names)
    elif view == "registered":
        append_registered()
    elif view == "registered_position":
        append_registered()
        out.extend([center, confidence, shift, abs(shift) / 11.0])
        out_names.extend(("position_center", "position_confidence", "position_shift", "position_clipped_proxy"))
    elif view == "dual":
        append_absolute()
        append_registered()
        out.extend([center, confidence, shift, abs(shift) / 11.0])
        out_names.extend(("position_center", "position_confidence", "position_shift", "position_clipped_proxy"))
    else:
        raise ValueError(f"Unknown view: {view}")
    result = np.asarray(out, dtype=np.float32)
    if not np.isfinite(result).all() or len(result) != len(out_names):
        raise AssertionError("Spatial view transformation is invalid")
    return result, tuple(out_names)


def aggregate_sessions(
    window_features: np.ndarray,
    sessions: Sequence[str],
    window_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Aggregate windows with robust distribution and ordered-trajectory summaries."""
    x = np.asarray(window_features, dtype=np.float64)
    session_array = np.asarray(sessions).astype(str)
    ids = np.asarray(window_ids, dtype=np.float64)
    unique = np.asarray(sorted(set(session_array.tolist())))
    aggregates = []
    names: tuple[str, ...] | None = None
    for session in unique:
        mask = session_array == session
        local = x[mask]
        order = np.argsort(ids[mask])
        local = local[order]
        axis = np.linspace(-1.0, 1.0, len(local))
        if len(local) > 1:
            slope = np.sum((local - local.mean(axis=0)) * axis[:, None], axis=0) / np.sum(axis**2)
        else:
            slope = np.zeros(local.shape[1], dtype=np.float64)
        parts = (
            np.mean(local, axis=0),
            np.std(local, axis=0),
            np.quantile(local, 0.10, axis=0),
            np.quantile(local, 0.50, axis=0),
            np.quantile(local, 0.90, axis=0),
            slope,
        )
        aggregates.append(np.concatenate(parts))
        if names is None:
            names = tuple(
                f"{stat}_f{feature:04d}"
                for stat in ("mean", "std", "q10", "q50", "q90", "slope")
                for feature in range(local.shape[1])
            )
    assert names is not None
    return np.asarray(aggregates, dtype=np.float32), unique, names


def extract_bundle(
    *,
    data_root: Path,
    manifest_path: Path,
    output_path: Path,
    temporal_stride: int = 5,
    limit: int | None = None,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    samples = build_sample_index(data_root, manifest_path)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No readable samples found")
    session_map = {str(row["session_id"]): row for row in manifest["sessions"]}
    rows = []
    names: tuple[str, ...] | None = None
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        vector, current_names = extract_morphology(load_array(sample), temporal_stride=temporal_stride)
        if names is None:
            names = current_names
        elif names != current_names:
            raise AssertionError("Morphology schema changed within extraction")
        rows.append(vector)
        if (index + 1) % 250 == 0 or index + 1 == len(samples):
            print(f"[MORPHOLOGY] {index + 1}/{len(samples)} windows in {time.perf_counter() - started:.1f}s", flush=True)
    assert names is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(rows, dtype=np.float32),
        feature_names=np.asarray(names),
        labels=np.asarray([sample.class_id for sample in samples], dtype=np.int64),
        sessions=np.asarray([sample.session_id for sample in samples]),
        rel_paths=np.asarray([sample.rel_path for sample in samples]),
        window_ids=np.asarray([sample.window_id for sample in samples], dtype=np.int32),
        partitions=np.asarray([sample.partition for sample in samples]),
        eras=np.asarray([session_map[sample.session_id]["era"] for sample in samples]),
        date_tokens=np.asarray([session_map[sample.session_id]["date_token"] for sample in samples]),
        source_tokens=np.asarray([session_map[sample.session_id]["source_token"] for sample in samples]),
    )
    metadata = {
        "schema_version": 1,
        "protocol": "complete-data interpretable window morphology v3",
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
        "temporal_stride": temporal_stride,
        "window_count": len(samples),
        "session_count": len({sample.session_id for sample in samples}),
        "feature_count": len(names),
        "limit": limit,
        "elapsed_seconds": time.perf_counter() - started,
        "bundle_sha256": _file_sha256(output_path),
    }
    metadata["payload_sha256"] = canonical_json_hash(metadata)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-stride", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            extract_bundle(
                data_root=args.data_root,
                manifest_path=args.manifest,
                output_path=args.output,
                temporal_stride=args.temporal_stride,
                limit=args.limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
