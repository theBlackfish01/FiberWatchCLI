"""Wavelet-energy and spatial effective-rank controls for PHI-OTDR v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_contract import CLASS_NAMES, canonical_json_hash
from .dataset import build_sample_index, load_array
from .era_contract import verify_acquisition_manifest
from .morphology_attributes_v3 import _fit_mapper, _risk_coverage
from .morphology_features import aggregate_sessions
from .spatial_experiment import _metrics, _temperature


WINDOW_FEATURE_NAMES = (
    "haar_detail_share_level1",
    "haar_detail_share_level2",
    "haar_detail_share_level3",
    "haar_detail_share_level4",
    "haar_approximation_share_level4",
    "haar_scale_entropy",
    "spatial_covariance_effective_rank_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_wavelet_rank(
    array: np.ndarray,
    *,
    temporal_stride: int = 5,
    levels: int = 4,
) -> np.ndarray:
    """Return orthonormal Haar scale shares and channel-covariance effective rank."""

    if array.shape != (10000, 12):
        raise ValueError(f"Expected (10000, 12), received {array.shape}")
    if temporal_stride < 1 or levels < 1:
        raise ValueError("Stride and levels must be positive")
    signal = np.diff(array[::temporal_stride].astype(np.float64, copy=False), axis=0)
    approximation = signal
    energies: list[float] = []
    scale = np.sqrt(2.0)
    for _ in range(levels):
        usable = approximation[: (len(approximation) // 2) * 2]
        if len(usable) < 2:
            raise ValueError("Insufficient temporal samples for requested Haar levels")
        even = usable[0::2]
        odd = usable[1::2]
        detail = (even - odd) / scale
        approximation = (even + odd) / scale
        energies.append(float(np.sum(detail * detail)))
    energies.append(float(np.sum(approximation * approximation)))
    energy = np.asarray(energies, dtype=np.float64)
    total = float(np.sum(energy))
    shares = energy / total if total > 1e-20 else np.zeros_like(energy)
    positive = shares[shares > 0]
    entropy = (
        float(-np.sum(positive * np.log(positive)) / np.log(len(shares)))
        if len(positive) > 0 and len(shares) > 1
        else 0.0
    )

    covariance = np.cov(signal, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    eigen_total = float(np.sum(eigenvalues))
    if eigen_total > 1e-20:
        probabilities = eigenvalues / eigen_total
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))) / 12.0)
    else:
        effective_rank = 0.0
    result = np.asarray([*shares.tolist(), entropy, effective_rank], dtype=np.float32)
    if len(result) != len(WINDOW_FEATURE_NAMES) or not np.isfinite(result).all():
        raise AssertionError("Invalid wavelet/rank descriptor")
    return result


def _session_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{stat}__{feature}"
        for stat in ("mean", "standard_deviation", "q10", "median", "q90", "ordered_slope")
        for feature in WINDOW_FEATURE_NAMES
    )


def _extract_cache(
    *,
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    temporal_stride: int,
    levels: int,
) -> dict[str, Any]:
    samples = build_sample_index(data_root, manifest_path)
    started = time.perf_counter()
    rows = []
    for index, sample in enumerate(samples):
        rows.append(
            extract_wavelet_rank(
                load_array(sample), temporal_stride=temporal_stride, levels=levels
            )
        )
        if (index + 1) % 500 == 0 or index + 1 == len(samples):
            print(
                f"[EXTENDED MORPHOLOGY] {index + 1}/{len(samples)} windows "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    window_features = np.asarray(rows, dtype=np.float32)
    sessions = np.asarray([sample.session_id for sample in samples])
    labels = np.asarray([sample.class_id for sample in samples], dtype=np.int64)
    window_ids = np.asarray([sample.window_id for sample in samples], dtype=np.int32)
    session_features, unique_sessions, _ = aggregate_sessions(window_features, sessions, window_ids)
    label_map = {sample.session_id: sample.class_id for sample in samples}
    session_labels = np.asarray([label_map[session] for session in unique_sessions], dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "wavelet_rank_windows.npz"
    session_path = output_dir / "wavelet_rank_sessions.npz"
    np.savez_compressed(
        window_path,
        features=window_features,
        feature_names=np.asarray(WINDOW_FEATURE_NAMES),
        sessions=sessions,
        labels=labels,
        window_ids=window_ids,
    )
    np.savez_compressed(
        session_path,
        features=session_features,
        feature_names=np.asarray(_session_feature_names()),
        sessions=unique_sessions,
        labels=session_labels,
    )
    return {
        "window_path": window_path,
        "session_path": session_path,
        "window_count": len(samples),
        "session_count": len(unique_sessions),
        "elapsed_seconds": time.perf_counter() - started,
        "window_sha256": _sha256(window_path),
        "session_sha256": _sha256(session_path),
    }


def _partitions(manifest: dict[str, Any], sessions: Sequence[str]) -> np.ndarray:
    mapping = {str(row["session_id"]): str(row["partition"]) for row in manifest["sessions"]}
    return np.asarray([mapping[str(session)] for session in sessions])


def run(
    *,
    phi_root: Path,
    data_root: Path,
    config_path: Path,
    config_hash_path: Path,
    manifest_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = config_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(config) != expected_hash:
        raise ValueError("Extended morphology config hash mismatch")
    existing_path = phi_root / "experiments/phi_research_v3/morphology_attributes/session_attributes.npz"
    if _sha256(existing_path) != config["input_session_attributes_sha256"]:
        raise ValueError("Existing morphology attribute bundle hash mismatch")
    manifests = {}
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        verify_acquisition_manifest(
            manifest, expected_dataset_fingerprint=config["dataset_fingerprint_sha256"]
        )
        direction = manifest["direction"]
        manifests[f"{direction['source']}_to_{direction['target']}"] = manifest

    cache = _extract_cache(
        data_root=data_root,
        manifest_path=manifest_paths[0],
        output_dir=output_dir,
        temporal_stride=int(config["temporal_stride"]),
        levels=int(config["wavelet"]["levels"]),
    )
    with np.load(cache["session_path"], allow_pickle=False) as bundle:
        extended = np.asarray(bundle["features"], dtype=np.float32)
        sessions = bundle["sessions"].astype(str)
        labels = np.asarray(bundle["labels"], dtype=np.int64)
    with np.load(existing_path, allow_pickle=False) as bundle:
        existing_sessions = bundle["sessions"].astype(str)
        existing_labels = np.asarray(bundle["labels"], dtype=np.int64)
        names = bundle["attribute_names"].astype(str)
        morphology_mask = np.asarray([not name.endswith("__absolute_center") for name in names])
        existing = np.asarray(bundle["attributes"][:, morphology_mask], dtype=np.float32)
    existing_map = {session: index for index, session in enumerate(existing_sessions)}
    reorder = np.asarray([existing_map[session] for session in sessions], dtype=np.int64)
    if not np.array_equal(labels, existing_labels[reorder]):
        raise AssertionError("Session labels disagree across morphology bundles")
    views = {
        "wavelet_rank_only": extended,
        "morphology_plus_wavelet_rank": np.concatenate([existing[reorder], extended], axis=1),
    }

    results = []
    predictions = []
    for direction, manifest in sorted(manifests.items()):
        partitions = _partitions(manifest, sessions)
        validation = partitions == "source_validation"
        query = partitions == "target_query"
        for view, features in views.items():
            model, selected_params, temperature, trace = _fit_mapper(features, labels, partitions)
            validation_probs = _temperature(model.predict_proba(features[validation]), temperature)
            query_probs = _temperature(model.predict_proba(features[query]), temperature)
            query_prediction = np.argmax(query_probs, axis=1)
            results.append(
                {
                    "direction": direction,
                    "view": view,
                    "session_feature_count": int(features.shape[1]),
                    "selected_params": selected_params,
                    "selection_trace": trace,
                    "selection_used_target_query": False,
                    "temperature": temperature,
                    "source_validation": _metrics(labels[validation], validation_probs),
                    "target_query_retrospective": _metrics(labels[query], query_probs),
                    "target_risk_coverage": _risk_coverage(labels[query], query_probs),
                }
            )
            for index, session_id in enumerate(sessions[query]):
                row = {
                    "direction": direction,
                    "view": view,
                    "session_id": str(session_id),
                    "true_label": int(labels[query][index]),
                    "predicted_label": int(query_prediction[index]),
                    "true_class": CLASS_NAMES[int(labels[query][index])],
                    "predicted_class": CLASS_NAMES[int(query_prediction[index])],
                }
                row.update(
                    {f"prob_{name}": float(query_probs[index, class_id]) for class_id, name in enumerate(CLASS_NAMES)}
                )
                predictions.append(row)
    prediction_path = output_dir / "classification_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 wavelet and spatial effective-rank missing-control analysis",
        "evidence_status": config["evidence_status"],
        "config_sha256": expected_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "selection_used_target_query": False,
        "cache": {
            key: value.as_posix() if isinstance(value, Path) else value for key, value in cache.items()
        },
        "existing_session_attributes_sha256": _sha256(existing_path),
        "prediction_sha256": _sha256(prediction_path),
        "window_feature_names": list(WINDOW_FEATURE_NAMES),
        "results": results,
        "limitations": config["limitations"],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "extended_morphology_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        phi_root=args.phi_root.resolve(),
        data_root=args.data_root.resolve(),
        config_path=args.config.resolve(),
        config_hash_path=args.config_hash.resolve(),
        manifest_paths=[path.resolve() for path in args.manifest],
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "payload_sha256": result["payload_sha256"],
                "results": len(result["results"]),
                "window_count": result["cache"]["window_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
