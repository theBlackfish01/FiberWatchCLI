"""Interpretable morphology-before-name experiments for PHI-OTDR v3."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES, canonical_json_hash
from .distributional_enrollment_v3 import (
    _classification_metrics,
    _detection_metrics,
    _seed,
    select_support,
)
from .era_contract import verify_acquisition_manifest
from .spatial_experiment import _metrics, _model_grid, _select_temperature, _temperature


ATTRIBUTE_NAMES = (
    "log_delta_rms",
    "log_delta_mean_abs",
    "log_delta_p95_abs",
    "log_difference_scale",
    "log_burstiness",
    "periodicity",
    "spectral_centroid",
    "very_low_frequency_share",
    "low_frequency_share",
    "high_frequency_share",
    "temporal_modulation_cv",
    "temporal_peak_ratio",
    "temporal_trend_normalized",
    "global_channel_coherence",
    "adjacent_channel_coherence",
    "spatial_width",
    "spatial_entropy",
    "spatial_peak_share",
    "activity_confidence",
    "absolute_center",
)
POSITION_ATTRIBUTES = {"absolute_center"}
PRIMARY_ENROLLMENT_VIEWS = ("morphology_only", "morphology_plus_position")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_window_attributes(
    features: np.ndarray, feature_names: Iterable[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    x = np.asarray(features, dtype=np.float64)
    names = [str(name) for name in feature_names]
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("Morphology feature matrix and names disagree")
    index = {name: i for i, name in enumerate(names)}

    def column(name: str) -> np.ndarray:
        return x[:, index[name]]

    profile_columns = [index[f"difference_profile_ch{channel:02d}"] for channel in range(12)]
    profile = np.maximum(x[:, profile_columns], 0.0)
    profile /= np.maximum(np.sum(profile, axis=1, keepdims=True), 1e-12)
    center = column("difference_center")
    channel_axis = np.arange(12, dtype=np.float64)[None, :]
    width = np.sqrt(np.sum(profile * (channel_axis - center[:, None]) ** 2, axis=1))
    entropy = -np.sum(profile * np.log(profile + 1e-12), axis=1) / np.log(12.0)
    block_mean = np.maximum(np.abs(column("dynamics_temporal_block_mean")), 1e-9)
    attributes = np.column_stack(
        (
            np.log1p(np.maximum(column("dynamics_delta_rms"), 0.0)),
            np.log1p(np.maximum(column("dynamics_delta_mean_abs"), 0.0)),
            np.log1p(np.maximum(column("dynamics_delta_p95_abs"), 0.0)),
            column("difference_log_scale"),
            np.log1p(np.maximum(column("dynamics_delta_burstiness"), 0.0)),
            1.0 - np.clip(column("dynamics_spectrum_entropy"), 0.0, 1.0),
            column("dynamics_spectrum_centroid"),
            column("dynamics_spectrum_band0"),
            column("dynamics_spectrum_band1"),
            column("dynamics_spectrum_band2") + column("dynamics_spectrum_band3"),
            column("dynamics_temporal_block_std") / block_mean,
            column("dynamics_temporal_block_max") / block_mean,
            column("dynamics_temporal_block_slope") / block_mean,
            column("dynamics_correlation_mean"),
            column("dynamics_correlation_neighbor_mean"),
            width,
            entropy,
            np.max(profile, axis=1),
            column("difference_confidence"),
            center,
        )
    ).astype(np.float32)
    if attributes.shape[1] != len(ATTRIBUTE_NAMES) or not np.isfinite(attributes).all():
        raise ValueError("Derived morphology attributes are invalid")
    return attributes, ATTRIBUTE_NAMES


def aggregate_attribute_sessions(
    window_attributes: np.ndarray,
    sessions: np.ndarray,
    window_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x = np.asarray(window_attributes, dtype=np.float64)
    session_array = np.asarray(sessions).astype(str)
    ids = np.asarray(window_ids, dtype=np.float64)
    unique = np.asarray(sorted(set(session_array.tolist())))
    rows = []
    statistic_names = ("mean", "standard_deviation", "q10", "median", "q90", "ordered_slope")
    for session in unique:
        mask = session_array == session
        local = x[mask][np.argsort(ids[mask])]
        axis = np.linspace(-1.0, 1.0, len(local))
        slope = (
            np.sum((local - local.mean(axis=0)) * axis[:, None], axis=0) / np.sum(axis**2)
            if len(local) > 1
            else np.zeros(local.shape[1])
        )
        rows.append(
            np.concatenate(
                (
                    np.mean(local, axis=0),
                    np.std(local, axis=0),
                    np.quantile(local, 0.10, axis=0),
                    np.quantile(local, 0.50, axis=0),
                    np.quantile(local, 0.90, axis=0),
                    slope,
                )
            )
        )
    names = tuple(
        f"{statistic}__{attribute}"
        for statistic in statistic_names
        for attribute in ATTRIBUTE_NAMES
    )
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(unique), len(names)) or not np.isfinite(result).all():
        raise ValueError("Session attribute aggregation failed")
    return result, unique, names


def _view_indices(names: Iterable[str], view: str) -> np.ndarray:
    names = [str(name) for name in names]
    if view == "morphology_only":
        return np.asarray([i for i, name in enumerate(names) if name.split("__", 1)[1] not in POSITION_ATTRIBUTES])
    if view == "position_only":
        return np.asarray([i for i, name in enumerate(names) if name.split("__", 1)[1] in POSITION_ATTRIBUTES])
    if view == "morphology_plus_position":
        return np.arange(len(names), dtype=np.int64)
    raise ValueError(f"Unknown attribute view: {view}")


def _risk_coverage(labels: np.ndarray, probs: np.ndarray) -> dict[str, object]:
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    order = np.argsort(-confidence, kind="stable")
    correct = prediction[order] == labels[order]
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(correct) + 1)
    points = {}
    for coverage in (0.50, 0.80, 0.90, 1.00):
        count = max(1, int(np.ceil(coverage * len(labels))))
        selected = order[:count]
        points[f"coverage_{coverage:.2f}"] = {
            "sessions": count,
            "accuracy": float(np.mean(prediction[selected] == labels[selected])),
            "macro_f1": float(
                f1_score(
                    labels[selected],
                    prediction[selected],
                    labels=np.arange(len(CLASS_NAMES)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "confidence_threshold": float(np.min(confidence[selected])),
        }
    return {"aurc": float(np.mean(risk)), "points": points}


def _fit_mapper(
    features: np.ndarray,
    labels: np.ndarray,
    partitions: np.ndarray,
) -> tuple[object, dict[str, object], float, list[dict[str, object]]]:
    train = partitions == "source_train"
    validation = partitions == "source_validation"
    calibration = partitions == "source_calibration"
    selected = None
    trace = []
    for params, model in _model_grid("logistic"):
        model.fit(features[train], labels[train])
        probs = model.predict_proba(features[validation])
        score = f1_score(labels[validation], np.argmax(probs, axis=1), average="macro", zero_division=0)
        trace.append({"params": params, "source_validation_macro_f1": float(score)})
        candidate = (float(score), -len(json.dumps(params)), json.dumps(params, sort_keys=True))
        if selected is None or candidate > selected[0]:
            selected = (candidate, params)
    assert selected is not None
    selected_params = dict(selected[1])
    final_model = next(model for params, model in _model_grid("logistic") if params == selected_params)
    final_model.fit(features[train | validation], labels[train | validation])
    temperature = _select_temperature(labels[calibration], final_model.predict_proba(features[calibration]))
    return final_model, selected_params, temperature, trace


def _retrieval(
    features: np.ndarray,
    labels: np.ndarray,
    partitions: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    gallery = (partitions == "source_train") | (partitions == "source_validation")
    query = partitions == "target_query"
    scaler = StandardScaler().fit(features[gallery])
    gallery_x = scaler.transform(features[gallery])
    query_x = scaler.transform(features[query])
    neighbors = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(gallery_x)
    distances, indices = neighbors.kneighbors(query_x)
    gallery_labels = labels[gallery]
    prediction = np.asarray(
        [int(np.argmax(np.bincount(gallery_labels[row], minlength=len(CLASS_NAMES)))) for row in indices]
    )
    top3_hit = np.asarray([labels[query][i] in gallery_labels[row] for i, row in enumerate(indices)])
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[prediction]
    metrics = _metrics(labels[query], one_hot)
    metrics["top3_contains_true_class"] = float(np.mean(top3_hit))
    metrics["mean_nearest_distance"] = float(np.mean(distances[:, 0]))
    return metrics, query, prediction


def _prototype_scores(
    query: np.ndarray,
    base: np.ndarray,
    base_labels: np.ndarray,
    support: np.ndarray | None,
    holdout: int,
) -> np.ndarray:
    scores = np.full((len(query), len(CLASS_NAMES)), np.inf, dtype=np.float64)
    for class_id in range(len(CLASS_NAMES)):
        if class_id == holdout:
            if support is not None:
                prototype = np.mean(support, axis=0)
            else:
                continue
        else:
            prototype = np.mean(base[base_labels == class_id], axis=0)
        scores[:, class_id] = np.linalg.norm(query - prototype, axis=1)
    return scores


def _write_csv(path: Path, rows: list[dict[str, object]], *, compressed: bool = False) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _enrollment(
    view_features: dict[str, np.ndarray],
    sessions: np.ndarray,
    labels: np.ndarray,
    manifests: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    episodes = []
    support_rows = []
    prediction_rows = []
    detections = []
    for manifest in manifests:
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
        partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
        query_mask = partitions == "target_query"
        query_sessions = sessions[query_mask]
        for view in PRIMARY_ENROLLMENT_VIEWS:
            raw = view_features[view]
            method = f"attribute_prototype_{view}"
            for holdout in range(len(CLASS_NAMES)):
                base_mask = (partitions == "source_train") & (labels != holdout)
                support_mask = (partitions == "target_support") & (labels == holdout)
                calibration_mask = (partitions == "target_calibration") & (labels != holdout)
                scaler = StandardScaler().fit(raw[base_mask])
                descriptor = scaler.transform(raw)
                base = descriptor[base_mask]
                base_labels = labels[base_mask]
                candidates = descriptor[support_mask]
                candidate_sessions = sessions[support_mask]
                calibration_scores = _prototype_scores(
                    descriptor[calibration_mask], base, base_labels, None, holdout
                )
                query_scores_pre = _prototype_scores(descriptor[query_mask], base, base_labels, None, holdout)
                seen = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
                detection = _detection_metrics(
                    np.min(calibration_scores[:, seen], axis=1),
                    np.min(query_scores_pre[:, seen], axis=1),
                    labels[query_mask],
                    holdout,
                    0.95,
                )
                detections.append(
                    {"direction": direction, "heldout_class": CLASS_NAMES[holdout], "method": method, **detection}
                )
                for shot in (1, 3, 5):
                    selector_draws = [("random", draw) for draw in range(30)]
                    selector_draws.extend((selector, 0) for selector in ("medoid", "k_center", "pool_coverage"))
                    for selector, draw in selector_draws:
                        support_seed = _seed(20260808, direction, holdout, shot, selector, draw)
                        local = select_support(candidates, selector=selector, shot=shot, seed=support_seed)
                        selected_sessions = candidate_sessions[local]
                        scores = _prototype_scores(
                            descriptor[query_mask], base, base_labels, candidates[local], holdout
                        )
                        prediction = np.argmin(scores, axis=1)
                        metrics = _classification_metrics(labels[query_mask], prediction, holdout)
                        episode_id = canonical_json_hash(
                            {
                                "direction": direction,
                                "holdout": holdout,
                                "method": method,
                                "shot": shot,
                                "selector": selector,
                                "draw": draw,
                                "support": selected_sessions.tolist(),
                            }
                        )
                        episodes.append(
                            {
                                "episode_id": episode_id,
                                "direction": direction,
                                "heldout_class": CLASS_NAMES[holdout],
                                "method": method,
                                "shot": shot,
                                "selector": selector,
                                "draw": draw,
                                "support_sessions": selected_sessions.tolist(),
                                **metrics,
                            }
                        )
                        for rank, session in enumerate(selected_sessions):
                            support_rows.append(
                                {
                                    "direction": direction,
                                    "heldout_class": CLASS_NAMES[holdout],
                                    "method": method,
                                    "shot": shot,
                                    "selector": selector,
                                    "draw": draw,
                                    "seed": support_seed,
                                    "rank": rank,
                                    "session_id": session,
                                }
                            )
                        for index, session in enumerate(query_sessions):
                            prediction_rows.append(
                                {
                                    "episode_id": episode_id,
                                    "session_id": session,
                                    "true_label": int(labels[query_mask][index]),
                                    "predicted_label": int(prediction[index]),
                                    "true_class": CLASS_NAMES[int(labels[query_mask][index])],
                                    "predicted_class": CLASS_NAMES[int(prediction[index])],
                                    "minimum_distance": float(np.min(scores[index])),
                                }
                            )
    return episodes, support_rows, prediction_rows, detections


def run(
    *,
    bundle_path: Path,
    bundle_metadata_path: Path,
    config_path: Path,
    config_hash_path: Path,
    manifest_paths: Iterable[Path],
    spatial_results_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config_hash = config_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(config) != expected_config_hash:
        raise ValueError("Morphology attribute config hash mismatch")
    if _sha256(bundle_path) != config["input_bundle_sha256"]:
        raise ValueError("Morphology input bundle hash mismatch")
    metadata = json.loads(bundle_metadata_path.read_text(encoding="utf-8"))
    stored_metadata_hash = str(metadata.pop("payload_sha256"))
    if stored_metadata_hash != canonical_json_hash(metadata):
        raise ValueError("Morphology metadata hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}

    attributes, attribute_names = derive_window_attributes(bundle["features"], bundle["feature_names"])
    session_attributes, sessions, session_attribute_names = aggregate_attribute_sessions(
        attributes, bundle["sessions"], bundle["window_ids"]
    )
    first = {session: index for index, session in enumerate(bundle["sessions"].astype(str))}
    labels = np.asarray([bundle["labels"][first[session]] for session in sessions], dtype=np.int64)
    views = {
        view: session_attributes[:, _view_indices(session_attribute_names, view)]
        for view in config["views"]
    }
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    manifest_hashes = {}
    for manifest in manifests:
        verified = verify_acquisition_manifest(
            manifest, expected_dataset_fingerprint=str(config["dataset_fingerprint_sha256"])
        )
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        manifest_hashes[direction] = verified["manifest_sha256"]

    spatial = json.loads(spatial_results_path.read_text(encoding="utf-8"))
    spatial_hash = str(spatial.pop("payload_sha256"))
    if spatial_hash != canonical_json_hash(spatial):
        raise ValueError("Spatial control hash mismatch")
    spatial["payload_sha256"] = spatial_hash

    classification_results = []
    classification_predictions = []
    retrieval_results = []
    retrieval_predictions = []
    for manifest in manifests:
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
        partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
        query = partitions == "target_query"
        for view, features in views.items():
            model, selected_params, temperature, selection_trace = _fit_mapper(features, labels, partitions)
            validation = partitions == "source_validation"
            validation_probs = _temperature(model.predict_proba(features[validation]), temperature)
            query_probs = _temperature(model.predict_proba(features[query]), temperature)
            classification_results.append(
                {
                    "direction": direction,
                    "view": view,
                    "session_feature_count": int(features.shape[1]),
                    "selected_params": selected_params,
                    "selection_trace": selection_trace,
                    "temperature": temperature,
                    "source_validation": _metrics(labels[validation], validation_probs),
                    "target_query_retrospective": _metrics(labels[query], query_probs),
                    "target_risk_coverage": _risk_coverage(labels[query], query_probs),
                    "selection_used_target_query": False,
                }
            )
            for index, session in enumerate(sessions[query]):
                row = {
                    "direction": direction,
                    "view": view,
                    "session_id": session,
                    "true_label": int(labels[query][index]),
                    "predicted_label": int(np.argmax(query_probs[index])),
                    "true_class": CLASS_NAMES[int(labels[query][index])],
                    "predicted_class": CLASS_NAMES[int(np.argmax(query_probs[index]))],
                }
                for class_id, class_name in enumerate(CLASS_NAMES):
                    row[f"prob_{class_name}"] = float(query_probs[index, class_id])
                classification_predictions.append(row)

            retrieval_metrics, retrieval_query, retrieval_prediction = _retrieval(
                features, labels, partitions
            )
            retrieval_results.append(
                {"direction": direction, "view": view, "neighbors": 3, "metrics": retrieval_metrics}
            )
            for index, session in enumerate(sessions[retrieval_query]):
                retrieval_predictions.append(
                    {
                        "direction": direction,
                        "view": view,
                        "session_id": session,
                        "true_label": int(labels[retrieval_query][index]),
                        "predicted_label": int(retrieval_prediction[index]),
                        "true_class": CLASS_NAMES[int(labels[retrieval_query][index])],
                        "predicted_class": CLASS_NAMES[int(retrieval_prediction[index])],
                    }
                )

    episodes, support_rows, enrollment_predictions, detections = _enrollment(
        views, sessions, labels, manifests
    )
    random_summary = []
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        if row["selector"] == "random":
            grouped[(str(row["direction"]), str(row["method"]), int(row["shot"]))].append(row)
    for (direction, method, shot), rows in sorted(grouped.items()):
        random_summary.append(
            {
                "direction": direction,
                "method": method,
                "shot": shot,
                "heldout_classes": len({row["heldout_class"] for row in rows}),
                "episodes": len(rows),
                "enrollment_h_mean": float(np.mean([float(row["enrollment_h"]) for row in rows])),
                "enrollment_h_worst": float(np.min([float(row["enrollment_h"]) for row in rows])),
                "macro_f1_mean": float(np.mean([float(row["session_macro_f1"]) for row in rows])),
                "worst_class_recall_mean": float(np.mean([float(row["worst_class_recall"]) for row in rows])),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "window_attributes.npz"
    session_path = output_dir / "session_attributes.npz"
    np.savez_compressed(
        window_path,
        attributes=attributes,
        attribute_names=np.asarray(attribute_names),
        sessions=bundle["sessions"],
        window_ids=bundle["window_ids"],
        rel_paths=bundle["rel_paths"],
    )
    np.savez_compressed(
        session_path,
        attributes=session_attributes,
        attribute_names=np.asarray(session_attribute_names),
        sessions=sessions,
        labels=labels,
    )
    classification_path = output_dir / "classification_predictions.csv"
    retrieval_path = output_dir / "retrieval_predictions.csv"
    support_path = output_dir / "support_draws.csv"
    enrollment_prediction_path = output_dir / "enrollment_query_predictions.csv.gz"
    _write_csv(classification_path, classification_predictions)
    _write_csv(retrieval_path, retrieval_predictions)
    _write_csv(support_path, support_rows)
    _write_csv(enrollment_prediction_path, enrollment_predictions, compressed=True)

    controls = [
        row
        for row in spatial["results"]
        if (
            (row["view"], row["estimator"], row["ablation"], row["model"])
            in {
                ("registered_position", "temporal_difference_energy", "dynamics", "logistic"),
                ("invariant", "none", "fused", "logistic"),
            }
        )
    ]
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 morphology-before-name attributes",
        "evidence_status": "retrospective development; not independent confirmation",
        "config_sha256": expected_config_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "input_hashes": {
            "bundle_sha256": _sha256(bundle_path),
            "bundle_metadata_payload_sha256": stored_metadata_hash,
            "spatial_control_payload_sha256": spatial_hash,
            "manifest_sha256": manifest_hashes,
        },
        "attribute_definitions": config["attribute_families"],
        "attribute_semantics": config["semantics_policy"],
        "window_count": int(len(attributes)),
        "session_count": int(len(sessions)),
        "window_attribute_count": len(attribute_names),
        "session_attribute_count": len(session_attribute_names),
        "classification_results": classification_results,
        "retrieval_results": retrieval_results,
        "enrollment": {
            "episode_count": len(episodes),
            "episodes": episodes,
            "random_summary": random_summary,
            "pre_enrollment_detection": detections,
        },
        "spatial_controls": controls,
        "output_hashes": {
            "window_attributes_sha256": _sha256(window_path),
            "session_attributes_sha256": _sha256(session_path),
            "classification_predictions_sha256": _sha256(classification_path),
            "retrieval_predictions_sha256": _sha256(retrieval_path),
            "support_draws_sha256": _sha256(support_path),
            "enrollment_query_predictions_sha256": _sha256(enrollment_prediction_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "selection_used_target_query": False,
        "limitations": [
            "Attribute names are physics-readable transformations, not independently annotated morphology ground truth.",
            "The event-name mapping remains supervised by the six historical class labels.",
            "All target-query comparisons are retrospective.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "morphology_attributes_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--spatial-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        bundle_path=args.bundle,
        bundle_metadata_path=args.bundle_metadata,
        config_path=args.config,
        config_hash_path=args.config_hash,
        manifest_paths=args.manifest,
        spatial_results_path=args.spatial_results,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "classification_results": len(result["classification_results"]),
                "enrollment_episodes": result["enrollment"]["episode_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "payload_sha256": result["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
