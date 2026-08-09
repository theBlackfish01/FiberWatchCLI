"""Support-identical robust distributional PHI-OTDR enrollment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, recall_score
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES, canonical_json_hash
from .evaluation_ladder_v1 import classification_metrics
from .morphology_attributes_v3 import _view_indices
from .shift_protocol_v1 import (
    finalize_payload,
    load_locked_config,
    process_memory_snapshot,
    sha256_file,
    write_csv,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def deterministic_support_draw(
    candidates: np.ndarray, *, shot: int, seed: int
) -> np.ndarray:
    candidates = np.asarray(sorted(candidates.astype(str).tolist()))
    if shot > len(candidates):
        raise ValueError(f"Requested {shot} support sessions from {len(candidates)} candidates")
    rng = np.random.default_rng(seed)
    return candidates[np.sort(rng.choice(len(candidates), size=shot, replace=False))]


def _session_metadata(
    window: Mapping[str, np.ndarray],
    session: Mapping[str, np.ndarray],
    morphology_bundle: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sessions = session["sessions"].astype(str)
    labels = session["labels"].astype(np.int64)
    bundle_sessions = morphology_bundle["sessions"].astype(str)
    dates, eras = [], []
    for item in sessions:
        indices = np.flatnonzero(bundle_sessions == item)
        dates.append(str(np.unique(morphology_bundle["date_tokens"][indices].astype(str)).item()))
        eras.append(str(np.unique(morphology_bundle["eras"][indices].astype(str)).item()))
    if set(window["sessions"].astype(str)) != set(sessions.tolist()):
        raise ValueError("Window and session attribute caches contain different sessions")
    return sessions, labels, np.asarray(dates), np.asarray(eras)


def _partition_vector(manifest: Mapping[str, object], sessions: np.ndarray) -> np.ndarray:
    rows = {str(row["session_id"]): str(row["partition"]) for row in manifest["sessions"]}
    if set(rows) != set(sessions.tolist()):
        raise ValueError("Acquisition manifest does not assign every session exactly once")
    return np.asarray([rows[session] for session in sessions])


def _window_descriptor(
    values: np.ndarray,
    window_sessions: np.ndarray,
    sessions: np.ndarray,
    fit_window: np.ndarray,
    *,
    projection_count: int,
    quantile_count: int,
    quantile_range: tuple[float, float],
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(values[fit_window])
    scaled = scaler.transform(values)
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(projection_count, scaled.shape[1]))
    projections /= np.maximum(np.linalg.norm(projections, axis=1, keepdims=True), 1e-12)
    projected = scaled @ projections.T
    quantiles = np.linspace(quantile_range[0], quantile_range[1], quantile_count)
    descriptors = []
    for session in sessions:
        local = projected[window_sessions == session]
        descriptors.append(np.quantile(local, quantiles, axis=0).T.reshape(-1))
    return np.asarray(descriptors, dtype=np.float64)


def _aggregate_descriptor(
    values: np.ndarray,
    fit: np.ndarray,
    components: int,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(values[fit])
    scaled = scaler.transform(values)
    count = min(components, int(np.sum(fit)) - 1, scaled.shape[1])
    pca = PCA(n_components=count, whiten=True, svd_solver="randomized", random_state=seed % (2**32))
    pca.fit(scaled[fit])
    return pca.transform(scaled)


def _standardize_descriptor(values: np.ndarray, fit: np.ndarray) -> np.ndarray:
    return StandardScaler().fit(values[fit]).transform(values)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _consensus_weights(gallery: np.ndarray) -> np.ndarray:
    if len(gallery) == 1:
        return np.ones(1)
    distances = cdist(gallery, gallery)
    typicality = np.median(distances, axis=1)
    positive = typicality[typicality > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    weights = 1.0 / (1.0 + typicality / max(scale, 1e-12))
    return weights / np.sum(weights)


def _base_scores(
    query: np.ndarray,
    gallery: np.ndarray,
    labels: np.ndarray,
    holdout: int,
    *,
    mode: str,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full((len(query), len(CLASS_NAMES)), np.inf)
    disagreement = np.full_like(scores, np.inf)
    for class_id in range(len(CLASS_NAMES)):
        if class_id == holdout:
            continue
        local_gallery = gallery[labels == class_id]
        if not len(local_gallery):
            continue
        distances = cdist(query, local_gallery)
        if mode == "gallery":
            count = min(neighbors, distances.shape[1])
            nearest = np.partition(distances, count - 1, axis=1)[:, :count]
            scores[:, class_id] = np.mean(nearest, axis=1)
            disagreement[:, class_id] = np.std(nearest, axis=1)
        elif mode == "median":
            centre = np.median(local_gallery, axis=0, keepdims=True)
            scores[:, class_id] = cdist(query, centre)[:, 0]
            disagreement[:, class_id] = np.median(distances, axis=1)
        elif mode == "weighted":
            weights = _consensus_weights(local_gallery)
            centre = np.sum(local_gallery * weights[:, None], axis=0, keepdims=True)
            scores[:, class_id] = cdist(query, centre)[:, 0]
            disagreement[:, class_id] = np.sum(distances * weights[None, :], axis=1)
        else:
            raise ValueError(f"Unknown score mode: {mode}")
    return scores, disagreement


def _add_support_scores(
    scores: np.ndarray,
    disagreement: np.ndarray,
    query: np.ndarray,
    support: np.ndarray,
    holdout: int,
    *,
    mode: str,
    neighbors: int,
) -> None:
    distances = cdist(query, support)
    if mode == "gallery":
        count = min(neighbors, distances.shape[1])
        nearest = np.partition(distances, count - 1, axis=1)[:, :count]
        scores[:, holdout] = np.mean(nearest, axis=1)
        disagreement[:, holdout] = np.std(nearest, axis=1)
    elif mode == "median":
        scores[:, holdout] = cdist(query, np.median(support, axis=0, keepdims=True))[:, 0]
        disagreement[:, holdout] = np.median(distances, axis=1)
    elif mode == "weighted":
        weights = _consensus_weights(support)
        centre = np.sum(support * weights[:, None], axis=0, keepdims=True)
        scores[:, holdout] = cdist(query, centre)[:, 0]
        disagreement[:, holdout] = np.sum(distances * weights[None, :], axis=1)
    else:
        raise ValueError(f"Unknown support mode: {mode}")


def _distance_probabilities(scores: np.ndarray, scale: float, temperature: float) -> np.ndarray:
    logits = -scores / max(scale * temperature, 1e-9)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def _risk_coverage_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    prediction = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    order = np.argsort(-confidence, kind="stable")
    correct = prediction[order] == labels[order]
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(correct) + 1)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    return float(np.trapezoid(risk, coverage))


def _episode_metrics(labels: np.ndarray, probabilities: np.ndarray, holdout: int) -> dict[str, object]:
    base = classification_metrics(labels, probabilities)
    prediction = np.argmax(probabilities, axis=1)
    seen = labels != holdout
    base_accuracy = float(np.mean(prediction[seen] == labels[seen]))
    enrolled_recall = float(np.mean(prediction[~seen] == holdout))
    enrollment_h = (
        2.0 * base_accuracy * enrolled_recall / (base_accuracy + enrolled_recall)
        if base_accuracy + enrolled_recall
        else 0.0
    )
    return {
        "session_macro_f1": base["macro_f1_six_classes"],
        "balanced_accuracy": base["balanced_accuracy_observed_classes"],
        "worst_class_recall": base["worst_observed_class_recall"],
        "per_class_recall": base["per_class_recall"],
        "base_class_accuracy": base_accuracy,
        "enrolled_class_recall": enrolled_recall,
        "enrollment_h": enrollment_h,
        "negative_log_likelihood": base["negative_log_likelihood"],
        "brier_score": base["brier_score"],
        "ece_10": base["ece_10"],
        "risk_coverage_auc": _risk_coverage_auc(labels, probabilities),
    }


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Prediction artifact is empty")
    fieldnames = list(rows[0])
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    morphology_bundle_path: Path,
    window_attributes_path: Path,
    session_attributes_path: Path,
    manifests: Sequence[Path],
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config, config_hash = load_locked_config(config_path, config_hash_path)
    if sha256_file(window_attributes_path) != config["input_window_attributes_sha256"]:
        raise ValueError("Window-attribute hash mismatch")
    if sha256_file(session_attributes_path) != config["input_session_attributes_sha256"]:
        raise ValueError("Session-attribute hash mismatch")
    window = _load_npz(window_attributes_path)
    session = _load_npz(session_attributes_path)
    bundle = _load_npz(morphology_bundle_path)
    sessions, labels, dates, eras = _session_metadata(window, session, bundle)
    window_names = window["attribute_names"].astype(str)
    window_indices = np.asarray([i for i, name in enumerate(window_names) if name != "absolute_center"])
    session_names = session["attribute_names"].astype(str)
    session_indices = _view_indices(session_names, "morphology_only")
    window_values = window["attributes"][:, window_indices].astype(np.float64)
    session_values = session["attributes"][:, session_indices].astype(np.float64)
    window_sessions = window["sessions"].astype(str)
    session_index = {item: i for i, item in enumerate(sessions)}
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    episodes = []
    support_rows = []
    prediction_rows = []
    method_specs = {
        "sliced_wasserstein_gallery": ("full", "gallery"),
        "trimmed_sliced_wasserstein_gallery": ("trimmed", "gallery"),
        "robust_class_barycenter": ("trimmed", "median"),
        "support_consensus_weighted": ("hybrid", "weighted"),
    }
    if set(method_specs) != set(config["methods"]):
        raise ValueError("Implementation/config method mismatch")
    for manifest in manifest_payloads:
        partitions = _partition_vector(manifest, sessions)
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        if direction not in config["directions"]:
            raise ValueError(f"Unexpected acquisition direction: {direction}")
        window_partition = np.asarray([partitions[session_index[item]] for item in window_sessions])
        for holdout, class_name in enumerate(CLASS_NAMES):
            fit_session = (partitions == "source_train") & (labels != holdout)
            fit_window = (window_partition == "source_train") & np.asarray(
                [labels[session_index[item]] != holdout for item in window_sessions]
            )
            descriptor_seed = _seed(config["projection_seed"], direction, holdout)
            full = _window_descriptor(
                window_values,
                window_sessions,
                sessions,
                fit_window,
                projection_count=int(config["projection_count"]),
                quantile_count=int(config["quantile_count"]),
                quantile_range=(0.05, 0.95),
                seed=descriptor_seed,
            )
            trimmed_range = tuple(float(value) for value in config["trimmed_quantile_range"])
            trimmed = _window_descriptor(
                window_values,
                window_sessions,
                sessions,
                fit_window,
                projection_count=int(config["projection_count"]),
                quantile_count=int(config["quantile_count"]),
                quantile_range=trimmed_range,
                seed=descriptor_seed,
            )
            aggregate = _aggregate_descriptor(
                session_values,
                fit_session,
                int(config["pca_components"]),
                descriptor_seed,
            )
            descriptors = {
                "full": _standardize_descriptor(full, fit_session),
                "trimmed": _standardize_descriptor(trimmed, fit_session),
            }
            descriptors["hybrid"] = np.concatenate(
                (_unit_rows(descriptors["trimmed"]), _unit_rows(aggregate)), axis=1
            )
            query_mask = partitions == "target_query"
            calibration_mask = (partitions == "target_calibration") & (labels != holdout)
            candidate_mask = (partitions == "target_support") & (labels == holdout)
            candidate_sessions = sessions[candidate_mask]
            if len(candidate_sessions) < max(config["shots"]):
                raise ValueError(f"Insufficient support pool for {direction}/{class_name}")
            if set(candidate_sessions) & set(sessions[query_mask]):
                raise AssertionError("Support/query session overlap")
            episode_draws = []
            for shot in config["shots"]:
                for draw in range(int(config["random_support_draws"])):
                    support_seed = _seed(config["support_seed"], direction, holdout, shot, draw)
                    selected = deterministic_support_draw(
                        candidate_sessions, shot=int(shot), seed=support_seed
                    )
                    episode_id = canonical_json_hash(
                        {
                            "direction": direction,
                            "heldout_class": class_name,
                            "shot": int(shot),
                            "draw": draw,
                            "support_sessions": selected.tolist(),
                        }
                    )
                    episode_draws.append((int(shot), draw, support_seed, episode_id, selected))
                    for rank, support_session in enumerate(selected):
                        support_rows.append(
                            {
                                "episode_id": episode_id,
                                "direction": direction,
                                "heldout_class": class_name,
                                "shot": int(shot),
                                "draw": draw,
                                "seed": support_seed,
                                "rank": rank,
                                "session_id": support_session,
                            }
                        )
            for method, (descriptor_name, score_mode) in method_specs.items():
                descriptor = descriptors[descriptor_name]
                base_gallery = descriptor[fit_session]
                base_labels = labels[fit_session]
                query = descriptor[query_mask]
                calibration = descriptor[calibration_mask]
                query_base, query_disagreement = _base_scores(
                    query,
                    base_gallery,
                    base_labels,
                    holdout,
                    mode=score_mode,
                    neighbors=int(config["gallery_neighbors"]),
                )
                calibration_scores, _ = _base_scores(
                    calibration,
                    base_gallery,
                    base_labels,
                    holdout,
                    mode=score_mode,
                    neighbors=int(config["gallery_neighbors"]),
                )
                finite_calibration = calibration_scores[np.isfinite(calibration_scores)]
                distance_scale = float(np.median(finite_calibration[finite_calibration > 0]))
                for shot, draw, support_seed, episode_id, selected in episode_draws:
                    selected_indices = np.asarray([session_index[item] for item in selected])
                    scores = query_base.copy()
                    disagreement = query_disagreement.copy()
                    _add_support_scores(
                        scores,
                        disagreement,
                        query,
                        descriptor[selected_indices],
                        holdout,
                        mode=score_mode,
                        neighbors=int(config["gallery_neighbors"]),
                    )
                    probabilities = _distance_probabilities(
                        scores, distance_scale, float(config["probability_temperature"])
                    )
                    metrics = _episode_metrics(labels[query_mask], probabilities, holdout)
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "direction": direction,
                            "heldout_class": class_name,
                            "method": method,
                            "shot": shot,
                            "draw": draw,
                            "support_sessions": selected.tolist(),
                            "distance_scale": distance_scale,
                            **metrics,
                        }
                    )
                    query_indices = np.flatnonzero(query_mask)
                    predictions = np.argmax(probabilities, axis=1)
                    for local_index, global_index in enumerate(query_indices):
                        predicted = int(predictions[local_index])
                        row: dict[str, object] = {
                            "episode_id": episode_id,
                            "direction": direction,
                            "heldout_class": class_name,
                            "method": method,
                            "shot": shot,
                            "draw": draw,
                            "session_id": sessions[global_index],
                            "date_token": dates[global_index],
                            "era": eras[global_index],
                            "true_label": int(labels[global_index]),
                            "true_class": CLASS_NAMES[int(labels[global_index])],
                            "predicted_label": predicted,
                            "predicted_class": CLASS_NAMES[predicted],
                            "minimum_distance": float(np.min(scores[local_index])),
                            "predicted_class_disagreement": float(disagreement[local_index, predicted]),
                        }
                        for class_id, name in enumerate(CLASS_NAMES):
                            row[f"prob_{name}"] = float(probabilities[local_index, class_id])
                        prediction_rows.append(row)
            print(f"[ROBUST ENROLLMENT] {direction} holdout={class_name}", flush=True)
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode["direction"], episode["heldout_class"], episode["method"], episode["shot"])].append(episode)
    summaries = []
    for key, local in sorted(grouped.items()):
        row: dict[str, object] = {
            "direction": key[0],
            "heldout_class": key[1],
            "method": key[2],
            "shot": key[3],
            "draws": len(local),
        }
        for metric in (
            "enrollment_h",
            "session_macro_f1",
            "worst_class_recall",
            "enrolled_class_recall",
            "base_class_accuracy",
            "negative_log_likelihood",
            "brier_score",
            "ece_10",
            "risk_coverage_auc",
        ):
            values = np.asarray([float(item[metric]) for item in local])
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_worst"] = float(np.min(values) if metric not in {"negative_log_likelihood", "brier_score", "ece_10", "risk_coverage_auc"} else np.max(values))
        summaries.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "support_draws.csv", support_rows)
    write_csv(
        output_dir / "episode_metrics.csv",
        [
            {
                **row,
                "support_sessions": json.dumps(row["support_sessions"]),
                "per_class_recall": json.dumps(row["per_class_recall"]),
            }
            for row in episodes
        ],
    )
    write_csv(output_dir / "random_draw_summary.csv", summaries)
    _write_gzip_csv(output_dir / "query_predictions.csv.gz", prediction_rows)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": config["protocol_name"],
        "evidence_status": config["evidence_status"],
        "config_sha256": config_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "input_hashes": {
            "morphology_bundle_sha256": sha256_file(morphology_bundle_path),
            "window_attributes_sha256": sha256_file(window_attributes_path),
            "session_attributes_sha256": sha256_file(session_attributes_path),
            **{path.name: sha256_file(path) for path in manifests},
        },
        "episode_count": len(episodes),
        "support_episode_count": len({row["episode_id"] for row in support_rows}),
        "prediction_count": len(prediction_rows),
        "random_draw_summary": summaries,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "robust_enrollment_results.json"
        },
        "limitations": [
            "All acquisition-era query outcomes are historically exposed; evidence is retrospective.",
            "The seven-session support pools restrict diversity-aware selection and tail estimation.",
            "Trimmed sliced projections approximate partial morphology overlap; they are not a full unbalanced optimal-transport solver.",
            "Distance softmax probabilities are calibrated controls, not guaranteed generative probabilities.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "process_memory": process_memory_snapshot(),
    }
    return finalize_payload(payload, output_dir / "robust_enrollment_results.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphology-bundle", type=Path, required=True)
    parser.add_argument("--window-attributes", type=Path, required=True)
    parser.add_argument("--session-attributes", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        morphology_bundle_path=args.morphology_bundle,
        window_attributes_path=args.window_attributes,
        session_attributes_path=args.session_attributes,
        manifests=args.manifests,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"], "episode_count": result["episode_count"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
