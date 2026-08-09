"""Session-distribution retrieval and few-shot enrollment for Phi-OTDR v3."""

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
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, pairwise_distances, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES, canonical_json_hash
from .metrics import harmonic_mean


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def select_support(candidates: np.ndarray, *, selector: str, shot: int, seed: int) -> np.ndarray:
    """Select support from its candidate pool without observing any query."""
    if shot < 1 or shot > len(candidates):
        raise ValueError("Invalid support shot count")
    if selector == "random":
        return np.sort(np.random.default_rng(seed).choice(len(candidates), shot, replace=False))
    distances = pairwise_distances(candidates, metric="euclidean")
    medoid = int(np.argmin(np.sum(distances, axis=1)))
    if selector == "medoid":
        return np.argsort(distances[medoid], kind="stable")[:shot]
    if selector == "k_center":
        selected = [medoid]
        while len(selected) < shot:
            minimum = np.min(distances[:, selected], axis=1)
            minimum[selected] = -np.inf
            selected.append(int(np.argmax(minimum)))
        return np.asarray(selected, dtype=np.int64)
    if selector == "pool_coverage":
        positive = distances[distances > 0]
        scale = max(float(np.median(positive)) if len(positive) else 1.0, 1e-6)
        similarity = np.exp(-distances / scale)
        coverage = np.zeros(len(candidates), dtype=np.float64)
        selected = []
        while len(selected) < shot:
            gains = np.full(len(candidates), -np.inf)
            for candidate in range(len(candidates)):
                if candidate in selected:
                    continue
                gains[candidate] = np.sum(np.maximum(coverage, similarity[:, candidate]) - coverage)
            chosen = int(np.argmax(gains))
            selected.append(chosen)
            coverage = np.maximum(coverage, similarity[:, chosen])
        return np.asarray(selected, dtype=np.int64)
    raise ValueError(f"Unknown support selector: {selector}")


def _session_metadata(bundle: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    session_array = bundle["sessions"].astype(str)
    unique = np.asarray(sorted(set(session_array.tolist())))
    first = {session: int(np.flatnonzero(session_array == session)[0]) for session in unique}
    labels = np.asarray([bundle["labels"][first[session]] for session in unique], dtype=np.int64)
    indices = {session: np.flatnonzero(session_array == session) for session in unique}
    return unique, labels, indices


def _aggregate_descriptor(
    aggregate: np.ndarray,
    fit: np.ndarray,
    components: int,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(aggregate[fit])
    scaled = scaler.transform(aggregate)
    count = min(components, int(np.sum(fit)) - 1, scaled.shape[1])
    pca = PCA(
        n_components=count,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed % (2**32)),
    )
    pca.fit(scaled[fit])
    return pca.transform(scaled).astype(np.float32)


def _sliced_wasserstein_descriptor(
    window_features: np.ndarray,
    window_sessions: np.ndarray,
    sessions: np.ndarray,
    fit_window: np.ndarray,
    fit_session: np.ndarray,
    *,
    projection_count: int,
    quantile_count: int,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(window_features[fit_window])
    scaled = scaler.transform(window_features).astype(np.float32)
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(projection_count, scaled.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12
    projected = scaled @ projections.T
    quantiles = np.linspace(0.05, 0.95, quantile_count)
    rows = []
    for session in sessions:
        local = projected[window_sessions == session]
        rows.append(np.quantile(local, quantiles, axis=0).T.reshape(-1))
    descriptors = np.asarray(rows, dtype=np.float32)
    descriptor_scaler = StandardScaler().fit(descriptors[fit_session])
    return descriptor_scaler.transform(descriptors).astype(np.float32)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _class_distances(
    query: np.ndarray,
    base_gallery: np.ndarray,
    base_labels: np.ndarray,
    support: np.ndarray | None,
    holdout: int,
    *,
    method: str,
    neighbors: int,
) -> np.ndarray:
    scores = np.full((len(query), len(CLASS_NAMES)), np.inf, dtype=np.float64)
    if method == "class_prototype":
        for class_id in range(len(CLASS_NAMES)):
            if class_id == holdout:
                gallery = support
            else:
                gallery = base_gallery[base_labels == class_id]
            if gallery is not None and len(gallery):
                scores[:, class_id] = np.linalg.norm(query - np.mean(gallery, axis=0), axis=1)
        return scores
    distances = pairwise_distances(query, base_gallery, metric="euclidean")
    for class_id in range(len(CLASS_NAMES)):
        if class_id == holdout:
            if support is None:
                continue
            local = pairwise_distances(query, support, metric="euclidean")
        else:
            local = distances[:, base_labels == class_id]
        count = min(neighbors, local.shape[1])
        scores[:, class_id] = np.mean(np.partition(local, count - 1, axis=1)[:, :count], axis=1)
    return scores


def _classification_metrics(labels: np.ndarray, prediction: np.ndarray, holdout: int) -> dict[str, object]:
    recall = recall_score(labels, prediction, labels=np.arange(len(CLASS_NAMES)), average=None, zero_division=0)
    seen = labels != holdout
    base_accuracy = float(np.mean(prediction[seen] == labels[seen]))
    enrolled_recall = float(np.mean(prediction[~seen] == holdout))
    return {
        "session_macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "base_class_accuracy": base_accuracy,
        "enrolled_class_recall": enrolled_recall,
        "enrollment_h": harmonic_mean(base_accuracy, enrolled_recall),
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(CLASS_NAMES)},
        "worst_class_recall": float(np.min(recall)),
    }


def _detection_metrics(
    calibration_distance: np.ndarray,
    query_distance: np.ndarray,
    query_labels: np.ndarray,
    holdout: int,
    quantile: float,
) -> dict[str, object]:
    threshold = float(np.quantile(calibration_distance, quantile))
    unknown = query_labels == holdout
    known_acceptance = float(np.mean(query_distance[~unknown] <= threshold))
    unknown_recall = float(np.mean(query_distance[unknown] > threshold))
    return {
        "threshold": threshold,
        "known_acceptance": known_acceptance,
        "unknown_recall": unknown_recall,
        "detection_h": harmonic_mean(known_acceptance, unknown_recall),
        "unknown_auroc": float(roc_auc_score(unknown.astype(int), query_distance)),
        "unknown_aupr": float(average_precision_score(unknown.astype(int), query_distance)),
    }


def run(
    *,
    bundle_path: Path,
    session_aggregate_path: Path,
    window_view_path: Path,
    manifests: Sequence[Path],
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = config_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(config) != expected_hash:
        raise ValueError("Enrollment v3 config hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    with np.load(session_aggregate_path, allow_pickle=False) as source:
        aggregate = source["features"].astype(np.float32)
        aggregate_sessions = source["sessions"].astype(str)
        aggregate_labels = source["labels"].astype(np.int64)
    with np.load(window_view_path, allow_pickle=False) as source:
        window_view = source["features"].astype(np.float32)
    if len(window_view) != len(bundle["features"]):
        raise ValueError("Window distribution view is misaligned with the base bundle")
    sessions, labels, session_indices = _session_metadata(bundle)
    if not np.array_equal(sessions, aggregate_sessions) or not np.array_equal(labels, aggregate_labels):
        raise ValueError("Session aggregate cache is misaligned")
    window_sessions = bundle["sessions"].astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    detections = []
    support_rows = []
    prediction_rows = []
    started = time.perf_counter()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
        partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        for holdout in range(len(CLASS_NAMES)):
            fit_session = (partitions == "source_train") & (labels != holdout)
            fit_window = np.asarray(
                [session_rows[session]["partition"] == "source_train" and session_rows[session]["class_id"] != holdout for session in window_sessions]
            )
            seed = _seed(20260808, direction, holdout, "descriptors")
            aggregate_descriptor = _aggregate_descriptor(
                aggregate,
                fit_session,
                int(config["distribution"]["aggregate_pca_components"]),
                seed,
            )
            sw_descriptor = _sliced_wasserstein_descriptor(
                window_view,
                window_sessions,
                sessions,
                fit_window,
                fit_session,
                projection_count=int(config["distribution"]["sliced_wasserstein_projections"]),
                quantile_count=int(config["distribution"]["sliced_wasserstein_quantiles"]),
                seed=seed,
            )
            descriptors = {
                "class_prototype": aggregate_descriptor,
                "sliced_wasserstein_session_gallery": sw_descriptor,
                "registered_distribution_hybrid": np.concatenate(
                    (_unit_rows(aggregate_descriptor), _unit_rows(sw_descriptor)), axis=1
                ).astype(np.float32),
            }
            query_mask = partitions == "target_query"
            calibration_mask = (partitions == "target_calibration") & (labels != holdout)
            support_mask = (partitions == "target_support") & (labels == holdout)
            base_mask = fit_session
            if np.any(np.isin(sessions[support_mask], sessions[query_mask])):
                raise AssertionError("Support and query sessions overlap")
            candidate_sessions = sessions[support_mask]
            query_sessions = sessions[query_mask]
            for method, descriptor in descriptors.items():
                base_gallery = descriptor[base_mask]
                base_labels = labels[base_mask]
                calibration_scores = _class_distances(
                    descriptor[calibration_mask],
                    base_gallery,
                    base_labels,
                    None,
                    holdout,
                    method=method,
                    neighbors=int(config["distribution"]["session_gallery_neighbors"]),
                )
                query_scores_pre = _class_distances(
                    descriptor[query_mask],
                    base_gallery,
                    base_labels,
                    None,
                    holdout,
                    method=method,
                    neighbors=int(config["distribution"]["session_gallery_neighbors"]),
                )
                seen_columns = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
                detection = _detection_metrics(
                    np.min(calibration_scores[:, seen_columns], axis=1),
                    np.min(query_scores_pre[:, seen_columns], axis=1),
                    labels[query_mask],
                    holdout,
                    float(config["detection"]["known_acceptance_quantile"]),
                )
                detections.append(
                    {
                        "direction": direction,
                        "heldout_class": CLASS_NAMES[holdout],
                        "method": method,
                        **detection,
                    }
                )
                candidates = descriptor[support_mask]
                for shot in config["support"]["shots"]:
                    selector_draws = [("random", draw) for draw in range(int(config["support"]["random_draws"]))]
                    selector_draws.extend((selector, 0) for selector in ("medoid", "k_center", "pool_coverage"))
                    for selector, draw in selector_draws:
                        support_seed = _seed(20260808, direction, holdout, shot, selector, draw)
                        selected_local = select_support(
                            candidates,
                            selector=selector,
                            shot=int(shot),
                            seed=support_seed,
                        )
                        selected_support = candidates[selected_local]
                        selected_sessions = candidate_sessions[selected_local]
                        if len(set(selected_sessions.tolist())) != int(shot):
                            raise AssertionError("Support selector returned duplicate sessions")
                        for rank, session in enumerate(selected_sessions):
                            support_rows.append(
                                {
                                    "direction": direction,
                                    "heldout_class": CLASS_NAMES[holdout],
                                    "method": method,
                                    "shot": int(shot),
                                    "selector": selector,
                                    "draw": draw,
                                    "seed": support_seed,
                                    "rank": rank,
                                    "session_id": session,
                                }
                            )
                        scores = _class_distances(
                            descriptor[query_mask],
                            base_gallery,
                            base_labels,
                            selected_support,
                            holdout,
                            method=method,
                            neighbors=int(config["distribution"]["session_gallery_neighbors"]),
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
                                "shot": int(shot),
                                "selector": selector,
                                "draw": draw,
                                "support_sessions": selected_sessions.tolist(),
                                **metrics,
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
            print(f"[ENROLLMENT] {direction} holdout={CLASS_NAMES[holdout]}", flush=True)
    random_groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        if row["selector"] == "random":
            random_groups[(row["direction"], row["heldout_class"], row["method"], row["shot"])].append(row)
    random_summary = []
    for key, rows in sorted(random_groups.items()):
        h = np.asarray([row["enrollment_h"] for row in rows])
        f1 = np.asarray([row["session_macro_f1"] for row in rows])
        worst = np.asarray([row["worst_class_recall"] for row in rows])
        random_summary.append(
            {
                "direction": key[0],
                "heldout_class": key[1],
                "method": key[2],
                "shot": key[3],
                "draws": len(rows),
                "enrollment_h_mean": float(np.mean(h)),
                "enrollment_h_std": float(np.std(h, ddof=1)),
                "enrollment_h_worst_draw": float(np.min(h)),
                "macro_f1_mean": float(np.mean(f1)),
                "macro_f1_worst_draw": float(np.min(f1)),
                "worst_class_recall_mean": float(np.mean(worst)),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "session-distribution open-world enrollment v3",
        "evidence_status": "retrospective development",
        "config_sha256": expected_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "episode_count": len(episodes),
        "detection_fold_count": len(detections),
        "elapsed_seconds": time.perf_counter() - started,
        "pre_enrollment_detection": detections,
        "random_draw_summary": random_summary,
        "episodes": episodes,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "distributional_enrollment_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for filename, rows in (
        ("support_draws.csv", support_rows),
        ("query_predictions.csv.gz", prediction_rows),
    ):
        opener = gzip.open if filename.endswith(".gz") else open
        mode = "wt" if filename.endswith(".gz") else "w"
        with opener(output_dir / filename, mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--session-aggregate", type=Path, required=True)
    parser.add_argument("--window-view", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        bundle_path=args.bundle,
        session_aggregate_path=args.session_aggregate,
        window_view_path=args.window_view,
        manifests=args.manifests,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "episode_count": result["episode_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "payload_sha256": result["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
