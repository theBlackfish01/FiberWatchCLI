"""Distributional and ordered-trajectory session models for Phi-OTDR.

This development evaluator never consumes the locked target query.  Each
leave-one-class-out fold fits window preprocessing only on seen-class source
training sessions, then simulates enrollment with held-out-class source
training sessions as support and source validation sessions as query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .metrics import classification_metrics, harmonic_mean, open_set_metrics


@dataclass(frozen=True)
class SessionWindows:
    session_id: str
    label: int
    partition: str
    window_ids: np.ndarray
    values: np.ndarray


def feature_masks(names: np.ndarray) -> dict[str, np.ndarray]:
    names = names.astype(str)
    amplitude_globals = {"global_mean", "global_std", "global_range"}
    amplitude = np.asarray([name.startswith("raw_") or name in amplitude_globals for name in names])
    return {
        "amplitude": amplitude,
        "dynamics": ~amplitude,
        "full": np.ones(len(names), dtype=bool),
    }


def _fold_seed(seed: int, direction: str, holdout: int, view: str) -> int:
    digest = hashlib.sha256(f"{seed}|{direction}|{holdout}|{view}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _session_windows(
    values: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    partitions: np.ndarray,
    window_ids: np.ndarray,
    selected: np.ndarray,
) -> list[SessionWindows]:
    rows: list[SessionWindows] = []
    for session_id in sorted(set(sessions[selected].astype(str).tolist())):
        indices = np.flatnonzero(selected & (sessions == session_id))
        order = np.argsort(window_ids[indices], kind="stable")
        indices = indices[order]
        unique_labels = set(labels[indices].tolist())
        unique_partitions = set(partitions[indices].astype(str).tolist())
        if len(unique_labels) != 1 or len(unique_partitions) != 1:
            raise ValueError(f"Session metadata is inconsistent: {session_id}")
        rows.append(
            SessionWindows(
                session_id=session_id,
                label=int(labels[indices[0]]),
                partition=str(partitions[indices[0]]),
                window_ids=window_ids[indices].astype(np.int32),
                values=values[indices].astype(np.float32),
            )
        )
    return rows


def _safe_std(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0) if len(values) > 1 else np.zeros(values.shape[1], dtype=float)


def session_descriptor(
    session: SessionWindows,
    method: str,
    *,
    projections: np.ndarray | None = None,
) -> np.ndarray:
    values = session.values.astype(np.float64)
    if method == "mean":
        return np.mean(values, axis=0)
    if method == "robust_quantiles":
        quantiles = np.quantile(values, (0.10, 0.50, 0.90), axis=0)
        return np.concatenate((np.mean(values, axis=0), _safe_std(values), quantiles.reshape(-1)))
    if method == "sliced_wasserstein":
        if projections is None:
            raise ValueError("Sliced-Wasserstein descriptor requires projections")
        projected = values @ projections.T
        quantiles = np.quantile(projected, np.linspace(0.05, 0.95, 19), axis=0)
        return quantiles.T.reshape(-1)
    if method != "ordered_trajectory":
        raise ValueError(f"Unknown session descriptor: {method}")

    count, dimensions = values.shape
    mean = np.mean(values, axis=0)
    standard_deviation = _safe_std(values)
    if count > 1:
        index = session.window_ids.astype(np.float64)
        span = max(float(index[-1] - index[0]), 1.0)
        normalized_time = (index - index[0]) / span
        centered = normalized_time - np.mean(normalized_time)
        denominator = max(float(np.sum(centered**2)), 1e-12)
        slope = np.sum(centered[:, None] * (values - mean), axis=0) / denominator
        delta = np.diff(values, axis=0)
        delta_mean_abs = np.mean(np.abs(delta), axis=0)
        gaps = np.diff(index)
        gap_summary = np.asarray(
            [np.mean(gaps), np.std(gaps), np.max(gaps)], dtype=np.float64
        ) / span
    else:
        normalized_time = np.zeros(1)
        slope = np.zeros(dimensions)
        delta_mean_abs = np.zeros(dimensions)
        gap_summary = np.zeros(3)
    # Normalized temporal bins preserve coarse evolution while remaining valid
    # for irregular indices and one-window sessions.
    phase_means: list[np.ndarray] = []
    for start, stop in zip(np.linspace(0.0, 0.75, 4), np.linspace(0.25, 1.0, 4), strict=True):
        in_phase = (normalized_time >= start) & (
            (normalized_time <= stop) if stop == 1.0 else (normalized_time < stop)
        )
        phase_means.append(np.mean(values[in_phase], axis=0) if np.any(in_phase) else mean)
    return np.concatenate((mean, standard_deviation, slope, delta_mean_abs, *phase_means, gap_summary))


def _descriptor_matrix(
    sessions: Iterable[SessionWindows], method: str, projections: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(sessions)
    return (
        np.stack([session_descriptor(row, method, projections=projections) for row in rows]),
        np.asarray([row.label for row in rows], dtype=np.int64),
        np.asarray([row.session_id for row in rows]),
    )


def class_scores(
    query: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    class_ids: list[int],
    *,
    neighbors: int = 3,
) -> np.ndarray:
    distances = pairwise_distances(query, gallery, metric="euclidean")
    scores: list[np.ndarray] = []
    for class_id in class_ids:
        class_distances = distances[:, gallery_labels == class_id]
        count = min(neighbors, class_distances.shape[1])
        nearest = np.partition(class_distances, count - 1, axis=1)[:, :count]
        scores.append(-np.mean(nearest, axis=1))
    return np.column_stack(scores)


def select_support(
    candidates: np.ndarray,
    base_gallery: np.ndarray,
    *,
    strategy: str,
    shot: int,
    seed: int,
) -> np.ndarray:
    """Select unique candidate indices without observing validation queries."""
    if shot > len(candidates):
        raise ValueError(f"Shot {shot} exceeds candidate count {len(candidates)}")
    if strategy == "random":
        return np.sort(np.random.default_rng(seed).choice(len(candidates), shot, replace=False))
    pairwise = pairwise_distances(candidates, candidates)
    center = int(np.argmin(np.sum(pairwise, axis=1)))
    if strategy == "medoid":
        return np.argsort(pairwise[center], kind="stable")[:shot]
    selected = [center]
    if strategy == "farthest_first":
        while len(selected) < shot:
            minimum = np.min(pairwise[:, selected], axis=1)
            minimum[selected] = -np.inf
            selected.append(int(np.argmax(minimum)))
        return np.asarray(selected, dtype=int)
    scale = max(float(np.median(pairwise[pairwise > 0])), 1e-6)
    similarity = np.exp(-pairwise / scale)
    novelty = np.min(pairwise_distances(candidates, base_gallery), axis=1)
    novelty = (novelty - np.min(novelty)) / max(float(np.ptp(novelty)), 1e-12)
    if strategy not in {"facility_location", "uncertainty_diversity"}:
        raise ValueError(f"Unknown support strategy: {strategy}")
    selected = []
    coverage = np.zeros(len(candidates), dtype=float)
    while len(selected) < shot:
        gains = np.full(len(candidates), -np.inf)
        for candidate in range(len(candidates)):
            if candidate in selected:
                continue
            gain = float(np.sum(np.maximum(coverage, similarity[:, candidate]) - coverage))
            if strategy == "uncertainty_diversity":
                gain += 0.25 * float(novelty[candidate]) * len(candidates)
            gains[candidate] = gain
        chosen = int(np.argmax(gains))
        selected.append(chosen)
        coverage = np.maximum(coverage, similarity[:, chosen])
    return np.asarray(selected, dtype=int)


def _post_metrics(true: np.ndarray, predicted: np.ndarray, holdout: int) -> dict[str, object]:
    base = true != holdout
    enrolled = ~base
    base_accuracy = float(np.mean(predicted[base] == true[base]))
    enrolled_recall = float(np.mean(predicted[enrolled] == holdout))
    return {
        "base_accuracy": base_accuracy,
        "enrolled_recall": enrolled_recall,
        "enrollment_h": harmonic_mean(base_accuracy, enrolled_recall),
        "classification": classification_metrics(true, predicted),
    }


def _draw_seed(seed: int, holdout: int, shot: int, draw: int, method: str) -> int:
    digest = hashlib.sha256(f"{seed}|{holdout}|{shot}|{draw}|{method}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def evaluate_development(
    bundle: np.lib.npyio.NpzFile,
    *,
    seed: int,
    views: tuple[str, ...],
    components: int,
    support_draws: int,
) -> dict[str, object]:
    features = bundle["features"].astype(np.float32)
    labels = bundle["labels"].astype(np.int64)
    session_ids = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    window_ids = bundle["window_ids"].astype(np.int32)
    names = bundle["feature_names"].astype(str)
    if "target_query" in set(partitions.tolist()):
        raise ValueError("Development session model must not receive target-query features")
    methods = ("mean", "robust_quantiles", "sliced_wasserstein", "ordered_trajectory")
    strategies = ("random", "medoid", "farthest_first", "facility_location", "uncertainty_diversity")
    masks = feature_masks(names)
    direction_eras = sorted(set(bundle["eras"].astype(str).tolist()))
    direction = "_and_".join(direction_eras)
    fold_results: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for view in views:
        feature_mask = masks[view]
        for holdout in range(len(CLASS_NAMES)):
            seen = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
            fit = (partitions == "source_train") & np.isin(labels, seen)
            source = np.asarray([value.startswith("source_") for value in partitions])
            fold_seed = _fold_seed(seed, direction, holdout, view)
            scaler = StandardScaler().fit(features[fit][:, feature_mask])
            scaled_fit = scaler.transform(features[fit][:, feature_mask])
            component_count = min(components, scaled_fit.shape[1], scaled_fit.shape[0] - 1)
            pca = PCA(
                n_components=component_count,
                whiten=True,
                svd_solver="randomized",
                random_state=fold_seed,
            ).fit(scaled_fit)
            source_values = np.zeros((len(features), component_count), dtype=np.float32)
            source_values[source] = pca.transform(
                scaler.transform(features[source][:, feature_mask])
            ).astype(np.float32)
            source_sessions = _session_windows(
                source_values, labels, session_ids, partitions, window_ids, source
            )
            rng = np.random.default_rng(fold_seed)
            projections = rng.normal(size=(16, component_count))
            projections /= np.maximum(np.linalg.norm(projections, axis=1, keepdims=True), 1e-12)

            for method in methods:
                descriptor, session_y, sessions = _descriptor_matrix(
                    source_sessions, method, projections
                )
                session_partition = np.asarray(
                    [row.partition for row in source_sessions]
                )
                train_known = (session_partition == "source_train") & np.isin(session_y, seen)
                support_candidates = (session_partition == "source_train") & (session_y == holdout)
                validation = session_partition == "source_validation"
                calibration = (session_partition == "source_calibration") & np.isin(session_y, seen)
                descriptor_scaler = StandardScaler().fit(descriptor[train_known])
                represented = descriptor_scaler.transform(descriptor).astype(np.float32)
                gallery = represented[train_known]
                gallery_y = session_y[train_known]
                candidate_x = represented[support_candidates]
                candidate_ids = sessions[support_candidates]
                if len(candidate_x) < 7:
                    raise ValueError(
                        f"Holdout {CLASS_NAMES[holdout]} has only {len(candidate_x)} support candidates"
                    )
                calibration_scores = class_scores(
                    represented[calibration], gallery, gallery_y, seen
                )
                known_confidence = np.max(calibration_scores, axis=1)
                threshold = float(np.quantile(known_confidence, 0.05, method="higher"))
                validation_scores = class_scores(
                    represented[validation], gallery, gallery_y, seen
                )
                best = np.argmax(validation_scores, axis=1)
                predicted = np.asarray(seen)[best]
                confidence = validation_scores[np.arange(len(best)), best]
                validation_y = session_y[validation]
                is_known = validation_y != holdout
                pre = open_set_metrics(
                    confidence,
                    is_known,
                    predicted == validation_y,
                    threshold=threshold,
                )

                post: dict[str, object] = {}
                for shot in (1, 3, 5):
                    for strategy in strategies:
                        draws = support_draws if strategy == "random" else 1
                        rows: list[dict[str, object]] = []
                        for draw in range(draws):
                            selected = select_support(
                                candidate_x,
                                gallery,
                                strategy=strategy,
                                shot=shot,
                                seed=_draw_seed(seed, holdout, shot, draw, method),
                            )
                            enrolled_gallery = np.concatenate((gallery, candidate_x[selected]), axis=0)
                            enrolled_labels = np.concatenate(
                                (gallery_y, np.full(len(selected), holdout, dtype=np.int64))
                            )
                            enrolled_classes = sorted(seen + [holdout])
                            scores = class_scores(
                                represented[validation],
                                enrolled_gallery,
                                enrolled_labels,
                                enrolled_classes,
                            )
                            enrolled_pred = np.asarray(enrolled_classes)[np.argmax(scores, axis=1)]
                            rows.append(
                                {
                                    "draw": draw,
                                    "support_sessions": candidate_ids[selected].tolist(),
                                    "metrics": _post_metrics(validation_y, enrolled_pred, holdout),
                                }
                            )
                        values = [float(row["metrics"]["enrollment_h"]) for row in rows]
                        post[f"{strategy}__{shot}"] = {
                            "draws": rows,
                            "enrollment_h_mean": float(np.mean(values)),
                            "enrollment_h_min": float(np.min(values)),
                            "enrollment_h_max": float(np.max(values)),
                        }
                fold_results.append(
                    {
                        "view": view,
                        "holdout_class_id": holdout,
                        "holdout_class": CLASS_NAMES[holdout],
                        "method": method,
                        "pca_components": component_count,
                        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                        "descriptor_dimensions": int(represented.shape[1]),
                        "train_gallery_sessions": int(np.sum(train_known)),
                        "calibration_sessions": int(np.sum(calibration)),
                        "support_candidate_sessions": int(np.sum(support_candidates)),
                        "validation_sessions": int(np.sum(validation)),
                        "known_only_threshold": threshold,
                        "pre_enrollment": pre,
                        "post_enrollment": post,
                    }
                )
                print(
                    f"[{view} {CLASS_NAMES[holdout]} {method}] "
                    f"preH={pre['detection_h']:.3f} "
                    f"random5={post['random__5']['enrollment_h_mean']:.3f} "
                    f"facility5={post['facility_location__5']['enrollment_h_mean']:.3f}",
                    flush=True,
                )

    summaries: dict[str, object] = {}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in fold_results:
        grouped[(str(row["view"]), str(row["method"]))].append(row)
    for (view, method), rows in grouped.items():
        entry: dict[str, object] = {
            "pre_detection_h_mean": float(
                np.mean([row["pre_enrollment"]["detection_h"] for row in rows])
            ),
            "pre_detection_h_worst_holdout": float(
                np.min([row["pre_enrollment"]["detection_h"] for row in rows])
            ),
            "unknown_auroc_mean": float(
                np.mean([row["pre_enrollment"]["unknown_auroc"] for row in rows])
            ),
        }
        for strategy in strategies:
            for shot in (1, 3, 5):
                key = f"{strategy}__{shot}"
                fold_means = [row["post_enrollment"][key]["enrollment_h_mean"] for row in rows]
                fold_mins = [row["post_enrollment"][key]["enrollment_h_min"] for row in rows]
                entry[f"{key}__h_mean"] = float(np.mean(fold_means))
                entry[f"{key}__worst_holdout_or_draw"] = float(np.min(fold_mins))
        summaries[f"{view}__{method}"] = entry
    ranking = sorted(
        summaries,
        key=lambda key: (
            summaries[key]["random__5__h_mean"],
            summaries[key]["pre_detection_h_mean"],
            summaries[key]["random__5__worst_holdout_or_draw"],
        ),
        reverse=True,
    )
    return {
        "protocol": "source-only leave-one-class-out session-distribution development",
        "final_query_used": False,
        "seed": seed,
        "views": list(views),
        "pca_components_requested": components,
        "support_draws": support_draws,
        "descriptor_methods": list(methods),
        "support_strategies": list(strategies),
        "ranking": ranking,
        "method_summary": summaries,
        "fold_results": fold_results,
        "elapsed_seconds": time.perf_counter() - started_all,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--views", nargs="+", choices=("amplitude", "dynamics", "full"), default=["full"])
    parser.add_argument("--components", type=int, default=24)
    parser.add_argument("--support-draws", type=int, default=30)
    args = parser.parse_args()
    bundle = np.load(args.features, allow_pickle=False)
    try:
        payload = evaluate_development(
            bundle,
            seed=args.seed,
            views=tuple(args.views),
            components=args.components,
            support_draws=args.support_draws,
        )
    finally:
        bundle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": payload["ranking"], "method_summary": payload["method_summary"]}, indent=2))


if __name__ == "__main__":
    main()
