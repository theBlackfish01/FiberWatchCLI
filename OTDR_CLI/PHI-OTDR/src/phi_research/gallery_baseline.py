"""Leave-one-class-out gallery enrollment on session-safe signal features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .metrics import calibrate_rejection_threshold, classification_metrics, harmonic_mean, open_set_metrics


def _feature_masks(names: np.ndarray) -> dict[str, np.ndarray]:
    names = names.astype(str)
    amplitude_globals = {"global_mean", "global_std", "global_range"}
    amplitude = np.asarray([name.startswith("raw_") or name in amplitude_globals for name in names])
    return {
        "amplitude": amplitude,
        "dynamics": ~amplitude,
        "full": np.ones(len(names), dtype=bool),
    }


def _session_prototypes(
    features: np.ndarray, labels: np.ndarray, sessions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    targets: list[int] = []
    ids: list[str] = []
    for session in sorted(np.unique(sessions.astype(str))):
        selected = sessions.astype(str) == session
        unique_labels = np.unique(labels[selected])
        if len(unique_labels) != 1:
            raise ValueError(f"Session {session} spans multiple classes")
        vectors.append(np.mean(features[selected], axis=0))
        targets.append(int(unique_labels[0]))
        ids.append(session)
    return np.stack(vectors), np.asarray(targets), np.asarray(ids)


def _class_prototypes(
    session_features: np.ndarray, session_labels: np.ndarray, class_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    prototypes = [np.mean(session_features[session_labels == class_id], axis=0) for class_id in class_ids]
    return np.stack(prototypes), np.asarray(class_ids, dtype=int)


def _score_matrix(query: np.ndarray, prototypes: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        prototype_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        return query_norm @ prototype_norm.T
    if metric == "euclidean":
        squared = (
            np.sum(query**2, axis=1, keepdims=True)
            + np.sum(prototypes**2, axis=1)[None, :]
            - 2.0 * query @ prototypes.T
        )
        return -np.sqrt(np.maximum(squared, 0.0))
    raise ValueError(f"Unknown metric: {metric}")


def _predict(query: np.ndarray, prototypes: np.ndarray, prototype_labels: np.ndarray, metric: str):
    scores = _score_matrix(query, prototypes, metric)
    best = np.argmax(scores, axis=1)
    return prototype_labels[best], scores[np.arange(len(scores)), best], scores


def _draw_seed(seed: int, holdout: int, shot: int, draw: int) -> int:
    digest = hashlib.sha256(f"{seed}|{holdout}|{shot}|{draw}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _post_enrollment_metrics(
    true: np.ndarray, predicted: np.ndarray, holdout: int
) -> dict[str, object]:
    base = true != holdout
    enrolled = true == holdout
    base_accuracy = float(np.mean(predicted[base] == true[base]))
    enrolled_recall = float(np.mean(predicted[enrolled] == holdout))
    return {
        "base_accuracy": base_accuracy,
        "enrolled_recall": enrolled_recall,
        "enrollment_h": harmonic_mean(base_accuracy, enrolled_recall),
        "classification": classification_metrics(true, predicted),
    }


def _aggregate_session_scores(
    scores: np.ndarray, true: np.ndarray, sessions: np.ndarray, prototype_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    session_true: list[int] = []
    session_pred: list[int] = []
    sessions = sessions.astype(str)
    for session in sorted(np.unique(sessions)):
        selected = sessions == session
        labels = np.unique(true[selected])
        if len(labels) != 1:
            raise ValueError(f"Session {session} spans labels")
        session_true.append(int(labels[0]))
        session_pred.append(int(prototype_labels[np.argmax(np.mean(scores[selected], axis=0))]))
    return np.asarray(session_true), np.asarray(session_pred)


def _calibration_scores(
    train_session_x: np.ndarray,
    train_session_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    seen_classes: list[int],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    known_scores: list[np.ndarray] = []
    pseudo_unknown_scores: list[np.ndarray] = []
    for pseudo_unknown in seen_classes:
        gallery_classes = [class_id for class_id in seen_classes if class_id != pseudo_unknown]
        prototypes, prototype_labels = _class_prototypes(
            train_session_x, train_session_y, gallery_classes
        )
        selected = np.isin(calibration_y, seen_classes)
        _, confidence, _ = _predict(calibration_x[selected], prototypes, prototype_labels, metric)
        selected_labels = calibration_y[selected]
        pseudo_unknown_scores.append(confidence[selected_labels == pseudo_unknown])
        known_scores.append(confidence[selected_labels != pseudo_unknown])
    return np.concatenate(known_scores), np.concatenate(pseudo_unknown_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--support-draws", type=int, default=20)
    args = parser.parse_args()

    bundle = np.load(args.features, allow_pickle=False)
    x_all = bundle["features"]
    y_all = bundle["labels"]
    sessions_all = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    names = bundle["feature_names"].astype(str)
    if np.any(partitions == "final_query"):
        raise ValueError("Development gallery benchmark must not receive final_query")
    masks = _feature_masks(names)
    results: list[dict[str, object]] = []

    for holdout in range(len(CLASS_NAMES)):
        seen_classes = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
        train = (partitions == "train") & np.isin(y_all, seen_classes)
        validation = partitions == "validation"
        calibration = (partitions == "calibration") & np.isin(y_all, seen_classes)
        support = (partitions == "support") & (y_all == holdout)
        for ablation, mask in masks.items():
            scaler = StandardScaler().fit(x_all[train][:, mask])
            transformed = scaler.transform(x_all[:, mask]).astype(np.float32)
            train_session_x, train_session_y, _ = _session_prototypes(
                transformed[train], y_all[train], sessions_all[train]
            )
            support_session_x, support_session_y, support_session_ids = _session_prototypes(
                transformed[support], y_all[support], sessions_all[support]
            )
            base_prototypes, base_labels = _class_prototypes(
                train_session_x, train_session_y, seen_classes
            )
            for metric in ("cosine", "euclidean"):
                known_calibration, unknown_calibration = _calibration_scores(
                    train_session_x,
                    train_session_y,
                    transformed[calibration],
                    y_all[calibration],
                    seen_classes,
                    metric,
                )
                thresholds = calibrate_rejection_threshold(known_calibration, unknown_calibration)
                validation_pred, validation_confidence, base_scores = _predict(
                    transformed[validation], base_prototypes, base_labels, metric
                )
                is_known = y_all[validation] != holdout
                known_correct = validation_pred == y_all[validation]
                pre_enrollment = {
                    mode: open_set_metrics(
                        validation_confidence,
                        is_known,
                        known_correct,
                        threshold=float(threshold),
                    )
                    for mode, threshold in (
                        ("balanced", thresholds["balanced_threshold"]),
                        ("known_acceptance_95", thresholds["known_acceptance_threshold"]),
                    )
                }

                shots: dict[str, object] = {}
                for shot in (1, 3, 5):
                    draws: list[dict[str, object]] = []
                    for draw in range(args.support_draws):
                        rng = np.random.default_rng(_draw_seed(args.seed, holdout, shot, draw))
                        selected = np.sort(
                            rng.choice(len(support_session_ids), size=shot, replace=False)
                        )
                        enrolled_prototype = np.mean(support_session_x[selected], axis=0, keepdims=True)
                        prototypes = np.concatenate((base_prototypes, enrolled_prototype), axis=0)
                        prototype_labels = np.concatenate((base_labels, np.asarray([holdout])))
                        predicted, _, scores = _predict(
                            transformed[validation], prototypes, prototype_labels, metric
                        )
                        window_metrics = _post_enrollment_metrics(y_all[validation], predicted, holdout)
                        session_true, session_pred = _aggregate_session_scores(
                            scores, y_all[validation], sessions_all[validation], prototype_labels
                        )
                        session_metrics = _post_enrollment_metrics(session_true, session_pred, holdout)
                        draws.append(
                            {
                                "draw": draw,
                                "support_sessions": support_session_ids[selected].tolist(),
                                "window": window_metrics,
                                "session": session_metrics,
                            }
                        )
                    shots[str(shot)] = {
                        "draws": draws,
                        "window_enrollment_h_mean": float(np.mean([row["window"]["enrollment_h"] for row in draws])),
                        "window_enrollment_h_min": float(np.min([row["window"]["enrollment_h"] for row in draws])),
                        "session_enrollment_h_mean": float(np.mean([row["session"]["enrollment_h"] for row in draws])),
                        "session_enrollment_h_min": float(np.min([row["session"]["enrollment_h"] for row in draws])),
                    }
                results.append(
                    {
                        "holdout_class_id": holdout,
                        "holdout_class": CLASS_NAMES[holdout],
                        "ablation": ablation,
                        "metric": metric,
                        "calibration": thresholds,
                        "pre_enrollment": pre_enrollment,
                        "post_enrollment": shots,
                        "support_session_count": int(len(support_session_ids)),
                    }
                )
                print(
                    f"[{CLASS_NAMES[holdout]} {ablation} {metric}] "
                    f"AUROC={pre_enrollment['balanced']['unknown_auroc']:.3f} "
                    f"preH={pre_enrollment['balanced']['detection_h']:.3f} "
                    f"postH@5={shots['5']['session_enrollment_h_mean']:.3f}",
                    flush=True,
                )

    method_summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[f"{row['ablation']}__{row['metric']}"] .append(row)
    for key, rows in grouped.items():
        method_summary[key] = {
            "pre_unknown_auroc_mean": float(np.mean([row["pre_enrollment"]["balanced"]["unknown_auroc"] for row in rows])),
            "pre_detection_h_mean": float(np.mean([row["pre_enrollment"]["balanced"]["detection_h"] for row in rows])),
            "pre_detection_h_worst_holdout": float(np.min([row["pre_enrollment"]["balanced"]["detection_h"] for row in rows])),
            "post_session_h_1shot_mean": float(np.mean([row["post_enrollment"]["1"]["session_enrollment_h_mean"] for row in rows])),
            "post_session_h_5shot_mean": float(np.mean([row["post_enrollment"]["5"]["session_enrollment_h_mean"] for row in rows])),
            "post_session_h_5shot_worst_holdout_draw": float(np.min([row["post_enrollment"]["5"]["session_enrollment_h_min"] for row in rows])),
        }
    ranking = sorted(
        method_summary,
        key=lambda key: (
            method_summary[key]["pre_detection_h_mean"],
            method_summary[key]["post_session_h_5shot_mean"],
        ),
        reverse=True,
    )
    payload = {
        "protocol": "leave-one-class-out development benchmark on validation sessions",
        "seed": args.seed,
        "support_draws": args.support_draws,
        "final_query_used": False,
        "method_summary": method_summary,
        "development_ranking": ranking,
        "fold_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "method_summary": method_summary}, indent=2))


if __name__ == "__main__":
    main()
