"""Session-level rejection analysis for raw and frozen embedding galleries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .embedding_gallery import FeatureEncoder, _encode, _feature_masks
from .gallery_baseline import _class_prototypes, _predict, _score_matrix, _session_prototypes
from .metrics import calibrate_rejection_threshold, open_set_metrics


def _aggregate_open_scores(
    scores: np.ndarray,
    true: np.ndarray,
    sessions: np.ndarray,
    prototype_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    session_true: list[int] = []
    session_pred: list[int] = []
    confidence: list[float] = []
    ordered: list[str] = []
    mean_scores: list[np.ndarray] = []
    sessions = sessions.astype(str)
    for session in sorted(np.unique(sessions)):
        selected = sessions == session
        labels = np.unique(true[selected])
        if len(labels) != 1:
            raise ValueError(f"Session {session} spans labels")
        averaged = np.mean(scores[selected], axis=0)
        best = int(np.argmax(averaged))
        session_true.append(int(labels[0]))
        session_pred.append(int(prototype_labels[best]))
        confidence.append(float(averaged[best]))
        ordered.append(session)
        mean_scores.append(averaged)
    return (
        np.asarray(session_true),
        np.asarray(session_pred),
        np.asarray(confidence),
        np.stack(mean_scores),
        ordered,
    )


def _session_calibration(
    train_session_x: np.ndarray,
    train_session_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    calibration_sessions: np.ndarray,
    seen_classes: list[int],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    full_prototypes, full_labels = _class_prototypes(train_session_x, train_session_y, seen_classes)
    full_scores = _score_matrix(calibration_x, full_prototypes, metric)
    _, _, full_confidence, _, _ = _aggregate_open_scores(
        full_scores, calibration_y, calibration_sessions, full_labels
    )
    pseudo_unknown: list[np.ndarray] = []
    for pseudo_class in seen_classes:
        gallery_classes = [value for value in seen_classes if value != pseudo_class]
        prototypes, prototype_labels = _class_prototypes(
            train_session_x, train_session_y, gallery_classes
        )
        selected = np.isin(calibration_y, seen_classes)
        scores = _score_matrix(calibration_x[selected], prototypes, metric)
        true, _, confidence, _, _ = _aggregate_open_scores(
            scores,
            calibration_y[selected],
            calibration_sessions[selected],
            prototype_labels,
        )
        pseudo_unknown.append(confidence[true == pseudo_class])
    return full_confidence, np.concatenate(pseudo_unknown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = np.load(args.features, allow_pickle=False)
    x = bundle["features"]
    y = bundle["labels"]
    sessions = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    names = bundle["feature_names"].astype(str)
    if np.any(partitions == "final_query"):
        raise ValueError("Development session analysis cannot receive final_query")

    rows: list[dict[str, object]] = []
    representations = ("raw", "embedding")
    for ablation, mask in _feature_masks(names).items():
        for holdout in range(len(CLASS_NAMES)):
            seen_classes = [value for value in range(len(CLASS_NAMES)) if value != holdout]
            train = (partitions == "train") & np.isin(y, seen_classes)
            calibration = (partitions == "calibration") & np.isin(y, seen_classes)
            validation = partitions == "validation"
            for representation in representations:
                if representation == "raw":
                    scaler = StandardScaler().fit(x[train][:, mask])
                    represented = scaler.transform(x[:, mask]).astype(np.float32)
                else:
                    fold_dir = args.embedding_dir / ablation / CLASS_NAMES[holdout]
                    scaler = joblib.load(fold_dir / "scaler.joblib")
                    transformed = scaler.transform(x[:, mask]).astype(np.float32)
                    model = FeatureEncoder(int(np.sum(mask)), len(seen_classes)).to("cuda")
                    model.load_state_dict(
                        torch.load(fold_dir / "best_model.pt", map_location="cuda", weights_only=True)
                    )
                    represented = _encode(model, transformed, torch.device("cuda"))
                train_session_x, train_session_y, _ = _session_prototypes(
                    represented[train], y[train], sessions[train]
                )
                base_prototypes, base_labels = _class_prototypes(
                    train_session_x, train_session_y, seen_classes
                )
                for metric in ("cosine", "euclidean"):
                    known_calibration, pseudo_unknown = _session_calibration(
                        train_session_x,
                        train_session_y,
                        represented[calibration],
                        y[calibration],
                        sessions[calibration],
                        seen_classes,
                        metric,
                    )
                    thresholds = calibrate_rejection_threshold(
                        known_calibration, pseudo_unknown, target_known_acceptance=0.95
                    )
                    validation_scores = _score_matrix(
                        represented[validation], base_prototypes, metric
                    )
                    true, predicted, confidence, _, session_ids = _aggregate_open_scores(
                        validation_scores,
                        y[validation],
                        sessions[validation],
                        base_labels,
                    )
                    is_known = true != holdout
                    correct = predicted == true
                    metrics = {
                        "balanced": open_set_metrics(
                            confidence,
                            is_known,
                            correct,
                            threshold=float(thresholds["balanced_threshold"]),
                        ),
                        "known_acceptance_95": open_set_metrics(
                            confidence,
                            is_known,
                            correct,
                            threshold=float(thresholds["known_acceptance_threshold"]),
                        ),
                    }
                    rows.append(
                        {
                            "representation": representation,
                            "ablation": ablation,
                            "holdout_class_id": holdout,
                            "holdout_class": CLASS_NAMES[holdout],
                            "metric": metric,
                            "calibration": thresholds,
                            "metrics": metrics,
                            "validation_session_ids": session_ids,
                        }
                    )
                    print(
                        f"[{representation} {ablation} {CLASS_NAMES[holdout]} {metric}] "
                        f"AUROC={metrics['known_acceptance_95']['unknown_auroc']:.3f} "
                        f"H95={metrics['known_acceptance_95']['detection_h']:.3f}",
                        flush=True,
                    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['representation']}__{row['ablation']}__{row['metric']}"] .append(row)
    summary: dict[str, dict[str, float]] = {}
    for key, group in grouped.items():
        summary[key] = {
            "unknown_auroc_mean": float(np.mean([row["metrics"]["known_acceptance_95"]["unknown_auroc"] for row in group])),
            "known_acceptance_mean": float(np.mean([row["metrics"]["known_acceptance_95"]["known_acceptance"] for row in group])),
            "unknown_recall_mean": float(np.mean([row["metrics"]["known_acceptance_95"]["unknown_recall"] for row in group])),
            "detection_h_mean": float(np.mean([row["metrics"]["known_acceptance_95"]["detection_h"] for row in group])),
            "detection_h_worst_holdout": float(np.min([row["metrics"]["known_acceptance_95"]["detection_h"] for row in group])),
            "oscr_mean": float(np.mean([row["metrics"]["known_acceptance_95"]["oscr"] for row in group])),
        }
    ranking = sorted(
        summary,
        key=lambda key: (summary[key]["detection_h_mean"], summary[key]["unknown_auroc_mean"]),
        reverse=True,
    )
    payload = {
        "protocol": "session-level open-set development; known-only 95% acceptance calibration",
        "final_query_used": False,
        "summary": summary,
        "ranking": ranking,
        "fold_results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
