"""Class-density and session-neighbour open-set analysis on frozen embeddings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from .data_contract import CLASS_NAMES
from .embedding_gallery import FeatureEncoder, _encode, _feature_masks
from .gallery_baseline import _draw_seed, _post_enrollment_metrics, _session_prototypes
from .metrics import calibrate_rejection_threshold, open_set_metrics


class DensityGallery:
    def __init__(self, features: np.ndarray, labels: np.ndarray, class_ids: list[int], method: str):
        self.class_ids = np.asarray(class_ids, dtype=int)
        self.method = method
        self.members = {class_id: features[labels == class_id] for class_id in class_ids}
        self.gaussians = {}
        if method == "mahalanobis":
            self.gaussians = {
                class_id: LedoitWolf().fit(self.members[class_id]) for class_id in class_ids
            }

    def score(self, query: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for class_id in self.class_ids:
            members = self.members[int(class_id)]
            if self.method == "knn_euclidean_3":
                distances = np.sqrt(
                    np.maximum(
                        np.sum(query**2, axis=1, keepdims=True)
                        + np.sum(members**2, axis=1)[None, :]
                        - 2.0 * query @ members.T,
                        0.0,
                    )
                )
                k = min(3, members.shape[0])
                columns.append(-np.mean(np.partition(distances, k - 1, axis=1)[:, :k], axis=1))
            elif self.method == "knn_cosine_3":
                q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
                m = members / np.maximum(np.linalg.norm(members, axis=1, keepdims=True), 1e-12)
                similarities = q @ m.T
                k = min(3, members.shape[0])
                columns.append(np.mean(np.partition(similarities, -k, axis=1)[:, -k:], axis=1))
            elif self.method == "mahalanobis":
                columns.append(-np.sqrt(np.maximum(self.gaussians[int(class_id)].mahalanobis(query), 0.0)))
            else:
                raise ValueError(f"Unknown density method: {self.method}")
        return np.stack(columns, axis=1)


def _confidence_prediction(scores: np.ndarray, class_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = np.argmax(scores, axis=1)
    return class_ids[best], scores[np.arange(len(scores)), best]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablations", nargs="+", choices=("dynamics", "full"), default=["dynamics", "full"])
    args = parser.parse_args()
    bundle = np.load(args.features, allow_pickle=False)
    x = bundle["features"]
    y = bundle["labels"]
    sessions = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    names = bundle["feature_names"].astype(str)
    if np.any(partitions == "final_query"):
        raise ValueError("Development analysis cannot receive final_query")
    results: list[dict[str, object]] = []

    available_masks = _feature_masks(names)
    for ablation in args.ablations:
        mask = available_masks[ablation]
        for holdout in range(len(CLASS_NAMES)):
            seen_classes = [value for value in range(len(CLASS_NAMES)) if value != holdout]
            fold_dir = args.embedding_dir / ablation / CLASS_NAMES[holdout]
            scaler = joblib.load(fold_dir / "scaler.joblib")
            transformed = scaler.transform(x[:, mask]).astype(np.float32)
            model = FeatureEncoder(int(np.sum(mask)), len(seen_classes)).to("cuda")
            model.load_state_dict(torch.load(fold_dir / "best_model.pt", map_location="cuda", weights_only=True))
            represented = _encode(model, transformed, torch.device("cuda"))
            session_x, session_y, session_ids = _session_prototypes(represented, y, sessions)
            session_partition = np.asarray(
                [np.unique(partitions[sessions == session])[0] for session in session_ids]
            )
            train = (session_partition == "train") & np.isin(session_y, seen_classes)
            calibration = (session_partition == "calibration") & np.isin(session_y, seen_classes)
            validation = session_partition == "validation"
            support = (session_partition == "support") & (session_y == holdout)
            for method in ("knn_euclidean_3", "knn_cosine_3", "mahalanobis"):
                gallery = DensityGallery(session_x[train], session_y[train], seen_classes, method)
                calibration_scores = gallery.score(session_x[calibration])
                _, known_confidence = _confidence_prediction(calibration_scores, gallery.class_ids)
                pseudo_unknown_confidence: list[np.ndarray] = []
                for pseudo_class in seen_classes:
                    reduced_classes = [value for value in seen_classes if value != pseudo_class]
                    reduced = DensityGallery(session_x[train], session_y[train], reduced_classes, method)
                    selected = calibration & (session_y == pseudo_class)
                    scores = reduced.score(session_x[selected])
                    _, confidence = _confidence_prediction(scores, reduced.class_ids)
                    pseudo_unknown_confidence.append(confidence)
                thresholds = calibrate_rejection_threshold(
                    known_confidence, np.concatenate(pseudo_unknown_confidence), target_known_acceptance=0.95
                )
                validation_scores = gallery.score(session_x[validation])
                predicted, confidence = _confidence_prediction(validation_scores, gallery.class_ids)
                true = session_y[validation]
                is_known = true != holdout
                correct = predicted == true
                metrics = {
                    mode: open_set_metrics(confidence, is_known, correct, threshold=float(threshold))
                    for mode, threshold in (
                        ("balanced", thresholds["balanced_threshold"]),
                        ("known_acceptance_95", thresholds["known_acceptance_threshold"]),
                    )
                }
                post_enrollment: dict[str, object] | None = None
                if method.startswith("knn_"):
                    shot_results: dict[str, object] = {}
                    support_indices = np.flatnonzero(support)
                    for shot in (1, 3, 5):
                        draws: list[dict[str, object]] = []
                        for draw in range(20):
                            rng = np.random.default_rng(_draw_seed(20260805, holdout, shot, draw))
                            selected = np.sort(rng.choice(support_indices, size=shot, replace=False))
                            enrolled_x = np.concatenate((session_x[train], session_x[selected]))
                            enrolled_y = np.concatenate((session_y[train], session_y[selected]))
                            enrolled_gallery = DensityGallery(
                                enrolled_x, enrolled_y, seen_classes + [holdout], method
                            )
                            enrolled_scores = enrolled_gallery.score(session_x[validation])
                            enrolled_predicted, _ = _confidence_prediction(
                                enrolled_scores, enrolled_gallery.class_ids
                            )
                            post = _post_enrollment_metrics(
                                session_y[validation], enrolled_predicted, holdout
                            )
                            draws.append(
                                {
                                    "draw": draw,
                                    "support_sessions": session_ids[selected].tolist(),
                                    **post,
                                }
                            )
                        shot_results[str(shot)] = {
                            "draws": draws,
                            "enrollment_h_mean": float(np.mean([row["enrollment_h"] for row in draws])),
                            "enrollment_h_min": float(np.min([row["enrollment_h"] for row in draws])),
                            "enrolled_recall_mean": float(np.mean([row["enrolled_recall"] for row in draws])),
                            "base_accuracy_mean": float(np.mean([row["base_accuracy"] for row in draws])),
                        }
                    post_enrollment = shot_results
                results.append(
                    {
                        "ablation": ablation,
                        "holdout_class_id": holdout,
                        "holdout_class": CLASS_NAMES[holdout],
                        "method": method,
                        "calibration": thresholds,
                        "metrics": metrics,
                        "post_enrollment": post_enrollment,
                    }
                )
                print(
                    f"[{ablation} {CLASS_NAMES[holdout]} {method}] "
                    f"AUROC={metrics['known_acceptance_95']['unknown_auroc']:.3f} "
                    f"H95={metrics['known_acceptance_95']['detection_h']:.3f}",
                    flush=True,
                )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[f"{row['ablation']}__{row['method']}"] .append(row)
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
        post_rows = [row for row in group if row["post_enrollment"] is not None]
        if post_rows:
            summary[key].update(
                {
                    "post_session_h_1shot_mean": float(np.mean([row["post_enrollment"]["1"]["enrollment_h_mean"] for row in post_rows])),
                    "post_session_h_5shot_mean": float(np.mean([row["post_enrollment"]["5"]["enrollment_h_mean"] for row in post_rows])),
                    "post_session_h_5shot_worst_draw": float(np.min([row["post_enrollment"]["5"]["enrollment_h_min"] for row in post_rows])),
                }
            )
    ranking = sorted(
        summary,
        key=lambda key: (summary[key]["detection_h_mean"], summary[key]["detection_h_worst_holdout"]),
        reverse=True,
    )
    payload = {
        "protocol": "session-density open-set development on frozen held-out-class embeddings",
        "final_query_used": False,
        "summary": summary,
        "ranking": ranking,
        "fold_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
