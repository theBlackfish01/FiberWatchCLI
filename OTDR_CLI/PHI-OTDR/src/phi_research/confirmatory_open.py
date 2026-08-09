"""Locked multi-seed confirmatory evaluation on untouched final-query sessions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .density_open_analysis import DensityGallery, _confidence_prediction
from .embedding_gallery import FeatureEncoder, _encode, _feature_masks
from .gallery_baseline import _draw_seed, _post_enrollment_metrics, _session_prototypes
from .metrics import open_set_metrics


SEEDS = (20260805, 20260806, 20260807)
SHOTS = (1, 3, 5)


def _combine_bundles(development_path: Path, final_path: Path) -> dict[str, np.ndarray]:
    development = np.load(development_path, allow_pickle=False)
    final = np.load(final_path, allow_pickle=False)
    if not np.array_equal(development["feature_names"], final["feature_names"]):
        raise ValueError("Development and final feature schemas disagree")
    return {
        "features": np.concatenate((development["features"], final["features"])),
        "labels": np.concatenate((development["labels"], final["labels"])),
        "sessions": np.concatenate((development["sessions"], final["sessions"])).astype(str),
        "partitions": np.concatenate((development["partitions"], final["partitions"])).astype(str),
        "feature_names": development["feature_names"].astype(str),
    }


def _bootstrap_class_ci(values: list[float], rng: np.random.Generator, draws: int = 10000) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    bootstrap = np.mean(rng.choice(array, size=(draws, len(array)), replace=True), axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "worst_holdout": float(np.min(array)),
        "best_holdout": float(np.max(array)),
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    rng = np.random.default_rng(20260805)
    summary: dict[str, object] = {}
    for method in ("raw_control", "frozen_embedding"):
        selected = [row for row in rows if row["method"] == method]
        by_holdout: dict[str, list[dict[str, object]]] = defaultdict(list)
        by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            by_holdout[str(row["holdout_class"])].append(row)
            by_seed[int(row["seed"])].append(row)
        method_summary: dict[str, object] = {"pre_enrollment": {}, "post_enrollment": {}}
        for metric in (
            "unknown_auroc", "unknown_aupr", "known_acceptance", "unknown_recall",
            "detection_h", "oscr", "known_classification_accuracy",
        ):
            holdout_values = [
                float(np.mean([row["pre_enrollment"][metric] for row in by_holdout[class_name]]))
                for class_name in CLASS_NAMES
            ]
            metric_summary = _bootstrap_class_ci(holdout_values, rng)
            seed_means = [
                float(np.mean([row["pre_enrollment"][metric] for row in by_seed[seed]]))
                for seed in SEEDS
            ]
            metric_summary["seed_means"] = {str(seed): value for seed, value in zip(SEEDS, seed_means)}
            metric_summary["seed_standard_deviation"] = float(np.std(seed_means, ddof=1))
            method_summary["pre_enrollment"][metric] = metric_summary
        for shot in SHOTS:
            shot_summary: dict[str, object] = {}
            for metric in ("enrollment_h_mean", "base_accuracy_mean", "enrolled_recall_mean"):
                holdout_values = [
                    float(np.mean([row["post_enrollment"][str(shot)][metric] for row in by_holdout[class_name]]))
                    for class_name in CLASS_NAMES
                ]
                shot_summary[metric] = _bootstrap_class_ci(holdout_values, rng)
            shot_summary["worst_single_draw_h"] = float(
                np.min([row["post_enrollment"][str(shot)]["enrollment_h_min"] for row in selected])
            )
            method_summary["post_enrollment"][str(shot)] = shot_summary
        method_summary["per_holdout"] = {
            class_name: {
                "pre_detection_h_mean": float(np.mean([row["pre_enrollment"]["detection_h"] for row in by_holdout[class_name]])),
                "pre_unknown_recall_mean": float(np.mean([row["pre_enrollment"]["unknown_recall"] for row in by_holdout[class_name]])),
                "post_5shot_h_mean": float(np.mean([row["post_enrollment"]["5"]["enrollment_h_mean"] for row in by_holdout[class_name]])),
            }
            for class_name in CLASS_NAMES
        }
        summary[method] = method_summary

    paired: dict[str, object] = {}
    for metric, source in (
        ("pre_detection_h", "pre"),
        ("post_5shot_h", "post"),
    ):
        primary: list[float] = []
        control: list[float] = []
        for class_name in CLASS_NAMES:
            primary_rows = [row for row in rows if row["method"] == "frozen_embedding" and row["holdout_class"] == class_name]
            control_rows = [row for row in rows if row["method"] == "raw_control" and row["holdout_class"] == class_name]
            if source == "pre":
                primary.append(float(np.mean([row["pre_enrollment"]["detection_h"] for row in primary_rows])))
                control.append(float(np.mean([row["pre_enrollment"]["detection_h"] for row in control_rows])))
            else:
                primary.append(float(np.mean([row["post_enrollment"]["5"]["enrollment_h_mean"] for row in primary_rows])))
                control.append(float(np.mean([row["post_enrollment"]["5"]["enrollment_h_mean"] for row in control_rows])))
        differences = np.asarray(primary) - np.asarray(control)
        test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        paired[metric] = {
            "primary_holdout_means": primary,
            "control_holdout_means": control,
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "primary_win_fraction": float(np.mean(differences > 0)),
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_pvalue": float(test.pvalue),
            "unit": "held-out class after averaging encoder seeds",
        }
    summary["paired_tests"] = paired
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--final-features", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-draws", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    bundle = _combine_bundles(args.development_features, args.final_features)
    x = bundle["features"]
    y = bundle["labels"]
    sessions = bundle["sessions"]
    partitions = bundle["partitions"]
    names = bundle["feature_names"]
    full_mask = _feature_masks(names)["full"]
    results: list[dict[str, object]] = []

    for seed in SEEDS:
        for holdout in range(len(CLASS_NAMES)):
            seen_classes = [value for value in range(len(CLASS_NAMES)) if value != holdout]
            train_windows = (partitions == "train") & np.isin(y, seen_classes)
            for method in ("raw_control", "frozen_embedding"):
                if method == "raw_control":
                    scaler = StandardScaler().fit(x[train_windows][:, full_mask])
                    represented = scaler.transform(x[:, full_mask]).astype(np.float32)
                    pre_method = "knn_euclidean_3"
                else:
                    fold_dir = args.embedding_root / f"seed{seed}" / "full" / CLASS_NAMES[holdout]
                    scaler = joblib.load(fold_dir / "scaler.joblib")
                    transformed = scaler.transform(x[:, full_mask]).astype(np.float32)
                    model = FeatureEncoder(int(np.sum(full_mask)), len(seen_classes)).to("cuda")
                    model.load_state_dict(
                        torch.load(fold_dir / "best_model.pt", map_location="cuda", weights_only=True)
                    )
                    represented = _encode(model, transformed, torch.device("cuda"))
                    pre_method = "mahalanobis"

                session_x, session_y, session_ids = _session_prototypes(represented, y, sessions)
                session_partition = np.asarray(
                    [np.unique(partitions[sessions == session])[0] for session in session_ids]
                )
                train = (session_partition == "train") & np.isin(session_y, seen_classes)
                calibration = (session_partition == "calibration") & np.isin(session_y, seen_classes)
                support = (session_partition == "support") & (session_y == holdout)
                final_query = session_partition == "final_query"
                pre_gallery = DensityGallery(session_x[train], session_y[train], seen_classes, pre_method)
                calibration_scores = pre_gallery.score(session_x[calibration])
                _, known_calibration_confidence = _confidence_prediction(
                    calibration_scores, pre_gallery.class_ids
                )
                threshold = float(
                    np.quantile(known_calibration_confidence, 0.05, method="higher")
                )
                final_scores = pre_gallery.score(session_x[final_query])
                final_predicted, final_confidence = _confidence_prediction(
                    final_scores, pre_gallery.class_ids
                )
                final_true = session_y[final_query]
                is_known = final_true != holdout
                pre_metrics = open_set_metrics(
                    final_confidence,
                    is_known,
                    final_predicted == final_true,
                    threshold=threshold,
                )

                post: dict[str, object] = {}
                post_prediction_arrays: dict[str, np.ndarray] = {}
                support_indices = np.flatnonzero(support)
                for shot in SHOTS:
                    draws: list[dict[str, object]] = []
                    draw_predictions: list[np.ndarray] = []
                    for draw in range(args.support_draws):
                        rng = np.random.default_rng(_draw_seed(seed, holdout, shot, draw))
                        selected = np.sort(rng.choice(support_indices, size=shot, replace=False))
                        enrolled_gallery = DensityGallery(
                            np.concatenate((session_x[train], session_x[selected])),
                            np.concatenate((session_y[train], session_y[selected])),
                            seen_classes + [holdout],
                            "knn_euclidean_3",
                        )
                        scores = enrolled_gallery.score(session_x[final_query])
                        predicted, _ = _confidence_prediction(scores, enrolled_gallery.class_ids)
                        metrics = _post_enrollment_metrics(final_true, predicted, holdout)
                        draw_predictions.append(predicted)
                        draws.append(
                            {
                                "draw": draw,
                                "support_sessions": session_ids[selected].tolist(),
                                **metrics,
                            }
                        )
                    post_prediction_arrays[f"post_{shot}shot_predictions"] = np.stack(draw_predictions)
                    post[str(shot)] = {
                        "draws": draws,
                        "enrollment_h_mean": float(np.mean([row["enrollment_h"] for row in draws])),
                        "enrollment_h_min": float(np.min([row["enrollment_h"] for row in draws])),
                        "base_accuracy_mean": float(np.mean([row["base_accuracy"] for row in draws])),
                        "enrolled_recall_mean": float(np.mean([row["enrolled_recall"] for row in draws])),
                    }

                score_path = score_dir / f"{method}_seed{seed}_{CLASS_NAMES[holdout]}.npz"
                np.savez_compressed(
                    score_path,
                    session_ids=session_ids[final_query],
                    y_true=final_true,
                    is_known=is_known,
                    pre_class_labels=pre_gallery.class_ids,
                    pre_scores=final_scores,
                    pre_predictions=final_predicted,
                    pre_confidence=final_confidence,
                    calibration_confidence=known_calibration_confidence,
                    threshold=np.asarray([threshold]),
                    **post_prediction_arrays,
                )
                row = {
                    "method": method,
                    "seed": seed,
                    "holdout_class_id": holdout,
                    "holdout_class": CLASS_NAMES[holdout],
                    "pre_score_method": pre_method,
                    "post_score_method": "knn_euclidean_3",
                    "calibration_known_sessions": int(np.sum(calibration)),
                    "final_query_sessions": int(np.sum(final_query)),
                    "final_unknown_sessions": int(np.sum(final_true == holdout)),
                    "threshold": threshold,
                    "pre_enrollment": pre_metrics,
                    "post_enrollment": post,
                    "score_artifact": score_path.relative_to(args.output_dir).as_posix(),
                }
                results.append(row)
                print(
                    f"[{method} seed={seed} holdout={CLASS_NAMES[holdout]}] "
                    f"AUROC={pre_metrics['unknown_auroc']:.3f} H={pre_metrics['detection_h']:.3f} "
                    f"postH@5={post['5']['enrollment_h_mean']:.3f}",
                    flush=True,
                )

    summary = _summarize(results)
    payload = {
        "protocol": "phi_otdr_open_world_confirmatory_v1",
        "seeds": list(SEEDS),
        "support_draws": args.support_draws,
        "final_query_used": True,
        "method_selection_after_final_query": False,
        "summary": summary,
        "fold_results": results,
    }
    (args.output_dir / "confirmatory_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
