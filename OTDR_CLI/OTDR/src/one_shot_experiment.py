from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from .model_functions.multi_similarity_siamese import MultiSimilaritySiamese
from .model_functions.zero_shot import require_cuda
from .one_shot_data import OneShotSplit, build_one_shot_split, sample_class_references
from .one_shot_gallery import (
    ReferenceGallery,
    ScoreNormalizer,
    baseline_scores_against_gallery,
    calibrate_unknown_threshold,
    classify_from_pair_scores,
    fit_score_normalizer,
    pair_scores_against_gallery,
)
from .one_shot_training import (
    OneShotTrainingConfig,
    config_dict,
    detection_metrics,
    encode_traces,
    one_shot_classification_metrics,
    train_multi_similarity_model,
)
from .zero_shot_data import INPUT_COLUMNS, OuterFold, build_outer_fold, file_sha256
from .zero_shot_training import fit_seen_scaler, gpu_metadata, save_json, transform_frame


METHODS = ("learned", "cosine_1nn", "euclidean_1nn")
REGIMES = ("uniform_one_reference", "operational_seen_rich")
OPERATING_POINTS = ("balanced", "normal_far")


def _transform_features(frame: pd.DataFrame, scaler: StandardScaler) -> torch.Tensor:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required OTDR columns: {missing}")
    values = scaler.transform(frame[INPUT_COLUMNS].to_numpy(dtype=np.float32)).astype(np.float32, copy=True)
    return torch.from_numpy(values)


def _save_scaler(path: Path, scaler: StandardScaler) -> None:
    save_json(
        path,
        {
            "feature_names": INPUT_COLUMNS,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_features_in": int(scaler.n_features_in_),
        },
    )


def load_scaler(path: Path) -> StandardScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["feature_names"] != INPUT_COLUMNS:
        raise ValueError("Saved scaler feature contract does not match OTDR inputs.")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=float)
    scaler.scale_ = np.asarray(payload["scale"], dtype=float)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = int(payload["n_features_in"])
    return scaler


def _manifest(outer: OuterFold, split: OneShotSplit) -> dict[str, object]:
    def entry(frame: pd.DataFrame) -> dict[str, object]:
        groups = sorted(frame["_input_group"].unique())
        return {
            "rows": int(len(frame)),
            "classes": {str(int(k)): int(v) for k, v in frame["Class"].value_counts().sort_index().items()},
            "group_count": len(groups),
            "groups_sha256": hashlib.sha256("\n".join(groups).encode()).hexdigest(),
        }

    return {
        "holdout": list(outer.holdout),
        "feature_names": INPUT_COLUMNS,
        "train": entry(outer.train),
        "validation": entry(outer.validation),
        "seen_test": entry(outer.seen_test),
        "support_pool": entry(split.support_pool),
        "unseen_query": entry(split.query),
    }


def _gallery(
    model: MultiSimilaritySiamese,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    *,
    class_ids: Sequence[int],
    references_per_class: int,
    seed: int,
    device: torch.device,
) -> ReferenceGallery:
    references = sample_class_references(
        frame,
        references_per_class=references_per_class,
        seed=seed,
        class_ids=list(class_ids),
    )
    embeddings = encode_traces(model, _transform_features(references, scaler), device=device)
    return ReferenceGallery(
        embeddings=embeddings,
        labels=torch.from_numpy(references["Class"].to_numpy(dtype=np.int64, copy=True)),
        row_indices=torch.from_numpy(references["_source_index"].to_numpy(dtype=np.int64, copy=True)),
    )


def _score_matrix(
    model: MultiSimilaritySiamese,
    embeddings: torch.Tensor,
    gallery: ReferenceGallery,
    *,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    if method == "learned":
        return pair_scores_against_gallery(model, embeddings, gallery, device=device)
    return baseline_scores_against_gallery(embeddings, gallery, method=method)


def _raw_confidence(scores: torch.Tensor, labels: torch.Tensor, *, top_k: int) -> torch.Tensor:
    _, confidence, _ = classify_from_pair_scores(scores, labels, threshold=-float("inf"), top_k=top_k)
    return confidence


def _classify_normalized(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalizer: ScoreNormalizer,
    threshold: float,
    top_k: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    predicted, confidence, _ = classify_from_pair_scores(scores, labels, threshold=-float("inf"), top_k=top_k)
    normalized = normalizer.transform(confidence)
    accepted = normalized >= threshold
    predicted = predicted.masked_fill(~torch.from_numpy(accepted), -1)
    return predicted, normalized, accepted


def _crossfit_calibration(
    outer: OuterFold,
    scaler: StandardScaler,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    config: OneShotTrainingConfig,
    rich_references_per_class: int,
    top_k: int,
    device: torch.device,
) -> dict[str, object]:
    seen_ids = sorted(set(range(8)) - set(outer.holdout))
    pseudo_unknowns = [class_id for class_id in seen_ids if class_id != 0]
    collected = {
        regime: {method: {"scores": [], "known": [], "normal": [], "fold_normalizers": []} for method in METHODS}
        for regime in REGIMES
    }
    training_rows: list[dict[str, object]] = []
    calibration_config = replace(
        config,
        epochs=min(config.epochs, config.calibration_epochs),
        pair_count=config.calibration_pair_count,
        validation_pair_count=min(config.validation_pair_count, config.calibration_pair_count),
    )
    labels = validation_y.numpy()
    for pseudo_unknown in pseudo_unknowns:
        print(f"[CAL] leave-one-fault-out class {pseudo_unknown} on {device}", flush=True)
        inner_seen = [class_id for class_id in seen_ids if class_id != pseudo_unknown]
        inner_train_mask = train_y != pseudo_unknown
        inner_validation_mask = validation_y != pseudo_unknown
        fold_config = replace(calibration_config, seed=config.seed + pseudo_unknown * 1009)
        model, training = train_multi_similarity_model(
            train_x[inner_train_mask],
            train_y[inner_train_mask],
            validation_x[inner_validation_mask],
            validation_y[inner_validation_mask],
            device=device,
            config=fold_config,
        )
        training_rows.append({"pseudo_unknown_class": pseudo_unknown, "training": training})
        validation_embeddings = encode_traces(model, validation_x, device=device)
        inner_frame = outer.train[outer.train["Class"] != pseudo_unknown]
        galleries = {
            "uniform_one_reference": _gallery(
                model, inner_frame, scaler, class_ids=inner_seen, references_per_class=1,
                seed=config.seed, device=device,
            ),
            "operational_seen_rich": _gallery(
                model, inner_frame, scaler, class_ids=inner_seen,
                references_per_class=rich_references_per_class, seed=config.seed, device=device,
            ),
        }
        is_known = labels != pseudo_unknown
        for regime, gallery in galleries.items():
            for method in METHODS:
                method_top_k = top_k if method == "learned" else 1
                scores = _score_matrix(model, validation_embeddings, gallery, method=method, device=device)
                confidence = _raw_confidence(scores, gallery.labels, top_k=method_top_k)
                normalizer = fit_score_normalizer(confidence.numpy()[is_known])
                normalized = normalizer.transform(confidence)
                bucket = collected[regime][method]
                bucket["scores"].append(normalized)
                bucket["known"].append(is_known.copy())
                bucket["normal"].append(normalized[labels == 0])
                bucket["fold_normalizers"].append(
                    {"pseudo_unknown_class": pseudo_unknown, **normalizer.to_dict()}
                )
        del model
        torch.cuda.empty_cache()
    entries: dict[str, dict[str, object]] = {regime: {} for regime in REGIMES}
    for regime in REGIMES:
        for method in METHODS:
            bucket = collected[regime][method]
            scores = np.concatenate(bucket["scores"])
            known = np.concatenate(bucket["known"])
            normal = np.concatenate(bucket["normal"])
            calibration = calibrate_unknown_threshold(scores, known, normal_scores=normal)
            entries[regime][method] = {
                "crossfit": asdict(calibration),
                "fold_normalizers": bucket["fold_normalizers"],
            }
    return {
        "schema_version": 2,
        "normalization": "known-validation median/IQR within each cross-fit model",
        "pseudo_unknown_classes": pseudo_unknowns,
        "calibration_training_config": config_dict(calibration_config),
        "training": training_rows,
        "entries": entries,
    }


def _fit_final_normalizers(
    model: MultiSimilaritySiamese,
    validation_embeddings: torch.Tensor,
    validation_labels: np.ndarray,
    galleries: dict[str, ReferenceGallery],
    calibration: dict[str, object],
    *,
    top_k: int,
    device: torch.device,
) -> None:
    entries = calibration["entries"]
    for regime, gallery in galleries.items():
        for method in METHODS:
            method_top_k = top_k if method == "learned" else 1
            scores = _score_matrix(model, validation_embeddings, gallery, method=method, device=device)
            confidence = _raw_confidence(scores, gallery.labels, top_k=method_top_k)
            normalizer = fit_score_normalizer(confidence)
            entries[regime][method]["final_normalizer"] = normalizer.to_dict()
            normalized = normalizer.transform(confidence)
            normal_scores = normalized[validation_labels == 0]
            entries[regime][method]["final_normal_far_threshold"] = float(
                np.quantile(normal_scores, 0.01)
            )


def _selected_support_gallery(
    support_frame: pd.DataFrame,
    support_embeddings: torch.Tensor,
    *,
    holdout: Sequence[int],
    seed: int,
) -> tuple[ReferenceGallery, list[int]]:
    selected = sample_class_references(
        support_frame, references_per_class=1, seed=seed, class_ids=list(holdout)
    )
    positions = {int(index): position for position, index in enumerate(support_frame.index)}
    embedding_rows: list[torch.Tensor] = []
    labels: list[int] = []
    row_indices: list[int] = []
    for _, row in selected.iterrows():
        index = int(row["_source_index"])
        embedding_rows.append(support_embeddings[positions[index]])
        labels.append(int(row["Class"]))
        row_indices.append(index)
    return ReferenceGallery(
        embeddings=torch.stack(embedding_rows),
        labels=torch.tensor(labels, dtype=torch.long),
        row_indices=torch.tensor(row_indices, dtype=torch.long),
    ), row_indices


def _aggregate_draws(draws: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for metric in (
        "accuracy", "balanced_accuracy", "seen_accuracy", "unseen_accuracy", "harmonic_mean", "rejection_rate"
    ):
        values = np.asarray([row[metric] for row in draws], dtype=float)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def _evaluate_method(
    model: MultiSimilaritySiamese,
    base_gallery: ReferenceGallery,
    *,
    method: str,
    calibration_entry: dict[str, object],
    combined_embeddings: torch.Tensor,
    combined_labels: np.ndarray,
    is_known: np.ndarray,
    support_frame: pd.DataFrame,
    support_embeddings: torch.Tensor,
    holdout: tuple[int, int],
    seen_ids: Sequence[int],
    support_draws: int,
    seed: int,
    top_k: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    method_top_k = top_k if method == "learned" else 1
    normalizer = ScoreNormalizer.from_dict(calibration_entry["final_normalizer"])
    crossfit = calibration_entry["crossfit"]
    thresholds = {
        "balanced": float(crossfit["threshold"]),
        "normal_far": float(calibration_entry["final_normal_far_threshold"]),
    }
    base_scores = _score_matrix(model, combined_embeddings, base_gallery, method=method, device=device)
    raw_confidence = _raw_confidence(base_scores, base_gallery.labels, top_k=method_top_k)
    normalized_confidence = normalizer.transform(raw_confidence)
    operating_points: dict[str, object] = {}
    draw_buckets: dict[str, list[dict[str, object]]] = {mode: [] for mode in OPERATING_POINTS}
    for mode, threshold in thresholds.items():
        accepted = normalized_confidence >= threshold
        operating_points[mode] = {
            "threshold_normalized": threshold,
            "pre_enrollment_detection": detection_metrics(
                is_known=is_known,
                confidence=normalized_confidence,
                accepted=accepted,
                true_labels=combined_labels,
            ),
        }
    for draw in range(support_draws):
        support_gallery, selected_rows = _selected_support_gallery(
            support_frame,
            support_embeddings,
            holdout=holdout,
            seed=seed + draw * 1009,
        )
        support_scores = _score_matrix(
            model, combined_embeddings, support_gallery, method=method, device=device
        )
        enrolled_scores = torch.cat((base_scores, support_scores), dim=1)
        enrolled_labels = torch.cat((base_gallery.labels, support_gallery.labels))
        for mode, threshold in thresholds.items():
            predicted, confidence, accepted = _classify_normalized(
                enrolled_scores,
                enrolled_labels,
                normalizer=normalizer,
                threshold=threshold,
                top_k=method_top_k,
            )
            metrics = one_shot_classification_metrics(
                y_true=combined_labels,
                y_pred=predicted.numpy(),
                seen_class_ids=seen_ids,
                unseen_class_ids=holdout,
            )
            draw_buckets[mode].append(
                {
                    "draw": draw,
                    "support_row_indices": selected_rows,
                    **metrics,
                    "mean_normalized_confidence": float(np.mean(confidence)),
                    "acceptance_rate": float(np.mean(accepted)),
                }
            )
    for mode in OPERATING_POINTS:
        operating_points[mode]["post_enrollment"] = {
            "draws": draw_buckets[mode],
            "aggregate": _aggregate_draws(draw_buckets[mode]),
        }
    diagnostic = {
        "method": method,
        "confidence": normalized_confidence,
        "is_known": is_known,
        "true_class": combined_labels,
    }
    return {"operating_points": operating_points}, diagnostic


def _save_diagnostics(fold_dir: Path, diagnostics: list[dict[str, object]]) -> None:
    rows = []
    for record in diagnostics:
        rows.append(
            pd.DataFrame(
                {
                    "regime": record["regime"],
                    "method": record["method"],
                    "normalized_confidence": record["confidence"],
                    "is_known": record["is_known"],
                    "true_class": record["true_class"],
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(
        fold_dir / "detection_scores.csv.gz", index=False, compression="gzip"
    )
    fig, axes = plt.subplots(len(REGIMES), len(METHODS), figsize=(15, 8), constrained_layout=True)
    for row, regime in enumerate(REGIMES):
        for column, method in enumerate(METHODS):
            record = next(item for item in diagnostics if item["regime"] == regime and item["method"] == method)
            scores = np.asarray(record["confidence"])
            known = np.asarray(record["is_known"])
            axes[row, column].hist(scores[known], bins=50, alpha=0.6, density=True, label="known")
            axes[row, column].hist(scores[~known], bins=50, alpha=0.6, density=True, label="unknown")
            axes[row, column].set_title(f"{regime}\n{method}")
            axes[row, column].legend()
    fig.savefig(fold_dir / "score_histograms.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(REGIMES), len(METHODS), figsize=(15, 8), constrained_layout=True)
    for row, regime in enumerate(REGIMES):
        for column, method in enumerate(METHODS):
            record = next(item for item in diagnostics if item["regime"] == regime and item["method"] == method)
            fpr, tpr, _ = roc_curve(record["is_known"], record["confidence"])
            axes[row, column].plot(fpr, tpr)
            axes[row, column].plot([0, 1], [0, 1], linestyle="--", color="gray")
            axes[row, column].set_title(f"{regime}\n{method}")
            axes[row, column].set_xlabel("unknown false acceptance")
            axes[row, column].set_ylabel("known acceptance")
    fig.savefig(fold_dir / "roc_curves.png", dpi=150)
    plt.close(fig)


def run_crossfit_fold(
    *,
    data_path: Path,
    out_dir: Path,
    holdout: tuple[int, int],
    device: torch.device,
    config: OneShotTrainingConfig,
    support_fraction: float = 0.2,
    support_draws: int = 20,
    top_k: int = 3,
    rich_references_per_class: int = 20,
) -> dict[str, object]:
    device = require_cuda(str(device))
    torch.cuda.reset_peak_memory_stats(device)
    frame = pd.read_csv(data_path)
    outer = build_outer_fold(frame, holdout=holdout, seed=config.seed)
    split = build_one_shot_split(outer, support_fraction=support_fraction, seed=config.seed)
    scaler = fit_seen_scaler(outer)
    train_x, train_y = transform_frame(outer.train, scaler)
    validation_x, validation_y = transform_frame(outer.validation, scaler)
    calibration = _crossfit_calibration(
        outer,
        scaler,
        train_x,
        train_y,
        validation_x,
        validation_y,
        config=config,
        rich_references_per_class=rich_references_per_class,
        top_k=top_k,
        device=device,
    )
    print(f"[TRAIN] final fold {holdout[0]}-{holdout[1]} on {device}", flush=True)
    model, training = train_multi_similarity_model(
        train_x, train_y, validation_x, validation_y, device=device, config=config
    )
    seen_ids = sorted(set(range(8)) - set(holdout))
    galleries = {
        "uniform_one_reference": _gallery(
            model, outer.train, scaler, class_ids=seen_ids, references_per_class=1,
            seed=config.seed, device=device,
        ),
        "operational_seen_rich": _gallery(
            model, outer.train, scaler, class_ids=seen_ids,
            references_per_class=rich_references_per_class, seed=config.seed, device=device,
        ),
    }
    validation_embeddings = encode_traces(model, validation_x, device=device)
    _fit_final_normalizers(
        model,
        validation_embeddings,
        validation_y.numpy(),
        galleries,
        calibration,
        top_k=top_k,
        device=device,
    )
    seen_x, seen_y = transform_frame(outer.seen_test, scaler)
    query_x, query_y = transform_frame(split.query, scaler)
    support_x, _ = transform_frame(split.support_pool, scaler)
    seen_embeddings = encode_traces(model, seen_x, device=device)
    query_embeddings = encode_traces(model, query_x, device=device)
    support_embeddings = encode_traces(model, support_x, device=device)
    combined_embeddings = torch.cat((seen_embeddings, query_embeddings))
    combined_labels = np.concatenate((seen_y.numpy(), query_y.numpy()))
    is_known = np.concatenate(
        (np.ones(len(seen_y), dtype=bool), np.zeros(len(query_y), dtype=bool))
    )
    metrics: dict[str, object] = {
        "schema_version": 2,
        "holdout": list(holdout),
        "regimes": {},
    }
    diagnostics: list[dict[str, object]] = []
    for regime, gallery in galleries.items():
        methods: dict[str, object] = {}
        for method in METHODS:
            print(f"[EVAL] {regime} / {method}", flush=True)
            result, diagnostic = _evaluate_method(
                model,
                gallery,
                method=method,
                calibration_entry=calibration["entries"][regime][method],
                combined_embeddings=combined_embeddings,
                combined_labels=combined_labels,
                is_known=is_known,
                support_frame=split.support_pool,
                support_embeddings=support_embeddings,
                holdout=holdout,
                seen_ids=seen_ids,
                support_draws=support_draws,
                seed=config.seed,
                top_k=top_k,
                device=device,
            )
            methods[method] = result
            diagnostics.append({"regime": regime, **diagnostic})
        metrics["regimes"][regime] = {"methods": methods}
    fold_dir = out_dir / f"fold_{holdout[0]:02d}_{holdout[1]:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "checkpoint.pt")
    galleries["uniform_one_reference"].save(fold_dir / "gallery_uniform.pt")
    galleries["operational_seen_rich"].save(fold_dir / "gallery_seen_rich.pt")
    _save_scaler(fold_dir / "scaler.json", scaler)
    save_json(fold_dir / "metrics.json", metrics)
    save_json(fold_dir / "calibration.json", calibration)
    save_json(fold_dir / "split_manifest.json", _manifest(outer, split))
    _save_diagnostics(fold_dir, diagnostics)
    artifact_names = (
        "checkpoint.pt", "gallery_uniform.pt", "gallery_seen_rich.pt", "scaler.json",
        "metrics.json", "calibration.json", "split_manifest.json", "detection_scores.csv.gz",
        "score_histograms.png", "roc_curves.png",
    )
    metadata = {
        "schema_version": 2,
        "method": "crossfit_normalized_multi_similarity_one_shot",
        "holdout": list(holdout),
        "seen_class_ids": seen_ids,
        "feature_names": INPUT_COLUMNS,
        "feature_signature": hashlib.sha256(json.dumps(INPUT_COLUMNS).encode()).hexdigest(),
        "forbidden_features_disabled": True,
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": file_sha256(data_path),
        "training_config": config_dict(config),
        "training": training,
        "support_fraction": support_fraction,
        "support_draws": support_draws,
        "top_k": top_k,
        "rich_references_per_class": rich_references_per_class,
        "methods": list(METHODS),
        "operating_points": list(OPERATING_POINTS),
        "gpu": gpu_metadata(device),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "artifacts_sha256": {name: file_sha256(fold_dir / name) for name in artifact_names},
    }
    save_json(fold_dir / "metadata.json", metadata)
    return metrics


def summarize_crossfit_benchmark(results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"schema_version": 2, "fold_count": len(results), "configurations": {}}
    for regime in REGIMES:
        summary["configurations"][regime] = {}
        for method in METHODS:
            summary["configurations"][regime][method] = {}
            for operating_point in OPERATING_POINTS:
                points = [
                    row["regimes"][regime]["methods"][method]["operating_points"][operating_point]
                    for row in results
                ]
                metrics = {
                    "detection_auroc": [p["pre_enrollment_detection"]["known_unknown_auroc"] for p in points],
                    "detection_aupr": [p["pre_enrollment_detection"]["known_unknown_aupr"] for p in points],
                    "known_acceptance": [p["pre_enrollment_detection"]["known_acceptance"] for p in points],
                    "unknown_recall": [p["pre_enrollment_detection"]["unknown_recall"] for p in points],
                    "normal_rejection_rate": [p["pre_enrollment_detection"]["normal_rejection_rate"] for p in points],
                    "post_harmonic_mean": [p["post_enrollment"]["aggregate"]["harmonic_mean"]["mean"] for p in points],
                    "post_unseen_accuracy": [p["post_enrollment"]["aggregate"]["unseen_accuracy"]["mean"] for p in points],
                }
                summary["configurations"][regime][method][operating_point] = {
                    name: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                    for name, values in metrics.items()
                }
    summary["folds"] = results
    return summary


def write_benchmark_tables(out_dir: Path, summary: dict[str, object]) -> None:
    configuration_rows: list[dict[str, object]] = []
    for regime, methods in summary["configurations"].items():
        for method, operating_points in methods.items():
            for operating_point, metrics in operating_points.items():
                row: dict[str, object] = {
                    "regime": regime,
                    "method": method,
                    "operating_point": operating_point,
                }
                for metric, distribution in metrics.items():
                    for statistic, value in distribution.items():
                        row[f"{metric}_{statistic}"] = value
                configuration_rows.append(row)
    pd.DataFrame(configuration_rows).to_csv(out_dir / "benchmark_summary.csv", index=False)
    per_fold_rows: list[dict[str, object]] = []
    per_fault_rows: list[dict[str, object]] = []
    for fold in summary["folds"]:
        holdout = "-".join(str(value) for value in fold["holdout"])
        for regime in REGIMES:
            for method in METHODS:
                for operating_point in OPERATING_POINTS:
                    point = fold["regimes"][regime]["methods"][method]["operating_points"][operating_point]
                    detection = point["pre_enrollment_detection"]
                    aggregate = point["post_enrollment"]["aggregate"]
                    per_fold_rows.append(
                        {
                            "holdout": holdout,
                            "regime": regime,
                            "method": method,
                            "operating_point": operating_point,
                            **{f"detection_{key}": value for key, value in detection.items() if isinstance(value, (int, float))},
                            **{f"post_{key}": value["mean"] for key, value in aggregate.items()},
                        }
                    )
                    for fault in fold["holdout"]:
                        values = [
                            draw["per_class_accuracy"][str(fault)]
                            for draw in point["post_enrollment"]["draws"]
                        ]
                        per_fault_rows.append(
                            {
                                "holdout": holdout,
                                "fault_class": fault,
                                "regime": regime,
                                "method": method,
                                "operating_point": operating_point,
                                "mean_accuracy": float(np.mean(values)),
                                "std_accuracy": float(np.std(values)),
                            }
                        )
    pd.DataFrame(per_fold_rows).to_csv(out_dir / "benchmark_per_fold.csv", index=False)
    pd.DataFrame(per_fault_rows).to_csv(out_dir / "benchmark_per_fault.csv", index=False)


def load_fold_artifacts(
    fold_dir: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], MultiSimilaritySiamese, StandardScaler]:
    metadata = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError("This command requires a cross-fitted schema-version 2 fold.")
    for name, expected in metadata["artifacts_sha256"].items():
        if file_sha256(fold_dir / name) != expected:
            raise ValueError(f"Saved artifact hash does not match metadata: {name}")
    training_config = metadata["training_config"]
    model = MultiSimilaritySiamese(
        embedding_dim=int(training_config["embedding_dim"]),
        dropout=float(training_config["dropout"]),
        similarity_mode=str(training_config["similarity_mode"]),
    ).to(device)
    model.load_state_dict(torch.load(fold_dir / "checkpoint.pt", map_location=device, weights_only=True))
    model.eval()
    calibration = json.loads((fold_dir / "calibration.json").read_text(encoding="utf-8"))
    return metadata, calibration, model, load_scaler(fold_dir / "scaler.json")


def recompute_saved_configuration(
    *,
    fold_dir: Path,
    data_path: Path,
    regime: str,
    method: str,
    operating_point: str,
    device: torch.device,
    include_one_shot: bool,
) -> dict[str, object]:
    if regime not in REGIMES or method not in METHODS or operating_point not in OPERATING_POINTS:
        raise ValueError("Unknown evaluation configuration.")
    metadata, calibration, model, scaler = load_fold_artifacts(fold_dir, device=device)
    if file_sha256(data_path) != metadata["dataset_sha256"]:
        raise ValueError("Dataset hash does not match saved fold metadata.")
    frame = pd.read_csv(data_path)
    holdout = tuple(int(value) for value in metadata["holdout"])
    seed = int(metadata["training_config"]["seed"])
    outer = build_outer_fold(frame, holdout=holdout, seed=seed)
    split = build_one_shot_split(
        outer, support_fraction=float(metadata["support_fraction"]), seed=seed
    )
    seen_x, seen_y = transform_frame(outer.seen_test, scaler)
    query_x, query_y = transform_frame(split.query, scaler)
    support_x, _ = transform_frame(split.support_pool, scaler)
    combined_embeddings = torch.cat(
        (encode_traces(model, seen_x, device=device), encode_traces(model, query_x, device=device))
    )
    support_embeddings = encode_traces(model, support_x, device=device)
    combined_labels = np.concatenate((seen_y.numpy(), query_y.numpy()))
    is_known = np.concatenate((np.ones(len(seen_y), bool), np.zeros(len(query_y), bool)))
    gallery_name = "gallery_uniform.pt" if regime == "uniform_one_reference" else "gallery_seen_rich.pt"
    gallery = ReferenceGallery.load(fold_dir / gallery_name)
    result, _ = _evaluate_method(
        model,
        gallery,
        method=method,
        calibration_entry=calibration["entries"][regime][method],
        combined_embeddings=combined_embeddings,
        combined_labels=combined_labels,
        is_known=is_known,
        support_frame=split.support_pool,
        support_embeddings=support_embeddings,
        holdout=holdout,
        seen_ids=metadata["seen_class_ids"],
        support_draws=int(metadata["support_draws"]) if include_one_shot else 1,
        seed=seed,
        top_k=int(metadata["top_k"]),
        device=device,
    )
    selected = result["operating_points"][operating_point]
    if not include_one_shot:
        selected = {"threshold_normalized": selected["threshold_normalized"], "pre_enrollment_detection": selected["pre_enrollment_detection"]}
    return selected


def classify_saved_frame(
    frame: pd.DataFrame,
    *,
    fold_dir: Path,
    gallery_path: Path | None,
    regime: str,
    method: str,
    operating_point: str,
    device: torch.device,
) -> pd.DataFrame:
    metadata, calibration, model, scaler = load_fold_artifacts(fold_dir, device=device)
    gallery_name = "gallery_uniform.pt" if regime == "uniform_one_reference" else "gallery_seen_rich.pt"
    gallery = ReferenceGallery.load(gallery_path or fold_dir / gallery_name)
    embeddings = encode_traces(model, _transform_features(frame, scaler), device=device)
    scores = _score_matrix(model, embeddings, gallery, method=method, device=device)
    entry = calibration["entries"][regime][method]
    normalizer = ScoreNormalizer.from_dict(entry["final_normalizer"])
    threshold = (
        float(entry["crossfit"]["threshold"])
        if operating_point == "balanced"
        else float(entry["final_normal_far_threshold"])
    )
    predicted, confidence, accepted = _classify_normalized(
        scores,
        gallery.labels,
        normalizer=normalizer,
        threshold=threshold,
        top_k=int(metadata["top_k"]) if method == "learned" else 1,
    )
    return pd.DataFrame(
        {
            "predicted_class": predicted.numpy(),
            "normalized_confidence": confidence,
            "accepted_as_known": accepted,
            "method": method,
            "gallery_regime": regime,
            "operating_point": operating_point,
        }
    )


def encode_saved_references(
    frame: pd.DataFrame,
    *,
    fold_dir: Path,
    device: torch.device,
) -> torch.Tensor:
    _, _, model, scaler = load_fold_artifacts(fold_dir, device=device)
    return encode_traces(model, _transform_features(frame, scaler), device=device)
