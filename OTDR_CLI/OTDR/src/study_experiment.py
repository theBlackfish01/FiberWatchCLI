from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .study_data import PreparedFold, deterministic_support_indices, feature_signature, prepare_fold
from .study_metrics import (
    balanced_threshold,
    classification_metrics,
    conformal_p_values,
    harmonic,
    macro_class_accuracy,
    open_set_metrics,
    post_enrollment_metrics,
    summarize_draws,
    threshold_at_normal_far,
)
from .study_semantics import semantic_prototypes
from .study_state import environment_metadata, file_sha256, stable_run_id, write_manifest
from .study_training import (
    ApproachAConfig,
    ApproachBConfig,
    ApproachCConfig,
    encode,
    train_approach_a,
    train_approach_b,
    train_approach_c,
)
from .zero_shot_data import INPUT_COLUMNS
from .zero_shot_training import save_json


def class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, class_ids: Sequence[int], *,
                     strategy: str = "prototype", seed: int = 42, count: int = 1) -> torch.Tensor:
    values: list[torch.Tensor] = []
    rng = np.random.default_rng(seed)
    for class_id in class_ids:
        rows = embeddings[labels == class_id]
        if not len(rows):
            raise ValueError(f"No embeddings for class {class_id}.")
        mean = torch.nn.functional.normalize(rows.mean(0), dim=0)
        if strategy == "prototype":
            selected = mean
        elif strategy == "medoid":
            candidates = rows if len(rows) <= 4096 else rows[torch.from_numpy(rng.choice(len(rows), 4096, replace=False))]
            selected = candidates[(candidates @ mean).argmax()]
        elif strategy in {"random", "equal"}:
            indices = rng.choice(len(rows), size=min(count, len(rows)), replace=False)
            selected = rows[torch.from_numpy(np.asarray(indices))].mean(0)
        else:
            raise ValueError(f"Unknown prototype strategy: {strategy}")
        values.append(torch.nn.functional.normalize(selected, dim=0))
    return torch.stack(values)


def cosine_scores(embeddings: torch.Tensor, prototypes: torch.Tensor) -> np.ndarray:
    return (torch.nn.functional.normalize(embeddings, dim=-1) @ torch.nn.functional.normalize(prototypes, dim=-1).T).numpy()


class DensityScorer:
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor, class_ids: Sequence[int], *,
                 density: str, shrinkage: float, knn_k: int) -> None:
        self.class_ids = list(class_ids)
        self.density = density
        self.means = np.stack([embeddings[labels == value].mean(0).numpy() for value in self.class_ids])
        self.knn_k = knn_k
        x = embeddings.numpy().astype(np.float64)
        if density == "mahalanobis":
            centered = np.concatenate([x[labels.numpy() == value] - self.means[index] for index, value in enumerate(self.class_ids)])
            covariance = np.cov(centered, rowvar=False)
            target = np.trace(covariance) / covariance.shape[0]
            covariance = (1 - shrinkage) * covariance + shrinkage * target * np.eye(covariance.shape[0])
            self.inverse = np.linalg.pinv(covariance, hermitian=True)
            self.references = None
        elif density == "knn":
            # A deterministic 512-reference density sketch keeps kNN tractable and
            # gives every class the same reference budget.
            max_per_class = 512
            self.references = [x[labels.numpy() == value][:max_per_class].astype(np.float32) for value in self.class_ids]
            self.inverse = None
        else:
            raise ValueError(f"Unknown density estimator: {density}")

    def distances(self, embeddings: torch.Tensor) -> np.ndarray:
        x = embeddings.numpy().astype(np.float64)
        if self.density == "mahalanobis":
            rows = []
            for mean in self.means:
                delta = x - mean
                rows.append(np.einsum("bi,ij,bj->b", delta, self.inverse, delta))
            return np.stack(rows, axis=1).astype(np.float32)
        rows = []
        for reference in self.references:
            chunks = []
            reference_norm = np.square(reference).sum(1)[None, :]
            for start in range(0, len(x), 2048):
                query = x[start:start + 2048].astype(np.float32)
                distance = np.maximum(np.square(query).sum(1)[:, None] + reference_norm - 2.0 * query @ reference.T, 0.0)
                k = min(self.knn_k, distance.shape[1])
                chunks.append(np.partition(distance, k - 1, axis=1)[:, :k].mean(1))
            rows.append(np.concatenate(chunks))
        return np.stack(rows, axis=1).astype(np.float32)


def _calibrate_similarity(train_z: torch.Tensor, train_y: torch.Tensor, val_z: torch.Tensor, val_y: torch.Tensor,
                          seen_ids: list[int], *, strategy: str, seed: int) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    prototypes = class_prototypes(train_z, train_y, seen_ids, strategy=strategy, seed=seed)
    scores = cosine_scores(val_z, prototypes)
    thresholds, predicted = _thresholds_from_class_scores(scores, val_y, seen_ids)
    return thresholds, scores, predicted


def _thresholds_from_class_scores(scores: np.ndarray, val_y: torch.Tensor, seen_ids: list[int]) -> tuple[dict[str, float], np.ndarray]:
    predicted_index = scores.argmax(1)
    confidence = scores.max(1)
    pseudo: list[np.ndarray] = []
    for class_id in seen_ids:
        if class_id == 0:
            continue
        class_index = seen_ids.index(class_id)
        mask = val_y.numpy() == class_id
        if mask.any():
            pseudo.append(np.delete(scores[mask], class_index, axis=1).max(1))
    pseudo_scores = np.concatenate(pseudo)
    thresholds = {"balanced": balanced_threshold(confidence, pseudo_scores)}
    normal = confidence[val_y.numpy() == 0]
    for far in (0.01, 0.02, 0.05):
        thresholds[f"normal_far_{int(far * 100)}pct"] = threshold_at_normal_far(normal, far)
    return thresholds, np.asarray([seen_ids[index] for index in predicted_index])


def _calibrate_density(scorer: DensityScorer, val_z: torch.Tensor, val_y: torch.Tensor) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    distances = scorer.distances(val_z)
    predicted_index = distances.argmin(1)
    predicted = np.asarray([scorer.class_ids[index] for index in predicted_index])
    calibration: dict[int, np.ndarray] = {}
    for index, class_id in enumerate(scorer.class_ids):
        mask = val_y.numpy() == class_id
        calibration[class_id] = distances[mask, index]
    p_values = np.asarray([
        conformal_p_values(calibration[class_id], np.asarray([distances[row, scorer.class_ids.index(class_id)]]))[0]
        for row, class_id in enumerate(predicted)
    ])
    pseudo = []
    for class_id in scorer.class_ids:
        if class_id == 0:
            continue
        class_index = scorer.class_ids.index(class_id)
        mask = val_y.numpy() == class_id
        other = np.delete(distances[mask], class_index, axis=1)
        other_ids = [value for value in scorer.class_ids if value != class_id]
        nearest = other.argmin(1)
        pseudo.extend(
            conformal_p_values(calibration[other_ids[index]], np.asarray([other[row, index]]))[0]
            for row, index in enumerate(nearest)
        )
    thresholds = {"balanced": balanced_threshold(p_values, np.asarray(pseudo))}
    normal = p_values[val_y.numpy() == 0]
    for far in (0.01, 0.02, 0.05):
        thresholds[f"normal_far_{int(far * 100)}pct"] = threshold_at_normal_far(normal, far)
    return thresholds, distances, predicted, calibration


def _density_confidence(scorer: DensityScorer, embeddings: torch.Tensor, calibration: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = scorer.distances(embeddings)
    indices = distances.argmin(1)
    predicted = np.asarray([scorer.class_ids[index] for index in indices])
    confidence = np.asarray([
        conformal_p_values(calibration[class_id], np.asarray([distances[row, scorer.class_ids.index(class_id)]]))[0]
        for row, class_id in enumerate(predicted)
    ])
    return distances, predicted, confidence


def _strict_and_gzsl(embeddings_seen: torch.Tensor, y_seen: torch.Tensor, embeddings_unseen: torch.Tensor, y_unseen: torch.Tensor,
                     prototypes: torch.Tensor, seen_ids: list[int], holdout: tuple[int, int], validation_z: torch.Tensor,
                     validation_y: torch.Tensor, gamma_max: float) -> tuple[dict[str, object], dict[str, object], dict[str, np.ndarray]]:
    all_scores_val = cosine_scores(validation_z, prototypes)
    candidates = np.linspace(0, gamma_max, 61)
    pseudo_faults = [value for value in seen_ids if value != 0]
    calibration_rows = []
    for gamma in candidates:
        fold_values = []
        for pseudo in pseudo_faults:
            adjusted = all_scores_val.copy()
            penalized = [value for value in seen_ids if value != pseudo]
            adjusted[:, penalized] -= gamma
            pred = adjusted.argmax(1)
            mask = np.isin(validation_y.numpy(), seen_ids)
            seen = macro_class_accuracy(validation_y.numpy()[mask], pred[mask], penalized)
            unseen = macro_class_accuracy(validation_y.numpy()[mask], pred[mask], [pseudo])
            fold_values.append(harmonic(seen, unseen))
        calibration_rows.append(float(np.mean(fold_values)))
    gamma = float(candidates[int(np.argmax(calibration_rows))])
    strict_scores = cosine_scores(embeddings_unseen, prototypes[list(holdout)])
    strict_pred = np.asarray([holdout[index] for index in strict_scores.argmax(1)])
    strict = classification_metrics(y_unseen.numpy(), strict_pred, class_ids=holdout)
    all_z = torch.cat([embeddings_seen, embeddings_unseen])
    all_y = np.concatenate([y_seen.numpy(), y_unseen.numpy()])
    scores = cosine_scores(all_z, prototypes)
    scores[:, seen_ids] -= gamma
    pred = scores.argmax(1)
    seen_accuracy = macro_class_accuracy(all_y, pred, seen_ids)
    unseen_accuracy = macro_class_accuracy(all_y, pred, holdout)
    gzsl = {
        "seen_macro_accuracy": seen_accuracy, "unseen_macro_accuracy": unseen_accuracy,
        "harmonic_mean": harmonic(seen_accuracy, unseen_accuracy), "seen_penalty": gamma,
        "seen_to_unseen_error": float(np.isin(pred[np.isin(all_y, seen_ids)], holdout).mean()),
        "unseen_to_seen_error": float(np.isin(pred[np.isin(all_y, holdout)], seen_ids).mean()),
        "per_class_recall": classification_metrics(all_y, pred)["per_class_recall"],
    }
    arrays = {"strict_scores": strict_scores, "strict_pred": strict_pred, "gzsl_scores": scores, "gzsl_pred": pred, "gzsl_true": all_y}
    return strict, gzsl, arrays


def _post_enrollment(train_z: torch.Tensor, train_y: torch.Tensor, support_z: torch.Tensor, support_y: torch.Tensor,
                     query_seen_z: torch.Tensor, query_seen_y: torch.Tensor, query_unseen_z: torch.Tensor, query_unseen_y: torch.Tensor,
                     *, seen_ids: list[int], holdout: tuple[int, int], strategy: str, thresholds: dict[str, float],
                     seed: int, support_draws: int) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, float]]:
    all_z = torch.cat([query_seen_z, query_unseen_z])
    all_y = np.concatenate([query_seen_y.numpy(), query_unseen_y.numpy()])
    arrays: dict[str, np.ndarray] = {"post_true": all_y.astype(np.int16)}
    result: dict[str, object] = {}
    enrollment_durations: list[float] = []
    gallery_memory_bytes: list[int] = []
    support_labels = support_y.numpy()
    for count in (1, 3, 5):
        rows_by_op: dict[str, list[dict[str, object]]] = {key: [] for key in thresholds}
        stored_pred = np.empty((support_draws, len(all_y)), dtype=np.int16)
        stored_conf = np.empty((support_draws, len(all_y)), dtype=np.float16)
        for draw in range(support_draws):
            enrollment_started = time.perf_counter()
            chosen = deterministic_support_indices(support_labels, holdout, count=count, draw=draw, seed=seed + 90_000)
            if strategy == "equal":
                seen_prototypes = class_prototypes(train_z, train_y, seen_ids, strategy="equal", seed=seed + draw, count=count)
            else:
                seen_prototypes = class_prototypes(train_z, train_y, seen_ids, strategy=strategy, seed=seed + draw, count=count)
            unseen_prototypes = class_prototypes(support_z[chosen], support_y[chosen], holdout, strategy="prototype")
            class_ids = seen_ids + list(holdout)
            prototypes = torch.cat([seen_prototypes, unseen_prototypes])
            enrollment_durations.append(time.perf_counter() - enrollment_started)
            gallery_memory_bytes.append(int(prototypes.numel() * prototypes.element_size()))
            scores = cosine_scores(all_z, prototypes)
            predicted = np.asarray([class_ids[index] for index in scores.argmax(1)], dtype=np.int16)
            confidence = scores.max(1)
            stored_pred[draw] = predicted
            stored_conf[draw] = confidence.astype(np.float16)
            for op, threshold in thresholds.items():
                rejected = predicted.copy()
                rejected[confidence < threshold] = -1
                row = post_enrollment_metrics(all_y, rejected, seen_ids=seen_ids, unseen_ids=holdout)
                row["draw"] = draw
                row["support_indices"] = chosen.tolist()
                rows_by_op[op].append(row)
        result[f"{count}_shot"] = {op: summarize_draws(rows) for op, rows in rows_by_op.items()}
        arrays[f"post_{count}_shot_pred"] = stored_pred
        arrays[f"post_{count}_shot_confidence"] = stored_conf
    efficiency = {
        "enrollment_latency_ms_mean": float(np.mean(enrollment_durations) * 1000),
        "enrollment_latency_ms_std": float(np.std(enrollment_durations, ddof=1) * 1000),
        "gallery_memory_bytes_max": float(max(gallery_memory_bytes)),
    }
    return result, arrays, efficiency


def split_manifest(prepared: PreparedFold) -> dict[str, object]:
    def entry(frame: pd.DataFrame) -> dict[str, object]:
        return {"rows": len(frame), "groups": frame["_input_group"].nunique(),
                "classes": {str(int(k)): int(v) for k, v in frame["Class"].value_counts().sort_index().items()},
                "groups_sha256": __import__("hashlib").sha256("\n".join(sorted(frame["_input_group"].unique())).encode()).hexdigest()}
    return {
        "holdout": list(prepared.outer.holdout), "features": INPUT_COLUMNS,
        "train": entry(prepared.outer.train), "validation": entry(prepared.outer.validation),
        "seen_test": entry(prepared.outer.seen_test), "support_pool": entry(prepared.enrollment.support_pool),
        "unseen_query": entry(prepared.enrollment.query),
    }


def run_study_fold(*, approach: str, frame: pd.DataFrame, data_path: Path, run_dir: Path, holdout: tuple[int, int], seed: int,
                   config: ApproachAConfig | ApproachBConfig | ApproachCConfig, device: torch.device,
                   physics_path: Path, description_path: Path, study_root: Path, support_draws: int = 20) -> dict[str, object]:
    started = time.perf_counter()
    prepared = prepare_fold(frame, holdout=holdout, seed=seed)
    seen_ids = sorted(set(range(8)) - set(holdout))
    semantic = None
    attribute_names = None
    if approach == "a":
        model, training = train_approach_a(prepared.train_x, prepared.train_y, prepared.validation_x, prepared.validation_y, device=device, config=config)
    elif approach == "b":
        attribute_names, _, semantic = semantic_prototypes(
            mode=config.prototype_mode, physics_path=physics_path, description_path=description_path,
            text_model="sentence-transformers/all-mpnet-base-v2", device=device, cache_dir=study_root / "cache",
        )
        model, training = train_approach_b(prepared.train_x, prepared.train_y, prepared.validation_x, prepared.validation_y, semantic, device=device, config=config)
    elif approach == "c":
        model, training = train_approach_c(prepared.train_x, device=device, config=config)
    else:
        raise ValueError(f"Unknown approach {approach}")
    train_z = encode(model, prepared.train_x, device=device, kind=approach)
    val_z = encode(model, prepared.validation_x, device=device, kind=approach)
    support_z = encode(model, prepared.support_x, device=device, kind=approach)
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    seen_z = encode(model, prepared.seen_test_x, device=device, kind=approach)
    query_z = encode(model, prepared.query_x, device=device, kind=approach)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    post_strategy = config.aggregation if approach == "a" else "prototype"
    post_thresholds, _, _ = _calibrate_similarity(
        train_z, prepared.train_y, val_z, prepared.validation_y, seen_ids,
        strategy=post_strategy, seed=seed,
    )
    density_fit_seconds = 0.0
    scoring_started = time.perf_counter()
    if approach == "c":
        density_started = time.perf_counter()
        scorer = DensityScorer(train_z, prepared.train_y, seen_ids, density=config.density, shrinkage=config.covariance_shrinkage, knn_k=config.knn_k)
        density_fit_seconds = time.perf_counter() - density_started
        thresholds, _, _, calibration = _calibrate_density(scorer, val_z, prepared.validation_y)
        _, seen_pred, seen_conf = _density_confidence(scorer, seen_z, calibration)
        _, unseen_pred, unseen_conf = _density_confidence(scorer, query_z, calibration)
    else:
        strategy = config.aggregation if approach == "a" else "prototype"
        if approach == "b":
            semantic_validation_scores = cosine_scores(val_z, semantic[seen_ids])
            thresholds, _ = _thresholds_from_class_scores(semantic_validation_scores, prepared.validation_y, seen_ids)
        else:
            thresholds = post_thresholds
        if approach == "b":
            prototype_matrix = semantic
        else:
            prototype_matrix = class_prototypes(train_z, prepared.train_y, seen_ids, strategy=strategy, seed=seed)
        seen_scores = cosine_scores(seen_z, prototype_matrix if approach == "b" else prototype_matrix)
        unseen_scores = cosine_scores(query_z, prototype_matrix if approach == "b" else prototype_matrix)
        candidate_ids = list(range(8)) if approach == "b" else seen_ids
        # Zero-day recognition is scored only against seen candidates; semantic held-out scores are handled separately.
        if approach == "b":
            seen_scores = seen_scores[:, seen_ids]
            unseen_scores = unseen_scores[:, seen_ids]
        seen_pred = np.asarray([seen_ids[index] for index in seen_scores.argmax(1)])
        unseen_pred = np.asarray([seen_ids[index] for index in unseen_scores.argmax(1)])
        seen_conf, unseen_conf = seen_scores.max(1), unseen_scores.max(1)
    scoring_seconds = time.perf_counter() - scoring_started

    pre_true = np.concatenate([prepared.seen_test_y.numpy(), prepared.query_y.numpy()])
    pre_pred = np.concatenate([seen_pred, unseen_pred])
    pre_conf = np.concatenate([seen_conf, unseen_conf])
    is_known = np.concatenate([np.ones(len(seen_pred), dtype=bool), np.zeros(len(unseen_pred), dtype=bool)])
    pre = {op: open_set_metrics(is_known=is_known, confidence=pre_conf, predicted=pre_pred, true_labels=pre_true, threshold=value) for op, value in thresholds.items()}
    predictions: dict[str, np.ndarray] = {
        "pre_true": pre_true.astype(np.int16), "pre_pred": pre_pred.astype(np.int16),
        "pre_confidence": pre_conf.astype(np.float32), "pre_is_known": is_known,
    }
    strict = gzsl = None
    if approach == "b":
        strict, gzsl, semantic_arrays = _strict_and_gzsl(
            seen_z, prepared.seen_test_y, query_z, prepared.query_y, semantic,
            seen_ids, holdout, val_z, prepared.validation_y, config.seen_penalty_grid_max,
        )
        predictions.update(semantic_arrays)
    post, post_arrays, enrollment_efficiency = _post_enrollment(
        train_z, prepared.train_y, support_z, prepared.support_y, seen_z, prepared.seen_test_y,
        query_z, prepared.query_y, seen_ids=seen_ids, holdout=holdout, strategy=post_strategy,
        thresholds=post_thresholds, seed=seed, support_draws=support_draws,
    )
    predictions.update(post_arrays)
    efficiency = {
        "parameter_count": training["parameter_count"],
        "training_duration_seconds": training["duration_seconds"],
        "inference_latency_ms_per_trace": 1000 * inference_seconds / max(len(prepared.seen_test_x) + len(prepared.query_x), 1),
        "pre_enrollment_scoring_latency_ms_per_trace": 1000 * scoring_seconds / max(len(prepared.seen_test_x) + len(prepared.query_x), 1),
        "density_fit_seconds": density_fit_seconds,
        "peak_gpu_memory_bytes": training["peak_allocated_bytes"],
        **enrollment_efficiency,
    }
    metrics = {
        "schema_version": 1, "approach": approach, "holdout": list(holdout), "seed": seed,
        "thresholds": thresholds, "post_thresholds": post_thresholds,
        "pre_enrollment": pre, "strict_zsl": strict, "gzsl": gzsl,
        "post_enrollment": post, "training": training, "efficiency": efficiency,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    np.savez_compressed(run_dir / "predictions_scores.npz", **predictions)
    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "split_manifest.json", split_manifest(prepared))
    save_json(run_dir / "scaler.json", {"features": INPUT_COLUMNS, "mean": prepared.scaler.mean_.tolist(), "scale": prepared.scaler.scale_.tolist()})
    try:
        source_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=study_root.parents[3], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = None
    metadata = {
        "schema_version": 1, "approach": approach, "holdout": list(holdout), "seed": seed,
        "config": asdict(config), "run_id": stable_run_id(approach, holdout, seed, config),
        "dataset_path": str(data_path.resolve()), "dataset_sha256": file_sha256(data_path),
        "feature_signature": feature_signature(), "forbidden_features_disabled": True,
        "physics_prototype_sha256": file_sha256(physics_path), "description_prototype_sha256": file_sha256(description_path),
        "attribute_names": attribute_names, "environment": environment_metadata(device),
        "source_revision": source_revision,
        "duration_seconds": time.perf_counter() - started,
    }
    save_json(run_dir / "metadata.json", metadata)
    write_manifest(run_dir, metadata)
    return metrics
