from __future__ import annotations

"""Restartable confirmatory TabPFN-v2 enrollment study for conventional OTDR."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .lifecycle_baselines import NearestNeighborReference
from .lifecycle_data import (
    FEATURE_REGIMES,
    deterministic_support_indices,
    fit_lifecycle_fold,
    lifecycle_split_manifest,
)
from .lifecycle_enrollment import EnrollmentSession
from .lifecycle_experiment import _git_metadata
from .lifecycle_metrics import hard_prediction_metrics
from .lifecycle_tabpfn import _balanced_query_indices, _ranked_indices
from .lifecycle_training import infer_lifecycle_model
from .model_functions.lifecycle import FeatureAssistedOTDR, LifecycleModelConfig
from .model_functions.zero_shot import require_cuda
from .study_state import (
    append_jsonl,
    artifact_hashes,
    atomic_json,
    config_hash,
    environment_metadata,
    file_sha256,
    utc_now,
    validate_run,
    write_manifest,
)


OTDR_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = Path(__file__).resolve().parent / "data" / "OTDR_DATA.csv"
STUDY_ROOT = OTDR_ROOT / "experiments" / "otdr_tabpfn_full_enrollment_study"
SOURCE_STUDY = OTDR_ROOT / "experiments" / "otdr_feature_assisted_lifecycle_study"
PROTOCOL_PATH = STUDY_ROOT / "configs" / "protocol.json"
FROZEN_PROTOCOL_SHA256 = "388e2e757f0904134d9a09e0b3d0c8ed70a5ae8a200868f3d972489f6b7339f6"
N_CLASSES = 8
METHODS = (
    "tabpfn_v2",
    "cfe_finalist",
    "cfe_uncalibrated_mean",
    "raw_cosine_1nn",
    "raw_euclidean_1nn",
    "raw_mahalanobis_1nn",
    "encoder_cosine_1nn",
    "logistic_regression",
    "linear_svm",
    "shrinkage_lda",
)
_SOURCE_INDEX_CACHE: dict[tuple[str, tuple[int, int], int], Path] | None = None


def load_protocol() -> dict[str, Any]:
    if file_sha256(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol hash mismatch; refusing to open confirmatory results.")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not payload.get("protocol_frozen_before_new_outer_results"):
        raise RuntimeError("Protocol is not marked frozen.")
    return payload


def _features(batch: Any, regime: str) -> np.ndarray:
    """Return the exact lifecycle-scaled feature contract for tabular methods."""
    context = batch.context.numpy()
    if regime == "full":
        return np.c_[batch.trace.numpy(), context]
    if regime == "summary_only":
        return context
    if regime == "trace_only":
        return np.c_[batch.trace.numpy(), context[:, :1]]
    raise ValueError(f"Unsupported feature regime: {regime}")


def _raw_features(batch: Any) -> np.ndarray:
    """Replicate the frozen CFE study's raw baseline representation."""
    return np.c_[
        batch.trace.numpy(),
        batch.context.numpy(),
        batch.context_missing.numpy(),
    ]


def _softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    result = np.exp(np.clip(x, -700, 0))
    return result / np.clip(result.sum(1, keepdims=True), 1e-15, None)


def _aligned_probability(
    probability: np.ndarray,
    classes: Iterable[int],
    *,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    result = np.zeros((len(probability), n_classes), dtype=np.float64)
    result[:, np.asarray(tuple(classes), dtype=int)] = np.asarray(
        probability, dtype=np.float64
    )
    result = np.clip(result, 1e-12, None)
    return result / result.sum(1, keepdims=True)


def probability_sufficient_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if probability.shape != (len(y), N_CLASSES):
        raise ValueError("Probabilities must be [examples, 8].")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Probabilities must be finite and non-negative.")
    probability = probability / probability.sum(1, keepdims=True)
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = prediction == y
    edges = np.linspace(0, 1, bins + 1)
    ece_rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence > left) & (confidence <= right)
        count = int(mask.sum())
        confidence_sum = float(confidence[mask].sum())
        correct_count = int(correct[mask].sum())
        if count:
            ece += count / len(y) * abs(
                correct_count / count - confidence_sum / count
            )
        ece_rows.append(
            {
                "bin": index,
                "count": count,
                "confidence_sum": confidence_sum,
                "correct_count": correct_count,
            }
        )
    one_hot = np.eye(N_CLASSES, dtype=np.float64)[y]
    nll_sum = float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1)).sum())
    brier_sum = float(np.square(probability - one_hot).sum())
    return {
        "nll": nll_sum / len(y),
        "brier": brier_sum / len(y),
        "ece_15": float(ece),
        "probability_sufficient": {
            "examples": len(y),
            "nll_sum": nll_sum,
            "brier_sum": brier_sum,
            "ece_bins": ece_rows,
        },
    }


def metric_row(
    *,
    labels: np.ndarray,
    probability: np.ndarray,
    base_class_ids: tuple[int, ...],
    enrolled_class_ids: tuple[int, int],
    method: str,
    shots: int,
    draw: int,
    elapsed_seconds: float,
    probability_source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = np.asarray(probability).argmax(1)
    result = {
        "method": method,
        "shots": int(shots),
        "draw": int(draw),
        "elapsed_seconds": float(elapsed_seconds),
        "probability_source": probability_source,
        **hard_prediction_metrics(
            labels,
            prediction,
            base_class_ids=base_class_ids,
            enrolled_class_ids=enrolled_class_ids,
        ),
        **probability_sufficient_statistics(probability, labels),
    }
    normal = labels == 0
    result["normal_far_after_enrollment"] = float(
        np.isin(prediction[normal], enrolled_class_ids).mean()
    )
    if extra:
        result.update(extra)
    return result


def reconstruct_row(row: dict[str, Any]) -> dict[str, float]:
    """Independently reconstruct reported metrics from saved sufficient data."""
    cm = np.asarray(row["confusion_matrix"], dtype=np.float64)
    support = cm.sum(1)
    predicted = cm.sum(0)
    diagonal = np.diag(cm)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    precision = np.divide(
        diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )
    base_ids = tuple(int(value) for value in row["base_class_ids"])
    enrolled_ids = tuple(int(value) for value in row["enrolled_class_ids"])
    base_accuracy = float(np.mean(recall[list(base_ids)]))
    enrolled_accuracy = float(np.mean(recall[list(enrolled_ids)]))
    harmonic = (
        0.0
        if base_accuracy + enrolled_accuracy == 0
        else 2 * base_accuracy * enrolled_accuracy / (base_accuracy + enrolled_accuracy)
    )
    sufficient = row["probability_sufficient"]
    examples = int(sufficient["examples"])
    ece = 0.0
    for bin_row in sufficient["ece_bins"]:
        count = int(bin_row["count"])
        if count:
            ece += count / examples * abs(
                int(bin_row["correct_count"]) / count
                - float(bin_row["confidence_sum"]) / count
            )
    return {
        "accuracy": float(diagonal.sum() / cm.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "base_accuracy": base_accuracy,
        "enrolled_accuracy": enrolled_accuracy,
        "harmonic_mean": harmonic,
        "worst_enrolled_recall": float(np.min(recall[list(enrolled_ids)])),
        "nll": float(sufficient["nll_sum"]) / examples,
        "brier": float(sufficient["brier_sum"]) / examples,
        "ece_15": float(ece),
    }


def _class_min_distances(
    reference: NearestNeighborReference,
    query: np.ndarray,
    *,
    class_ids: tuple[int, ...],
    chunk_size: int = 512,
) -> np.ndarray:
    result = []
    for start in range(0, len(query), chunk_size):
        distance = reference._distance(query[start : start + chunk_size], reference.features)
        result.append(
            np.stack(
                [
                    distance[:, reference.labels == class_id].min(1)
                    for class_id in class_ids
                ],
                axis=1,
            )
        )
    return np.vstack(result)


def _support_distances(
    reference: NearestNeighborReference,
    query: np.ndarray,
    support: np.ndarray,
    support_labels: np.ndarray,
    *,
    enrolled_ids: tuple[int, int],
    chunk_size: int = 2048,
) -> np.ndarray:
    result = []
    for start in range(0, len(query), chunk_size):
        distance = reference._distance(query[start : start + chunk_size], support)
        result.append(
            np.stack(
                [
                    distance[:, support_labels == class_id].min(1)
                    for class_id in enrolled_ids
                ],
                axis=1,
            )
        )
    return np.vstack(result)


def _distance_probability(
    base_distance: np.ndarray,
    support_distance: np.ndarray,
    *,
    base_ids: tuple[int, ...],
    enrolled_ids: tuple[int, int],
    temperature: float,
) -> np.ndarray:
    distance = np.full((len(base_distance), N_CLASSES), np.inf, dtype=np.float64)
    distance[:, np.asarray(base_ids)] = base_distance
    distance[:, np.asarray(enrolled_ids)] = support_distance
    return _softmax(-distance / temperature)


def _classical_probability(
    method: str,
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
) -> np.ndarray:
    if method == "logistic_regression":
        model = LogisticRegression(
            C=1.0, max_iter=2000, random_state=0, solver="liblinear"
        )
        model.fit(context_x, context_y)
        return _aligned_probability(model.predict_proba(query_x), model.classes_)
    if method == "linear_svm":
        model = SVC(C=1.0, kernel="linear", probability=False, decision_function_shape="ovr")
        model.fit(context_x, context_y)
        score = model.decision_function(query_x)
        return _aligned_probability(_softmax(score), model.classes_)
    if method == "shrinkage_lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        if len(context_x) <= len(np.unique(context_y)):
            # sklearn requires n_samples > n_classes. Exact repetition changes
            # sample weight only and introduces no synthetic information.
            context_x = np.vstack((context_x, context_x))
            context_y = np.r_[context_y, context_y]
        model.fit(context_x, context_y)
        return _aligned_probability(model.predict_proba(query_x), model.classes_)
    raise ValueError(f"Unknown classical method: {method}")


def _source_run_index() -> dict[tuple[str, tuple[int, int], int], Path]:
    global _SOURCE_INDEX_CACHE
    if _SOURCE_INDEX_CACHE is not None:
        return _SOURCE_INDEX_CACHE
    result: dict[tuple[str, tuple[int, int], int], Path] = {}
    for run_dir in sorted((SOURCE_STUDY / "full_benchmark").glob("lifecycle-*")):
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        key = (
            str(config["regime"]),
            tuple(int(value) for value in config["holdout"]),
            int(config["seed"]),
        )
        if key in result:
            raise RuntimeError(f"Ambiguous frozen CFE source for {key}.")
        valid, reason = validate_run(run_dir, expected={"run_id": run_dir.name})
        if not valid:
            raise RuntimeError(f"Invalid frozen source {run_dir}: {reason}")
        result[key] = run_dir
    _SOURCE_INDEX_CACHE = result
    return result


def _load_cfe_source(
    *,
    regime: str,
    pair: tuple[int, int],
    seed: int,
    tensor_fold: Any,
    device: torch.device,
    expected_outer_groups: tuple[str, ...],
    expected_reference_groups: tuple[str, ...],
) -> dict[str, Any] | None:
    source = _source_run_index().get((regime, pair, seed))
    if source is None:
        return None
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest_device = manifest.get("device") or manifest.get("environment", {}).get(
        "device"
    )
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    training_device = metrics.get("training", {}).get("device")
    inference_device = metrics.get("inference", {}).get("device")
    if (
        manifest_device != "cuda:0"
        or training_device != "cuda:0"
        or (inference_device is not None and inference_device != "cuda:0")
    ):
        raise RuntimeError(f"Frozen CFE source lacks CUDA provenance: {source}")
    checkpoint = torch.load(
        source / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("scaler") != tensor_fold.scaler.payload():
        raise RuntimeError(f"Frozen CFE scaler differs from reconstructed fold: {source}")
    with np.load(source / "predictions.npz") as saved:
        outer_groups = tuple(str(value) for value in saved["group_ids"])
        if outer_groups != expected_outer_groups:
            raise RuntimeError(f"Frozen CFE outer group ordering mismatch: {source}")
        has_embeddings = all(
            name in saved.files
            for name in (
                "train_embedding",
                "reference_embedding",
                "reference_competence",
                "reference_group_ids",
                "outer_embedding",
            )
        )
        if has_embeddings:
            reference_groups = tuple(
                str(value) for value in saved["reference_group_ids"]
            )
            if reference_groups != expected_reference_groups:
                raise RuntimeError(
                    f"Frozen CFE reference group ordering mismatch: {source}"
                )
            train_embedding = saved["train_embedding"].copy()
            train_labels = saved["train_labels"].astype(int)
            reference_embedding = saved["reference_embedding"].copy()
            reference_competence = saved["reference_competence"].copy()
            reference_labels = saved["reference_labels"].astype(int)
            outer_embedding = saved["outer_embedding"].copy()
            embedding_source = "saved_cuda_embedding_artifact"
            embedding_inference_seconds = 0.0
        else:
            model = FeatureAssistedOTDR(
                LifecycleModelConfig(**checkpoint["model_config"])
            )
            model.load_state_dict(checkpoint["state_dict"])
            inference_started = time.perf_counter()
            train_output = infer_lifecycle_model(
                model, tensor_fold.batches["train"], device=device
            )
            reference_output = infer_lifecycle_model(
                model, tensor_fold.batches["reference_pool"], device=device
            )
            seen_output = infer_lifecycle_model(
                model, tensor_fold.batches["seen_test"], device=device
            )
            query_output = infer_lifecycle_model(
                model, tensor_fold.batches["query"], device=device
            )
            torch.cuda.synchronize(device)
            embedding_inference_seconds = time.perf_counter() - inference_started
            train_embedding = train_output["embedding"].numpy()
            train_labels = tensor_fold.batches["train"].labels.numpy()
            reference_embedding = reference_output["embedding"].numpy()
            reference_competence = torch.sigmoid(
                reference_output["competence"]
            ).numpy()
            reference_labels = tensor_fold.batches[
                "reference_pool"
            ].labels.numpy()
            outer_embedding = np.vstack(
                (seen_output["embedding"].numpy(), query_output["embedding"].numpy())
            )
            embedding_source = "cuda_reconstructed_from_frozen_checkpoint"
        payload = {
            "source_dir": str(source),
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "source_checkpoint_sha256": file_sha256(source / "checkpoint.pt"),
            "embedding_source": embedding_source,
            "embedding_inference_seconds": embedding_inference_seconds,
            "embedding_device": str(device),
            "train_embedding": train_embedding,
            "train_labels": train_labels,
            "reference_embedding": reference_embedding,
            "reference_competence": reference_competence,
            "reference_labels": reference_labels,
            "outer_embedding": outer_embedding,
        }
    return payload


def _unit_dir(regime: str, pair: tuple[int, int], seed: int) -> Path:
    stage = "full_benchmark" if regime == "full" else "summary_only"
    return STUDY_ROOT / stage / f"pair_{pair[0]:02d}_{pair[1]:02d}" / f"seed_{seed}"


def _update_state(
    *,
    status: str,
    unit_id: str | None = None,
    failed: bool = False,
    opened_results: bool | None = None,
) -> None:
    path = STUDY_ROOT / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["status"] = status
    state["updated_at"] = utc_now()
    if unit_id:
        key = "failed_units" if failed else "completed_units"
        if unit_id not in state[key]:
            state[key].append(unit_id)
        if not failed and unit_id in state["failed_units"]:
            state["failed_units"].remove(unit_id)
    if opened_results is not None:
        state["opened_confirmatory_outer_results"] = opened_results
    atomic_json(path, state)


def _weight_paths(model: object) -> list[Path]:
    result = []
    for value in vars(model).values():
        if isinstance(value, Path) and value.is_file():
            result.append(value)
        elif isinstance(value, str) and Path(value).is_file():
            result.append(Path(value))
    return sorted(set(result))


def run_unit(
    *,
    frame: pd.DataFrame,
    regime: str,
    pair: tuple[int, int],
    seed: int,
    device: str = "cuda:0",
    shots_values: tuple[int, ...] = (1, 3, 5),
    draws: int = 20,
    include_tabpfn: bool = True,
) -> dict[str, Any]:
    """Evaluate one pair/seed unit; completed units validate and resume."""
    protocol = load_protocol()
    if regime not in {"full", "summary_only"}:
        raise ValueError("Confirmatory study supports full and summary_only.")
    pair = tuple(sorted(int(value) for value in pair))
    unit_root = _unit_dir(regime, pair, seed)
    unit_id = f"{regime}-pair-{pair[0]}-{pair[1]}-seed-{seed}"
    unit_config = {
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "regime": regime,
        "pair": list(pair),
        "seed": int(seed),
        "shots": list(shots_values),
        "draws": int(draws),
        "include_tabpfn": bool(include_tabpfn),
    }
    unit_config_hash = config_hash(unit_config)
    expected = {
        "run_id": unit_id,
        "evidence_schema": 3,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "requested_draws": int(draws),
        "requested_shots": list(shots_values),
        "include_tabpfn": bool(include_tabpfn),
    }
    valid, _ = validate_run(unit_root, expected=expected)
    if valid:
        return json.loads((unit_root / "metrics.json").read_text(encoding="utf-8"))

    cuda = require_cuda(device)
    torch.cuda.synchronize(cuda)
    torch.cuda.reset_peak_memory_stats(cuda)
    started = time.perf_counter()
    unit_root.mkdir(parents=True, exist_ok=True)
    old_manifest = unit_root / "manifest.json"
    if old_manifest.is_file():
        try:
            old_manifest_payload = json.loads(
                old_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            old_manifest_payload = {}
        old_evidence_schema = int(
            old_manifest_payload.get("evidence_schema", 1)
        )
        archive = (
            unit_root
            / "pre_evidence_upgrade"
            / f"schema_{old_evidence_schema}"
        )
        archive.mkdir(parents=True, exist_ok=True)
        for name in (
            "manifest.json",
            "metrics.json",
            "prediction_evidence.npz",
            "support_manifest.json",
            "context_manifest.json",
            "query_manifest.json",
            "split_manifest.json",
        ):
            source_path = unit_root / name
            target_path = archive / name
            if source_path.is_file() and not target_path.exists():
                shutil.copy2(source_path, target_path)
    append_jsonl(
        STUDY_ROOT / "experiment_registry.jsonl",
        {
            "event": "started",
            "timestamp": utc_now(),
            "run_id": unit_id,
            "stage": "full_benchmark" if regime == "full" else "summary_only",
            "config_hash": unit_config_hash,
            "regime": regime,
            "pair": list(pair),
            "seed": seed,
            "device": str(cuda),
        },
    )
    _update_state(status="running", opened_results=True)
    try:
        fold = fit_lifecycle_fold(frame, holdout=pair, seed=seed, regime=regime)
        batches = fold.batches
        train_x = _features(batches["train"], regime)
        train_y = batches["train"].labels.numpy()
        train_groups = batches["train"].group_ids
        reference_x = _features(batches["reference_pool"], regime)
        reference_raw = _raw_features(batches["reference_pool"])
        reference_frame = fold.split.reference_pool
        outer_frame = pd.concat(
            (fold.split.seen_test, fold.split.query), ignore_index=True
        )
        outer_groups = (
            *batches["seen_test"].group_ids,
            *batches["query"].group_ids,
        )
        outer_y_all = np.r_[
            batches["seen_test"].labels.numpy(),
            batches["query"].labels.numpy(),
        ]
        outer_x_all = np.vstack(
            (_features(batches["seen_test"], regime), _features(batches["query"], regime))
        )
        outer_raw_all = np.vstack(
            (_raw_features(batches["seen_test"]), _raw_features(batches["query"]))
        )
        query_indices = _balanced_query_indices(
            outer_y_all,
            outer_groups,
            per_class=int(protocol["query_examples_per_class_cap"]),
        )
        query_x = outer_x_all[query_indices]
        query_raw = outer_raw_all[query_indices]
        query_y = outer_y_all[query_indices]
        query_groups = tuple(outer_groups[index] for index in query_indices)
        if len(query_groups) != len(set(query_groups)):
            raise AssertionError("Balanced query is not group-distinct.")
        base_ids = tuple(sorted(int(value) for value in np.unique(train_y)))
        if set(base_ids) & set(pair) or set(base_ids) | set(pair) != set(range(8)):
            raise AssertionError("Base/enrolled class partition is invalid.")

        cfe = _load_cfe_source(
            regime=regime,
            pair=pair,
            seed=seed,
            tensor_fold=fold,
            device=cuda,
            expected_outer_groups=tuple(outer_groups),
            expected_reference_groups=batches["reference_pool"].group_ids,
        )
        query_embedding = (
            cfe["outer_embedding"][query_indices] if cfe is not None else None
        )
        reference_embedding = cfe["reference_embedding"] if cfe is not None else None

        raw_references = {
            "raw_cosine_1nn": NearestNeighborReference.fit(
                _raw_features(batches["train"]), train_y, metric="cosine", seed=seed
            ),
            "raw_euclidean_1nn": NearestNeighborReference.fit(
                _raw_features(batches["train"]), train_y, metric="euclidean", seed=seed
            ),
            "raw_mahalanobis_1nn": NearestNeighborReference.fit(
                _raw_features(batches["train"]),
                train_y,
                metric="diagonal_mahalanobis",
                seed=seed,
            ),
        }
        raw_base_distances = {
            name: _class_min_distances(reference, query_raw, class_ids=base_ids)
            for name, reference in raw_references.items()
        }
        encoder_reference = None
        encoder_base_distance = None
        if cfe is not None:
            encoder_reference = NearestNeighborReference.fit(
                cfe["train_embedding"],
                cfe["train_labels"],
                metric="cosine",
                seed=seed,
            )
            encoder_base_distance = _class_min_distances(
                encoder_reference, query_embedding, class_ids=base_ids
            )

        tabpfn_models: list[Any] = []
        resolved_model_path: Path | None = None
        resolved_model_name: str | None = None
        checkpoint_resolution_seconds = 0.0
        model_initialization_seconds = 0.0
        if include_tabpfn:
            from tabpfn import TabPFNClassifier
            from tabpfn.model_loading import resolve_model_path

            checkpoint_started = time.perf_counter()
            resolved_model_path, _, resolved_model_name, _ = resolve_model_path(
                None, "classifier", "v2"
            )
            checkpoint_resolution_seconds = time.perf_counter() - checkpoint_started
            if file_sha256(resolved_model_path) != protocol["tabpfn"]["checkpoint_sha256"]:
                raise RuntimeError("TabPFN checkpoint hash differs from frozen protocol.")
            initialization_started = time.perf_counter()
            tabpfn_models = [
                TabPFNClassifier(
                    n_estimators=1,
                    model_path=resolved_model_path,
                    device=str(cuda),
                    ignore_pretraining_limits=True,
                    fit_mode="fit_with_cache",
                    random_state=context_seed,
                    n_jobs=1,
                )
                for context_seed in protocol["tabpfn"]["context_seeds"]
            ]
            model_initialization_seconds = (
                time.perf_counter() - initialization_started
            )

        rows: list[dict[str, Any]] = []
        evidence_row_ids: list[str] = []
        evidence_predictions: list[np.ndarray] = []
        evidence_probabilities: list[np.ndarray] = []
        context_sensitivity_rows: list[dict[str, Any]] = []
        support_manifest: list[dict[str, Any]] = []
        context_manifest: list[dict[str, Any]] = []
        first_tabpfn_context_seconds: float | None = None

        def append_evidence_row(
            row: dict[str, Any], probability: np.ndarray
        ) -> None:
            probability_array = np.asarray(probability, dtype=np.float64)
            probability_array /= probability_array.sum(1, keepdims=True)
            rows.append(row)
            evidence_row_ids.append(
                f"{row['method']}|shot={row['shots']}|draw={row['draw']}"
            )
            evidence_predictions.append(
                probability_array.argmax(1).astype(np.uint8)
            )
            evidence_probabilities.append(probability_array.astype(np.float64))

        for shots in shots_values:
            for draw in range(draws):
                selected = deterministic_support_indices(
                    reference_frame,
                    class_ids=pair,
                    shots=shots,
                    seed=seed,
                    draw=draw,
                    namespace="tabpfn-support",
                )
                positions = reference_frame.index.get_indexer(selected)
                if np.any(positions < 0):
                    raise AssertionError("Support indices did not map to reference pool.")
                support_y = reference_frame.loc[selected, "Class"].to_numpy(dtype=int)
                support_x = reference_x[positions]
                support_raw = reference_raw[positions]
                support_groups = tuple(
                    reference_frame.loc[selected, "_input_group"].astype(str)
                )
                if len(support_groups) != len(set(support_groups)):
                    raise AssertionError("Support set is not group-distinct.")
                if set(support_groups) & set(query_groups):
                    raise AssertionError("Support/query group leakage.")
                support_manifest.append(
                    {
                        "shots": int(shots),
                        "draw": int(draw),
                        "groups_by_class": {
                            str(class_id): [
                                support_groups[index]
                                for index in np.flatnonzero(support_y == class_id)
                            ]
                            for class_id in pair
                        },
                        "query_used": False,
                    }
                )

                context_probabilities: dict[str, list[np.ndarray]] = {
                    "tabpfn_v2": [],
                    "logistic_regression": [],
                    "linear_svm": [],
                    "shrinkage_lda": [],
                }
                context_seconds = {name: 0.0 for name in context_probabilities}
                for ensemble_index, context_seed in enumerate(
                    protocol["tabpfn"]["context_seeds"]
                ):
                    base_indices = np.concatenate(
                        [
                            _ranked_indices(
                                train_y,
                                train_groups,
                                class_id,
                                shots,
                                f"tabpfn-context:{context_seed}:{draw}:{shots}",
                            )
                            for class_id in base_ids
                        ]
                    )
                    base_context_groups = tuple(
                        train_groups[index] for index in base_indices
                    )
                    if len(base_context_groups) != len(set(base_context_groups)):
                        raise AssertionError("Base context is not group-distinct.")
                    if set(base_context_groups) & (
                        set(support_groups) | set(query_groups)
                    ):
                        raise AssertionError("Context overlaps support or query.")
                    context_manifest.append(
                        {
                            "shots": int(shots),
                            "draw": int(draw),
                            "context_seed": int(context_seed),
                            "base_groups_by_class": {
                                str(class_id): [
                                    train_groups[index]
                                    for index in base_indices
                                    if train_y[index] == class_id
                                ]
                                for class_id in base_ids
                            },
                        }
                    )
                    context_x = np.vstack((train_x[base_indices], support_x))
                    context_y = np.r_[train_y[base_indices], support_y]
                    if include_tabpfn:
                        method_started = time.perf_counter()
                        model = tabpfn_models[ensemble_index]
                        model.fit(context_x, context_y)
                        context_probability = _aligned_probability(
                            model.predict_proba(query_x), model.classes_
                        )
                        elapsed = time.perf_counter() - method_started
                        if first_tabpfn_context_seconds is None:
                            first_tabpfn_context_seconds = elapsed
                        context_probabilities["tabpfn_v2"].append(
                            context_probability
                        )
                        context_seconds["tabpfn_v2"] += elapsed
                        context_metrics = hard_prediction_metrics(
                            query_y,
                            context_probability.argmax(1),
                            base_class_ids=base_ids,
                            enrolled_class_ids=pair,
                        )
                        context_sensitivity_rows.append(
                            {
                                "shots": int(shots),
                                "draw": int(draw),
                                "context_seed": int(context_seed),
                                "elapsed_seconds": elapsed,
                                "base_accuracy": context_metrics["base_accuracy"],
                                "enrolled_accuracy": context_metrics[
                                    "enrolled_accuracy"
                                ],
                                "harmonic_mean": context_metrics["harmonic_mean"],
                                "accuracy": context_metrics["accuracy"],
                                "worst_enrolled_recall": context_metrics[
                                    "worst_enrolled_recall"
                                ],
                            }
                        )
                    for method in (
                        "logistic_regression",
                        "linear_svm",
                        "shrinkage_lda",
                    ):
                        method_started = time.perf_counter()
                        context_probabilities[method].append(
                            _classical_probability(
                                method, context_x, context_y, query_x
                            )
                        )
                        context_seconds[method] += (
                            time.perf_counter() - method_started
                        )
                for method, probabilities in context_probabilities.items():
                    if not probabilities:
                        continue
                    mean_probability = np.mean(probabilities, axis=0)
                    row = metric_row(
                        labels=query_y,
                        probability=mean_probability,
                        base_class_ids=base_ids,
                        enrolled_class_ids=pair,
                        method=method,
                        shots=shots,
                        draw=draw,
                        elapsed_seconds=context_seconds[method],
                        probability_source=(
                            "native_tabpfn_probability_ensemble"
                            if method == "tabpfn_v2"
                            else "native_or_fixed_softmax_score_ensemble"
                        ),
                        extra={
                            "base_class_ids": list(base_ids),
                            "enrolled_class_ids": list(pair),
                            "ensemble_contexts": 3,
                            "context_examples": int(N_CLASSES * shots),
                            "query_examples": int(len(query_y)),
                        },
                    )
                    append_evidence_row(row, mean_probability)

                for method, reference in raw_references.items():
                    method_started = time.perf_counter()
                    support_distance = _support_distances(
                        reference,
                        query_raw,
                        support_raw,
                        support_y,
                        enrolled_ids=pair,
                    )
                    temperature = 0.1 if method == "raw_cosine_1nn" else 1.0
                    probability = _distance_probability(
                        raw_base_distances[method],
                        support_distance,
                        base_ids=base_ids,
                        enrolled_ids=pair,
                        temperature=temperature,
                    )
                    row = metric_row(
                        labels=query_y,
                        probability=probability,
                        base_class_ids=base_ids,
                        enrolled_class_ids=pair,
                        method=method,
                        shots=shots,
                        draw=draw,
                        elapsed_seconds=time.perf_counter() - method_started,
                        probability_source=f"fixed_distance_softmax_t{temperature}",
                        extra={
                            "base_class_ids": list(base_ids),
                            "enrolled_class_ids": list(pair),
                            "query_examples": int(len(query_y)),
                        },
                    )
                    append_evidence_row(row, probability)

                if cfe is not None:
                    support_embedding = reference_embedding[positions]
                    cfe_variants = (
                        ("cfe_uncalibrated_mean", "mean", 0.0),
                        ("cfe_finalist", "quality_weighted", 0.2),
                    )
                    for method, prototype_method, teen_alpha in cfe_variants:
                        method_started = time.perf_counter()
                        session = EnrollmentSession.from_base(
                            cfe["train_embedding"],
                            cfe["train_labels"],
                            metric="cosine",
                        )
                        for class_id in pair:
                            mask = support_y == class_id
                            session = session.enroll(
                                class_id,
                                support_embedding[mask],
                                method=prototype_method,
                                quality=(
                                    cfe["reference_competence"][positions[mask]]
                                    if prototype_method == "quality_weighted"
                                    else None
                                ),
                                teen_alpha=teen_alpha,
                                teen_temperature=0.5,
                                support_group_ids=tuple(
                                    support_groups[index]
                                    for index in np.flatnonzero(mask)
                                ),
                            )
                        probability = session.predict_proba(
                            query_embedding, temperature=0.1
                        )
                        row = metric_row(
                            labels=query_y,
                            probability=probability,
                            base_class_ids=base_ids,
                            enrolled_class_ids=pair,
                            method=method,
                            shots=shots,
                            draw=draw,
                            elapsed_seconds=time.perf_counter() - method_started,
                            probability_source="fixed_cosine_distance_softmax_t0.1",
                            extra={
                                "base_class_ids": list(base_ids),
                                "enrolled_class_ids": list(pair),
                                "prototype_method": prototype_method,
                                "teen_alpha": teen_alpha,
                                "storage_bytes": session.storage_bytes,
                                "query_examples": int(len(query_y)),
                            },
                        )
                        append_evidence_row(row, probability)

                    method_started = time.perf_counter()
                    encoder_support_distance = _support_distances(
                        encoder_reference,
                        query_embedding,
                        support_embedding,
                        support_y,
                        enrolled_ids=pair,
                    )
                    probability = _distance_probability(
                        encoder_base_distance,
                        encoder_support_distance,
                        base_ids=base_ids,
                        enrolled_ids=pair,
                        temperature=0.1,
                    )
                    row = metric_row(
                        labels=query_y,
                        probability=probability,
                        base_class_ids=base_ids,
                        enrolled_class_ids=pair,
                        method="encoder_cosine_1nn",
                        shots=shots,
                        draw=draw,
                        elapsed_seconds=time.perf_counter() - method_started,
                        probability_source="fixed_cosine_distance_softmax_t0.1",
                        extra={
                            "base_class_ids": list(base_ids),
                            "enrolled_class_ids": list(pair),
                            "query_examples": int(len(query_y)),
                        },
                    )
                    append_evidence_row(row, probability)

        expected_methods = set(METHODS)
        if cfe is None:
            expected_methods -= {
                "cfe_finalist",
                "cfe_uncalibrated_mean",
                "encoder_cosine_1nn",
            }
        if not include_tabpfn:
            expected_methods.remove("tabpfn_v2")
        expected_rows = len(expected_methods) * len(shots_values) * draws
        if len(rows) != expected_rows or {row["method"] for row in rows} != expected_methods:
            raise AssertionError(
                f"Unit produced {len(rows)} rows for {sorted({r['method'] for r in rows})}; "
                f"expected {expected_rows} for {sorted(expected_methods)}."
            )
        if (
            len(evidence_row_ids) != len(rows)
            or len(evidence_predictions) != len(rows)
            or len(evidence_probabilities) != len(rows)
        ):
            raise AssertionError("Per-example evidence does not align with metric rows.")
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(cuda))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(cuda))
        if include_tabpfn and peak_allocated_bytes <= 0:
            raise RuntimeError(
                "TabPFN unit produced output without observed CUDA allocation."
            )
        weight_hashes = {}
        if resolved_model_path is not None:
            weight_hashes[str(resolved_model_path)] = file_sha256(resolved_model_path)
            for model in tabpfn_models:
                for path in _weight_paths(model):
                    weight_hashes[str(path)] = file_sha256(path)
        result = {
            "schema_version": 1,
            "evidence_schema": 3,
            "run_id": unit_id,
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "config_hash": unit_config_hash,
            "unit_config": unit_config,
            "dataset_sha256": file_sha256(DATA_PATH),
            "regime": regime,
            "pair": list(pair),
            "seed": int(seed),
            "requested_draws": int(draws),
            "requested_shots": list(shots_values),
            "include_tabpfn": bool(include_tabpfn),
            "methods": sorted(expected_methods),
            "query_examples": int(len(query_y)),
            "query_class_counts": {
                str(class_id): int((query_y == class_id).sum())
                for class_id in range(N_CLASSES)
            },
            "rows": rows,
            "context_sensitivity_rows": context_sensitivity_rows,
            "duration_seconds": time.perf_counter() - started,
            "device": str(cuda),
            "environment": environment_metadata(cuda),
            "tabpfn_package_version": (
                importlib.metadata.version("tabpfn") if include_tabpfn else None
            ),
            "tabpfn_model_name": resolved_model_name,
            "tabpfn_checkpoint_resolution_seconds": checkpoint_resolution_seconds,
            "tabpfn_model_initialization_seconds": model_initialization_seconds,
            "tabpfn_first_context_seconds_including_lazy_load": first_tabpfn_context_seconds,
            "tabpfn_checkpoint_size_bytes": (
                resolved_model_path.stat().st_size
                if resolved_model_path is not None
                else None
            ),
            "cuda_diagnostics": {
                "actual_device": str(cuda),
                "current_device_index": int(torch.cuda.current_device()),
                "compute_capability": list(
                    torch.cuda.get_device_capability(cuda)
                ),
                "peak_allocated_bytes": int(
                    peak_allocated_bytes
                ),
                "peak_reserved_bytes": int(
                    peak_reserved_bytes
                ),
                "amp_policy": "TabPFN internal policy; CFE reconstruction uses lifecycle bfloat16/float16 autocast",
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": bool(
                    torch.backends.cudnn.deterministic
                ),
            },
            "weight_hashes": weight_hashes,
            "cfe_source": (
                {
                    key: cfe[key]
                    for key in (
                        "source_dir",
                        "source_manifest_sha256",
                        "source_checkpoint_sha256",
                        "embedding_source",
                        "embedding_inference_seconds",
                        "embedding_device",
                    )
                }
                if cfe is not None
                else None
            ),
            "source": _git_metadata(REPOSITORY_ROOT),
        }
        atomic_json(unit_root / "metrics.json", result)
        np.savez_compressed(
            unit_root / "prediction_evidence.npz",
            labels=query_y.astype(np.uint8),
            query_group_ids=np.asarray(query_groups),
            row_ids=np.asarray(evidence_row_ids),
            predictions=np.stack(evidence_predictions),
            probabilities=np.stack(evidence_probabilities),
        )
        atomic_json(unit_root / "support_manifest.json", support_manifest)
        atomic_json(unit_root / "context_manifest.json", context_manifest)
        atomic_json(
            unit_root / "query_manifest.json",
            {
                "groups": list(query_groups),
                "labels": query_y.tolist(),
                "selection_namespace": "tabpfn-query",
                "cap_per_class": int(protocol["query_examples_per_class_cap"]),
                "used_for_fitting_or_selection": False,
            },
        )
        atomic_json(
            unit_root / "split_manifest.json",
            lifecycle_split_manifest(
                fold.split, data_path=DATA_PATH, regime=regime
            ),
        )
        write_manifest(unit_root, {**expected, "completed": True, "device": str(cuda)})
        append_jsonl(
            STUDY_ROOT / "experiment_registry.jsonl",
            {
                "event": "completed",
                "timestamp": utc_now(),
                "run_id": unit_id,
                "stage": (
                    "full_benchmark"
                    if regime == "full"
                    else "summary_only"
                ),
                "config_hash": unit_config_hash,
                "regime": regime,
                "pair": list(pair),
                "seed": seed,
                "device": str(cuda),
                "duration_seconds": result["duration_seconds"],
                "rows": len(rows),
            },
        )
        _update_state(status="running", unit_id=unit_id)
        del tabpfn_models
        torch.cuda.empty_cache()
        return result
    except Exception as exc:
        failure = {
            "event": "failed",
            "timestamp": utc_now(),
            "run_id": unit_id,
            "regime": regime,
            "pair": list(pair),
            "seed": seed,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(STUDY_ROOT / "failures.jsonl", failure)
        append_jsonl(STUDY_ROOT / "experiment_registry.jsonl", failure)
        _update_state(status="running_with_failures", unit_id=unit_id, failed=True)
        raise


def run_matrix(
    *,
    regime: str,
    device: str = "cuda:0",
    pair_limit: int | None = None,
    seed_limit: int | None = None,
    draws: int | None = None,
    shots_values: tuple[int, ...] | None = None,
    include_tabpfn: bool = True,
) -> dict[str, Any]:
    protocol = load_protocol()
    frame = pd.read_csv(DATA_PATH)
    pairs = [tuple(pair) for pair in protocol["pairs"]]
    seeds = [int(seed) for seed in protocol["seeds"]]
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    if seed_limit is not None:
        seeds = seeds[:seed_limit]
    requested_draws = int(protocol["draws"] if draws is None else draws)
    requested_shots = tuple(
        int(value)
        for value in (
            protocol["shots"] if shots_values is None else shots_values
        )
    )
    results = []
    for pair, seed in itertools.product(pairs, seeds):
        results.append(
            run_unit(
                frame=frame,
                regime=regime,
                pair=pair,
                seed=seed,
                device=device,
                shots_values=requested_shots,
                draws=requested_draws,
                include_tabpfn=include_tabpfn,
            )
        )
    return {
        "regime": regime,
        "units": len(results),
        "rows": sum(len(result["rows"]) for result in results),
        "device": device,
    }


def _parse_pair(value: str) -> tuple[int, int]:
    parts = tuple(int(item) for item in value.replace("_", "-").split("-"))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Pair must look like 1-2.")
    return parts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--device", default="cuda:0")
    unit = subparsers.add_parser("unit")
    unit.add_argument("--regime", choices=("full", "summary_only"), default="full")
    unit.add_argument("--pair", type=_parse_pair, required=True)
    unit.add_argument("--seed", type=int, required=True)
    unit.add_argument("--draws", type=int, default=20)
    unit.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5))
    unit.add_argument("--device", default="cuda:0")
    unit.add_argument("--without-tabpfn", action="store_true")
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--regime", choices=("full", "summary_only"), default="full")
    matrix.add_argument("--pair-limit", type=int)
    matrix.add_argument("--seed-limit", type=int)
    matrix.add_argument("--draws", type=int)
    matrix.add_argument("--shots", type=int, nargs="+")
    matrix.add_argument("--device", default="cuda:0")
    matrix.add_argument("--without-tabpfn", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "audit":
        protocol = load_protocol()
        device = require_cuda(args.device)
        source_index = _source_run_index()
        print(
            json.dumps(
                {
                    "protocol_sha256": file_sha256(PROTOCOL_PATH),
                    "dataset_sha256": file_sha256(DATA_PATH),
                    "pairs": len(protocol["pairs"]),
                    "seeds": len(protocol["seeds"]),
                    "frozen_cfe_sources": len(source_index),
                    "frozen_full_sources": sum(
                        key[0] == "full" for key in source_index
                    ),
                    "frozen_summary_sources": sum(
                        key[0] == "summary_only" for key in source_index
                    ),
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device),
                    "tabpfn_version": importlib.metadata.version("tabpfn"),
                },
                indent=2,
            )
        )
    elif args.command == "unit":
        result = run_unit(
            frame=pd.read_csv(DATA_PATH),
            regime=args.regime,
            pair=args.pair,
            seed=args.seed,
            device=args.device,
            shots_values=tuple(args.shots),
            draws=args.draws,
            include_tabpfn=not args.without_tabpfn,
        )
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "rows": len(result["rows"]),
                    "duration_seconds": result["duration_seconds"],
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                run_matrix(
                    regime=args.regime,
                    device=args.device,
                    pair_limit=args.pair_limit,
                    seed_limit=args.seed_limit,
                    draws=args.draws,
                    shots_values=tuple(args.shots) if args.shots else None,
                    include_tabpfn=not args.without_tabpfn,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
