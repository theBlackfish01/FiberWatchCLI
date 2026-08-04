from __future__ import annotations

"""Leakage-safe external validation for MCF-OTDR compatible tasks."""

from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .event_openworld_baselines import closed_set_group_split
from .event_openworld_external import load_measured_external, load_simulated_external
from .event_openworld_metrics import raw_partial_auroc
from .lifecycle_data import (
    LifecycleBatch,
    LifecycleScaler,
    fit_lifecycle_scaler,
    transform_lifecycle,
)
from .lifecycle_domain import (
    DomainAlignmentConfig,
    finetune_source_domain_alignment,
    finetune_unlabeled_target_alignment,
    propose_event,
)
from .lifecycle_metrics import classification_metrics
from .lifecycle_experiment import _git_metadata
from .lifecycle_training import (
    LifecycleTrainingConfig,
    infer_lifecycle_model,
    train_lifecycle_model,
)
from .model_functions.lifecycle import LifecycleModelConfig
from .model_functions.zero_shot import require_cuda
from .study_state import (
    atomic_json,
    append_jsonl,
    environment_metadata,
    file_sha256,
    validate_run,
    write_manifest,
    utc_now,
)
from .zero_shot_data import INPUT_COLUMNS


def external_batch(
    features: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    lifecycle_scaler: LifecycleScaler | None = None,
    source_snr_mean: float | None = None,
    source_snr_scale: float | None = None,
    trace_mode: str = "window_standardized",
) -> LifecycleBatch:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 31:
        raise ValueError("External features must contain standardized SNR plus 30 samples.")
    if trace_mode not in {"window_standardized", "source_range_aligned"}:
        raise ValueError("Unknown external trace preprocessing mode.")
    if trace_mode == "source_range_aligned" and lifecycle_scaler is None:
        raise ValueError("source_range_aligned requires the lifecycle scaler.")
    context = np.zeros((len(values), 3), dtype=np.float32)
    trace = values[:, 1:].copy()
    if lifecycle_scaler is None:
        context[:, 0] = values[:, 0]
    else:
        if source_snr_mean is None or source_snr_scale is None:
            raise ValueError(
                "External SNR conversion requires its source mean and scale."
            )
        raw_snr = (
            values[:, 0] * float(source_snr_scale)
            + float(source_snr_mean)
        )
        context[:, 0] = (
            raw_snr - float(lifecycle_scaler.context_location[0])
        ) / float(lifecycle_scaler.context_scale[0])
        for column in (1, 2):
            context[:, column] = (
                float(lifecycle_scaler.context_median[column])
                - float(lifecycle_scaler.context_location[column])
            ) / float(lifecycle_scaler.context_scale[column])
        if trace_mode == "source_range_aligned":
            minimum = trace.min(1, keepdims=True)
            span = np.clip(
                trace.max(1, keepdims=True) - minimum, 1e-8, None
            )
            trace = (trace - minimum) / span
            trace = (
                trace - np.asarray(
                    lifecycle_scaler.trace_location, dtype=np.float32
                )
            ) / np.asarray(
                lifecycle_scaler.trace_scale, dtype=np.float32
            )
    missing = np.zeros_like(context)
    missing[:, 1:] = 1
    group_ids = tuple(
        hashlib.sha256(f"{row.get('file')}:{row.get('center')}:{index}".encode()).hexdigest()
        for index, row in enumerate(metadata)
    )
    return LifecycleBatch(
        trace=torch.from_numpy(trace.astype(np.float32, copy=False)),
        context=torch.from_numpy(context),
        context_missing=torch.from_numpy(missing),
        labels=torch.zeros(len(values), dtype=torch.long),
        position=torch.full((len(values),), float("nan")),
        group_ids=group_ids,
    )


def _threshold_by_balanced_accuracy(score: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(np.unique(y)) < 2:
        raise ValueError("Threshold calibration requires event and no-event examples.")
    candidates = np.unique(np.quantile(values, np.linspace(0.02, 0.98, 97)))
    performance = [balanced_accuracy_score(y, values >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(performance))])


def _compatible_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    traces: np.ndarray,
    *,
    threshold: float,
    class_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    event = np.asarray(labels) > 0
    prediction = score >= threshold
    centers = np.asarray([propose_event(trace).center for trace in traces])
    clipped_score = np.clip(np.asarray(score, dtype=float), 1e-8, 1 - 1e-8)
    confidence = np.maximum(clipped_score, 1 - clipped_score)
    correct = prediction == event
    edges = np.linspace(0, 1, 16)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            ece += mask.mean() * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    location_error = centers[event] - 14.5
    result: dict[str, Any] = {
        "examples": len(labels),
        "event_examples": int(event.sum()),
        "no_event_examples": int((~event).sum()),
        "event_auroc": float(roc_auc_score(event, score)),
        "event_aupr": float(average_precision_score(event, score)),
        "event_pauroc_0_05": raw_partial_auroc(event, score, 0.05),
        "balanced_accuracy": float(balanced_accuracy_score(event, prediction)),
        "macro_f1": float(f1_score(event, prediction, average="macro")),
        "observed_no_event_far": float(prediction[~event].mean()),
        "event_recall": float(prediction[event].mean()),
        "binary_nll": float(
            -np.mean(
                event * np.log(clipped_score)
                + (~event) * np.log(1 - clipped_score)
            )
        ),
        "binary_brier": float(np.square(clipped_score - event).mean()),
        "binary_ece_15": float(ece),
        "proposal_location_mae_bins": float(np.abs(location_error).mean()),
        "proposal_location_rmse_bins": float(np.sqrt(np.square(location_error).mean())),
        "threshold": threshold,
        "subgroups": {},
    }
    if class_logits is not None and {1, 2}.issubset(set(np.unique(labels))):
        logits = np.asarray(class_logits, dtype=float)
        event_rows = np.isin(labels, (1, 2))
        reflective_local = logits[:, [4, 6, 7]].max(1)
        attenuating_local = logits[:, [1, 2, 3, 5]].max(1)
        result["reflection_vs_attenuation_auroc"] = float(
            roc_auc_score(
                np.asarray(labels)[event_rows] == 1,
                (reflective_local - attenuating_local)[event_rows],
            )
        )
        result["reflection_attenuation_protocol"] = (
            "compatible physical ranking only; external labels are not mapped "
            "into local fault classes"
        )
    for field in ("wavelength_nm", "pulse_width_ns", "averaging_time_s"):
        rows = {}
        for value in sorted({item[field] for item in metadata if field in item}):
            mask = np.asarray([item.get(field) == value for item in metadata])
            if mask.sum() >= 4 and len(np.unique(event[mask])) == 2:
                rows[str(value)] = {
                    "n": int(mask.sum()),
                    "event_auroc": float(roc_auc_score(event[mask], score[mask])),
                    "event_aupr": float(average_precision_score(event[mask], score[mask])),
                }
        if rows:
            result["subgroups"][field] = rows
    return result


def _few_shot_calibration(
    score: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    cohort: str,
) -> list[dict[str, Any]]:
    event = np.asarray(labels) > 0
    files = sorted({str(row["file"]) for row in metadata})
    rows = []
    for groups in (1, 5, 10, 20):
        if len(files) <= groups:
            continue
        for seed in (42, 123, 2026, 7, 31415):
            ranked = sorted(files, key=lambda value: hashlib.sha256(f"external:{seed}:{value}".encode()).hexdigest())
            selected = set(ranked[:groups])
            calibration = np.asarray([str(row["file"]) in selected for row in metadata])
            test = ~calibration
            if len(np.unique(event[calibration])) < 2 or len(np.unique(event[test])) < 2:
                continue
            threshold = _threshold_by_balanced_accuracy(score[calibration], event[calibration])
            prediction = score[test] >= threshold
            rows.append({
                "cohort": cohort,
                "calibration_groups": groups,
                "seed": seed,
                "calibration_examples": int(calibration.sum()),
                "test_examples": int(test.sum()),
                "threshold": threshold,
                "test_event_auroc": float(roc_auc_score(event[test], score[test])),
                "test_balanced_accuracy": float(balanced_accuracy_score(event[test], prediction)),
                "test_no_event_far": float(prediction[~event[test]].mean()),
                "test_event_recall": float(prediction[event[test]].mean()),
                "group_disjoint": True,
            })
    return rows


def run_external_lifecycle_validation(
    *,
    frame: pd.DataFrame,
    data_path: Path,
    external_root: Path,
    study_root: Path,
    model_config: LifecycleModelConfig,
    training_config: LifecycleTrainingConfig,
    device: torch.device | str,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    output_root = study_root / "external"
    valid, _ = validate_run(
        output_root,
        expected={"run_id": "lifecycle-external-validation-v1"},
    )
    if valid:
        return json.loads(
            (output_root / "metrics.json").read_text(encoding="utf-8")
        )
    environment = environment_metadata(device)
    provenance = {
        "dataset_sha256": file_sha256(data_path),
        "source": _git_metadata(Path(__file__).resolve().parents[3]),
        "environment": environment,
    }
    stage_started = time.perf_counter()
    append_jsonl(
        study_root / "experiment_registry.jsonl",
        {
            "event": "started",
            "run_id": "lifecycle-external-validation-v1",
            "stage": "external",
            "timestamp": utc_now(),
            "device": str(device),
        },
    )
    train, validation, source_test = closed_set_group_split(frame)
    lifecycle_scaler = fit_lifecycle_scaler(train, regime="full")
    train_batch = transform_lifecycle(train, lifecycle_scaler)
    validation_batch = transform_lifecycle(validation, lifecycle_scaler)
    source_test_batch = transform_lifecycle(source_test, lifecycle_scaler)

    # Existing loader performs its declared window normalization and estimates SNR.
    trace_scaler = StandardScaler().fit(train[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    simulated_x, simulated_y, simulated_meta = load_simulated_external(external_root, trace_scaler)
    measured_x, measured_y, measured_meta = load_measured_external(external_root, trace_scaler)
    cohorts = {
        "simulated": (simulated_x, simulated_y, simulated_meta),
        "measured": (measured_x, measured_y, measured_meta),
    }
    def make_external_batch(
        features: np.ndarray,
        metadata: list[dict[str, Any]],
        *,
        mode: str = "source_range_aligned",
    ) -> LifecycleBatch:
        return external_batch(
            features,
            metadata,
            lifecycle_scaler=lifecycle_scaler,
            source_snr_mean=float(trace_scaler.mean_[0]),
            source_snr_scale=float(trace_scaler.scale_[0]),
            trace_mode=mode,
        )
    variants = {
        "matched_full_late_fusion": model_config,
        "morphology_only": replace(model_config, mode="morphology_only"),
        "context_only": replace(model_config, mode="context_only"),
        "mcf_canonicalized_fusion": replace(model_config, mode="late_fusion", canonicalize=True),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "task_definition": "compatible external event/no-event; no mapping to local seven-fault taxonomy",
        "preprocessing_contract": {
            "primary": "source_range_aligned",
            "primary_description": (
                "Label-free per-window min-max alignment to the local [0,1] "
                "trace convention, followed by the source-fitted lifecycle "
                "scaler; external SNR is converted back to physical units and "
                "then transformed with the lifecycle robust scaler."
            ),
            "sensitivity": "window_standardized",
            "selection_uses_external_labels": False,
        },
        "zero_target": {},
        "unsupervised_target_adaptation": {},
        "few_shot_target_calibration": [],
        **provenance,
    }
    predictions: dict[str, np.ndarray] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    trained: dict[str, tuple[torch.nn.Module, dict[str, Any], LifecycleModelConfig]] = {}
    for variant, variant_config in variants.items():
        config = replace(training_config, seed=42)
        model, training = train_lifecycle_model(
            train_batch, validation_batch, device=device,
            model_config=variant_config, training_config=config,
        )
        trained[variant] = (model, training, variant_config)
    base_model = trained["matched_full_late_fusion"][0]
    for method in ("coral", "mmd"):
        model, training = finetune_source_domain_alignment(
            copy.deepcopy(base_model),
            train_batch,
            device=device,
            config=DomainAlignmentConfig(method=method, seed=42),
        )
        trained[f"mcf_{method}_aligned"] = (model, training, model_config)
    for variant, (model, training, variant_config) in trained.items():
        config = replace(training_config, seed=42)
        torch.save({
            "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
            "model_config": asdict(variant_config),
            "training_config": asdict(config),
            "scaler": lifecycle_scaler.payload(),
        }, output_root / f"{variant}.pt")
        validation_output = infer_lifecycle_model(model, validation_batch, device=device)
        validation_probability = validation_output["logits"].softmax(1).numpy()
        validation_event_score = 1 - validation_probability[:, 0]
        validation_normal = validation_batch.labels.numpy() == 0
        source_threshold = float(np.quantile(
            validation_event_score[validation_normal], 0.99, method="higher"
        ))
        variant_result = {
            "training": training,
            "source_test": classification_metrics(
                infer_lifecycle_model(model, source_test_batch, device=device)["logits"].numpy(),
                source_test_batch.labels.numpy(),
            ),
            "cohorts": {},
        }
        for cohort, (features, labels, metadata) in cohorts.items():
            batch = make_external_batch(features, metadata)
            output = infer_lifecycle_model(model, batch, device=device)
            probability = output["logits"].softmax(1).numpy()
            score = 1 - probability[:, 0]
            variant_result["cohorts"][cohort] = _compatible_metrics(
                score,
                labels,
                metadata,
                features[:, 1:],
                threshold=source_threshold,
                class_logits=output["logits"].numpy(),
            )
            result["few_shot_target_calibration"].extend([
                {"variant": variant, **row}
                for row in _few_shot_calibration(score, labels, metadata, cohort=cohort)
            ])
            predictions[f"{variant}_{cohort}_score"] = score.astype(np.float32)
            predictions[f"{cohort}_labels"] = np.asarray(labels, dtype=np.int8)
        result["zero_target"][variant] = variant_result

    # Predeclared preprocessing sensitivity on the identical matched model.
    direct_name = "matched_full_late_fusion_window_standardized"
    direct_model, direct_training, _ = trained["matched_full_late_fusion"]
    validation_output = infer_lifecycle_model(
        direct_model, validation_batch, device=device
    )
    validation_event_score = (
        1 - validation_output["logits"].softmax(1).numpy()[:, 0]
    )
    validation_normal = validation_batch.labels.numpy() == 0
    direct_threshold = float(np.quantile(
        validation_event_score[validation_normal], 0.99, method="higher"
    ))
    direct_result = {
        "training": direct_training,
        "source_test": classification_metrics(
            infer_lifecycle_model(
                direct_model, source_test_batch, device=device
            )["logits"].numpy(),
            source_test_batch.labels.numpy(),
        ),
        "preprocessing_sensitivity_only": True,
        "cohorts": {},
    }
    for cohort, (features, labels, metadata) in cohorts.items():
        batch = make_external_batch(
            features, metadata, mode="window_standardized"
        )
        output = infer_lifecycle_model(direct_model, batch, device=device)
        score = 1 - output["logits"].softmax(1).numpy()[:, 0]
        direct_result["cohorts"][cohort] = _compatible_metrics(
            score,
            labels,
            metadata,
            features[:, 1:],
            threshold=direct_threshold,
            class_logits=output["logits"].numpy(),
        )
        result["few_shot_target_calibration"].extend([
            {"variant": direct_name, **row}
            for row in _few_shot_calibration(
                score, labels, metadata, cohort=cohort
            )
        ])
        predictions[f"{direct_name}_{cohort}_score"] = score.astype(
            np.float32
        )
    result["zero_target"][direct_name] = direct_result

    # Declared transductive arm: each target cohort is adapted and evaluated as
    # an unlabeled mixture. It is never pooled with zero-target results.
    for cohort, (features, labels, metadata) in cohorts.items():
        target_batch = make_external_batch(features, metadata)
        adapted, adaptation = finetune_unlabeled_target_alignment(
            copy.deepcopy(base_model),
            train_batch,
            target_batch,
            device=device,
            config=DomainAlignmentConfig(method="coral", steps=40, seed=42),
        )
        validation_output = infer_lifecycle_model(
            adapted, validation_batch, device=device
        )
        validation_event_score = (
            1 - validation_output["logits"].softmax(1).numpy()[:, 0]
        )
        validation_normal = validation_batch.labels.numpy() == 0
        threshold = float(np.quantile(
            validation_event_score[validation_normal], 0.99, method="higher"
        ))
        output = infer_lifecycle_model(adapted, target_batch, device=device)
        score = 1 - output["logits"].softmax(1).numpy()[:, 0]
        result["unsupervised_target_adaptation"][cohort] = {
            "variant": "matched_full_late_fusion_coral",
            "protocol": "transductive_unlabeled_target_mixture",
            "adaptation": adaptation,
            "metrics": _compatible_metrics(
                score,
                labels,
                metadata,
                features[:, 1:],
                threshold=threshold,
                class_logits=output["logits"].numpy(),
            ),
        }
        predictions[f"transductive_coral_{cohort}_score"] = score.astype(np.float32)
    atomic_json(output_root / "metrics.json", result)
    np.savez_compressed(output_root / "predictions.npz", **predictions)
    write_manifest(output_root, {
        "run_id": "lifecycle-external-validation-v1",
        "completed": True,
        "device": str(device),
        "taxonomy_mapping": False,
        "source_only_architecture_selection": True,
        **provenance,
    })
    append_jsonl(
        study_root / "experiment_registry.jsonl",
        {
            "event": "completed",
            "run_id": "lifecycle-external-validation-v1",
            "stage": "external",
            "timestamp": utc_now(),
            "device": str(device),
            "duration_seconds": time.perf_counter() - stage_started,
        },
    )
    return result
