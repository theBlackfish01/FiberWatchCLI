from __future__ import annotations

"""Representative-pair acquisition and feature-degradation validation."""

from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .lifecycle_data import fit_lifecycle_fold, transform_lifecycle
from .lifecycle_domain import (
    DomainAlignmentConfig,
    apply_stress,
    finetune_source_domain_alignment,
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


STRESS_KINDS = (
    "snr_noise", "amplitude_scale", "amplitude_offset", "event_width",
    "position_shift", "resampling", "structured_noise", "loss_noise",
    "reflectance_noise", "missing_loss", "missing_reflectance",
    "scalar_quantization",
)


def _stressed_frame(
    frame: pd.DataFrame,
    *,
    kind: str,
    severity: float,
    seed: int,
) -> pd.DataFrame:
    trace = frame[[f"P{i}" for i in range(1, 31)]].to_numpy(dtype=np.float32)
    context = frame[["SNR", "loss", "Reflectance"]].to_numpy(dtype=np.float32)
    stressed_trace, stressed_context, _ = apply_stress(
        trace, context, kind=kind, severity=severity, seed=seed
    )
    result = frame.copy()
    result[[f"P{i}" for i in range(1, 31)]] = stressed_trace
    result[["SNR", "loss", "Reflectance"]] = stressed_context
    return result


def _feature_ablation_frame(
    frame: pd.DataFrame,
    name: str,
    *,
    scaler,
) -> pd.DataFrame:
    result = frame.copy()
    if name in {"loss_removed", "both_summaries_removed", "scalar_branch_removed"}:
        result["loss"] = np.nan
    if name in {"reflectance_removed", "both_summaries_removed", "scalar_branch_removed"}:
        result["Reflectance"] = np.nan
    if name == "scalar_branch_removed":
        result["SNR"] = float(scaler.context_median[0])
    if name == "waveform_branch_removed":
        for index, column in enumerate(
            (f"P{i}" for i in range(1, 31))
        ):
            result[column] = float(scaler.trace_location[index])
    return result


def _permuted_frame(frame: pd.DataFrame, feature: str, *, seed: int) -> pd.DataFrame:
    """Conditionally permute a scalar inside SNR quintiles."""
    result = frame.copy()
    rng = np.random.default_rng(seed)
    bins = pd.qcut(result["SNR"], q=5, labels=False, duplicates="drop")
    for value in sorted(bins.dropna().unique()):
        indices = np.flatnonzero(bins.to_numpy() == value)
        result.iloc[indices, result.columns.get_loc(feature)] = rng.permutation(
            result.iloc[indices][feature].to_numpy()
        )
    return result


def run_stress_validation(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    model_config: LifecycleModelConfig,
    training_config: LifecycleTrainingConfig,
    device: torch.device | str,
    pairs: tuple[tuple[int, int], ...] = ((1, 2), (3, 5), (6, 7)),
) -> dict[str, Any]:
    device = require_cuda(str(device))
    root = study_root / "stress"
    valid, _ = validate_run(
        root,
        expected={"run_id": "lifecycle-stress-validation-v1"},
    )
    if valid:
        return json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    environment = environment_metadata(device)
    dataset_path = Path(__file__).resolve().parent / "data" / "OTDR_DATA.csv"
    provenance = {
        "dataset_sha256": file_sha256(dataset_path),
        "source": _git_metadata(Path(__file__).resolve().parents[3]),
        "environment": environment,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "pairs": [list(pair) for pair in pairs],
        **provenance,
        "rows": [],
        "feature_contribution": [],
        "training": {},
    }
    root.mkdir(parents=True, exist_ok=True)
    for pair_index, pair in enumerate(pairs):
        unit_root = root / f"pair_{pair[0]}_{pair[1]}"
        unit_run_id = f"lifecycle-stress-{pair[0]}_{pair[1]}"
        unit_valid, _ = validate_run(
            unit_root,
            expected={"run_id": unit_run_id},
        )
        if unit_valid:
            unit = json.loads(
                (unit_root / "metrics.json").read_text(encoding="utf-8")
            )
            result["rows"].extend(unit["rows"])
            result["feature_contribution"].extend(
                unit["feature_contribution"]
            )
            result["training"].update(unit["training"])
            continue
        unit_root.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            study_root / "experiment_registry.jsonl",
            {
                "event": "started",
                "run_id": unit_run_id,
                "stage": "stress",
                "timestamp": utc_now(),
                "device": str(device),
            },
        )
        row_start = len(result["rows"])
        feature_start = len(result["feature_contribution"])
        training_keys_before = set(result["training"])
        tensor_fold = fit_lifecycle_fold(frame, holdout=pair, seed=42, regime="full")
        trained: dict[str, tuple[torch.nn.Module, dict[str, Any], LifecycleModelConfig]] = {}
        for variant, variant_config in {
            "matched_late_fusion": model_config,
            "plain_tcn_mean_pool": replace(model_config, pooling="mean"),
            "compact_tcn_self_attention": replace(model_config, pooling="self_attention"),
            "morphology_only": replace(model_config, mode="morphology_only"),
            "context_only": replace(model_config, mode="context_only"),
            "mcf_canonicalized": replace(model_config, canonicalize=True),
        }.items():
            model, training = train_lifecycle_model(
                tensor_fold.batches["train"], tensor_fold.batches["validation"],
                device=device, model_config=variant_config,
                training_config=replace(training_config, seed=42),
            )
            trained[variant] = (model, training, variant_config)
        no_dropout_model, no_dropout_training = train_lifecycle_model(
            tensor_fold.batches["train"],
            tensor_fold.batches["validation"],
            device=device,
            model_config=model_config,
            training_config=replace(
                training_config, seed=42, scalar_dropout=0.0
            ),
        )
        trained["late_fusion_no_scalar_dropout"] = (
            no_dropout_model,
            no_dropout_training,
            model_config,
        )
        localization_model, localization_training = train_lifecycle_model(
            tensor_fold.batches["train"],
            tensor_fold.batches["validation"],
            device=device,
            model_config=model_config,
            training_config=replace(
                training_config, seed=42, localization_weight=0.03
            ),
        )
        trained["multitask_localization_weight_0_03"] = (
            localization_model,
            localization_training,
            model_config,
        )
        base_model = trained["matched_late_fusion"][0]
        for method in ("coral", "mmd"):
            model, training = finetune_source_domain_alignment(
                copy.deepcopy(base_model),
                tensor_fold.batches["train"],
                device=device,
                config=DomainAlignmentConfig(method=method, seed=42),
            )
            trained[f"mcf_{method}_aligned"] = (model, training, model_config)
        for variant, (model, training, variant_config) in trained.items():
            checkpoint = unit_root / f"{variant}_{pair[0]}_{pair[1]}.pt"
            torch.save({
                "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                "model_config": asdict(variant_config),
                "training_config": asdict(training_config),
                "scaler": tensor_fold.scaler.payload(),
            }, checkpoint)
            clean_output = infer_lifecycle_model(
                model, tensor_fold.batches["seen_test"], device=device
            )
            clean_metrics = classification_metrics(
                clean_output["logits"].numpy(),
                tensor_fold.batches["seen_test"].labels.numpy(),
                positions=tensor_fold.batches["seen_test"].position.numpy(),
                predicted_positions=clean_output["position"].numpy(),
            )
            clean_recall = {
                class_id: float(values["recall"])
                for class_id, values in clean_metrics["per_class"].items()
            }
            result["rows"].append({
                "pair": list(pair), "variant": variant, "stress": "clean",
                "severity": 0.0,
                **{
                    key: clean_metrics[key]
                    for key in (
                        "accuracy", "balanced_accuracy", "macro_f1", "ece_15",
                        "localization_mae", "localization_rmse",
                    )
                    if key in clean_metrics
                },
                "per_class_recall": clean_recall,
            })
            for kind in STRESS_KINDS:
                for severity in (0.25, 0.5, 0.75, 1.0):
                    stressed_frame = _stressed_frame(
                        tensor_fold.split.seen_test,
                        kind=kind, severity=severity,
                        seed=pair_index * 1000 + int(severity * 100) + len(kind),
                    )
                    stressed_batch = transform_lifecycle(stressed_frame, tensor_fold.scaler)
                    output = infer_lifecycle_model(model, stressed_batch, device=device)
                    metrics = classification_metrics(
                        output["logits"].numpy(), stressed_batch.labels.numpy()
                    )
                    result["rows"].append({
                        "pair": list(pair), "variant": variant, "stress": kind,
                        "severity": severity,
                        **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "ece_15")},
                        "balanced_accuracy_delta": float(metrics["balanced_accuracy"]) - float(clean_metrics["balanced_accuracy"]),
                        "ece_delta": float(metrics["ece_15"]) - float(
                            clean_metrics["ece_15"]
                        ),
                        "per_class_recall": {
                            class_id: float(values["recall"])
                            for class_id, values in metrics["per_class"].items()
                        },
                        "per_class_recall_delta": {
                            class_id: float(values["recall"])
                            - clean_recall[class_id]
                            for class_id, values in metrics["per_class"].items()
                        },
                    })
            for ablation in (
                "loss_removed", "reflectance_removed", "both_summaries_removed",
                "waveform_branch_removed", "scalar_branch_removed",
            ):
                ablated_frame = _feature_ablation_frame(
                    tensor_fold.split.seen_test,
                    ablation,
                    scaler=tensor_fold.scaler,
                )
                batch = transform_lifecycle(ablated_frame, tensor_fold.scaler)
                output = infer_lifecycle_model(model, batch, device=device)
                metrics = classification_metrics(output["logits"].numpy(), batch.labels.numpy())
                result["feature_contribution"].append({
                    "pair": list(pair), "variant": variant, "condition": ablation,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "ece_15": metrics["ece_15"],
                    "balanced_accuracy_delta": float(metrics["balanced_accuracy"]) - float(clean_metrics["balanced_accuracy"]),
                    "ece_delta": float(metrics["ece_15"]) - float(
                        clean_metrics["ece_15"]
                    ),
                    "per_class_recall": {
                        class_id: float(values["recall"])
                        for class_id, values in metrics["per_class"].items()
                    },
                    "per_class_recall_delta": {
                        class_id: float(values["recall"])
                        - clean_recall[class_id]
                        for class_id, values in metrics["per_class"].items()
                    },
                })
            for feature in ("loss", "Reflectance"):
                permuted = _permuted_frame(tensor_fold.split.seen_test, feature, seed=pair_index + 800)
                batch = transform_lifecycle(permuted, tensor_fold.scaler)
                output = infer_lifecycle_model(model, batch, device=device)
                metrics = classification_metrics(output["logits"].numpy(), batch.labels.numpy())
                result["feature_contribution"].append({
                    "pair": list(pair), "variant": variant,
                    "condition": f"conditional_permutation_{feature}",
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "ece_15": metrics["ece_15"],
                    "balanced_accuracy_delta": float(metrics["balanced_accuracy"]) - float(clean_metrics["balanced_accuracy"]),
                    "ece_delta": float(metrics["ece_15"]) - float(
                        clean_metrics["ece_15"]
                    ),
                    "per_class_recall": {
                        class_id: float(values["recall"])
                        for class_id, values in metrics["per_class"].items()
                    },
                    "per_class_recall_delta": {
                        class_id: float(values["recall"])
                        - clean_recall[class_id]
                        for class_id, values in metrics["per_class"].items()
                    },
                    "interpretation": "predictive importance, not causality",
                })
            result["training"][f"{variant}_{pair[0]}_{pair[1]}"] = training
        pair_training = {
            key: value
            for key, value in result["training"].items()
            if key not in training_keys_before
        }
        unit = {
            "schema_version": 1,
            "pair": list(pair),
            "rows": result["rows"][row_start:],
            "feature_contribution": result["feature_contribution"][
                feature_start:
            ],
            "training": pair_training,
            "environment": environment,
        }
        atomic_json(unit_root / "metrics.json", unit)
        write_manifest(
            unit_root,
            {
                "run_id": unit_run_id,
                "completed": True,
                "device": str(device),
                "pair": list(pair),
                **provenance,
            },
        )
        append_jsonl(
            study_root / "experiment_registry.jsonl",
            {
                "event": "completed",
                "run_id": unit_run_id,
                "stage": "stress",
                "timestamp": utc_now(),
                "device": str(device),
            },
        )
    atomic_json(root / "metrics.json", result)
    write_manifest(root, {
        "run_id": "lifecycle-stress-validation-v1",
        "completed": True,
        "device": str(device),
        "representative_pairs_only": True,
        **provenance,
    })
    return result
