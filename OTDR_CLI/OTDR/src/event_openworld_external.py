from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .event_openworld_baselines import closed_set_group_split
from .event_openworld_metrics import NormalOnlyCalibrator, ScoreNormalizer, raw_partial_auroc
from .event_openworld_training import ECConfig, PC2Config, infer_event_model, train_ec_czsl, train_pc2_oe
from .model_functions.event_openworld import load_event_recipes
from .model_functions.zero_shot import require_cuda
from .study_state import atomic_json, file_sha256
from .zero_shot_data import INPUT_COLUMNS


def _normalize_window(power: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    power = np.asarray(power, dtype=np.float32)
    median = float(np.median(power))
    mad = float(1.4826 * np.median(np.abs(power - median)))
    scale = mad if mad > 1e-7 else float(power.std()) or 1.0
    normalized = np.clip((power - median) / scale, -6, 6)
    noise = float(np.median(np.abs(np.diff(power) - np.median(np.diff(power))))) + 1e-8
    snr = float(np.clip(20 * np.log10((float(power.std()) + 1e-8) / noise), 4, 40))
    snr_standardized = (snr - float(scaler.mean_[0])) / max(float(scaler.scale_[0]), 1e-8)
    return np.r_[snr_standardized, normalized].astype(np.float32)


def _window(values: np.ndarray, center: int, width: int = 30) -> np.ndarray | None:
    left = center - width // 2
    right = left + width
    if left < 0 or right > len(values):
        return None
    return values[left:right]


def load_simulated_external(root: Path, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    folder = next(root.rglob("2023-12-08_simulated_measurements"))
    features, labels, metadata = [], [], []
    for path in sorted(folder.glob("*.tsv")):
        data = np.loadtxt(path, delimiter="\t")
        point_label = data[:, 2].astype(int)
        power = data[:, 3]
        event_indices = np.flatnonzero(point_label > 0)
        groups = np.split(event_indices, np.flatnonzero(np.diff(event_indices) > 1) + 1) if len(event_indices) else []
        occupied = np.zeros(len(power), dtype=bool)
        for group in groups:
            center = int(np.round(group.mean()))
            values = _window(power, center)
            if values is None:
                continue
            label = int(np.bincount(point_label[group]).argmax())
            features.append(_normalize_window(values, scaler)); labels.append(label)
            metadata.append({"file": path.name, "center": center, "kind": "event", "point_label": label})
            occupied[max(0, center - 25):min(len(power), center + 26)] = True
        candidates = np.flatnonzero(~occupied & (np.arange(len(power)) >= 15) & (np.arange(len(power)) < len(power) - 15))
        if len(candidates):
            ranked = candidates[np.linspace(0, len(candidates) - 1, 10, dtype=int)]
            for center in ranked:
                values = _window(power, int(center))
                features.append(_normalize_window(values, scaler)); labels.append(0)
                metadata.append({"file": path.name, "center": int(center), "kind": "no_event", "point_label": 0})
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), metadata


def load_measured_external(root: Path, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    import otdrs

    features, labels, metadata = [], [], []
    for path in sorted(root.rglob("*.sor")):
        sor = otdrs.parse_file(str(path))
        block = sor.data_points.scale_factors[0]
        raw = np.asarray(block.data, dtype=np.float32)
        power = (raw.max() - raw) * float(block.scale_factor) / 1_000_000
        acquisition_range = max(int(sor.fixed_parameters.acquisition_range), 1)
        events = list(sor.key_events.key_events) + [sor.key_events.last_key_event]
        occupied = np.zeros(len(power), dtype=bool)
        for event in events:
            center = int(round(int(event.event_propogation_time) / acquisition_range * (len(power) - 1)))
            values = _window(power, center)
            if values is None:
                continue
            features.append(_normalize_window(values, scaler)); labels.append(1)
            metadata.append({"file": path.name, "center": center, "kind": "event", "event_code": event.event_code,
                             "wavelength_nm": int(sor.general_parameters.nominal_wavelength),
                             "pulse_width_ns": int(sor.fixed_parameters.pulse_widths_used[0]),
                             "averaging_time_s": int(sor.fixed_parameters.averaging_time)})
            occupied[max(0, center - 40):min(len(power), center + 41)] = True
        candidates = np.flatnonzero(~occupied & (np.arange(len(power)) >= 15) & (np.arange(len(power)) < len(power) - 15))
        if len(candidates):
            for center in candidates[np.linspace(0, len(candidates) - 1, 7, dtype=int)]:
                values = _window(power, int(center))
                features.append(_normalize_window(values, scaler)); labels.append(0)
                metadata.append({"file": path.name, "center": int(center), "kind": "no_event",
                                 "wavelength_nm": int(sor.general_parameters.nominal_wavelength),
                                 "pulse_width_ns": int(sor.fixed_parameters.pulse_widths_used[0]),
                                 "averaging_time_s": int(sor.fixed_parameters.averaging_time)})
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), metadata


def _evaluate_external(
    outputs: dict[str, torch.Tensor],
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    novelty_score: np.ndarray,
) -> dict[str, Any]:
    event = labels > 0
    event_score = 1 - outputs["logits"].softmax(-1)[:, 0].numpy()
    prediction = (event_score >= 0.5).astype(int)
    result: dict[str, Any] = {
        "examples": len(labels), "event_examples": int(event.sum()), "no_event_examples": int((~event).sum()),
        "event_auroc": float(roc_auc_score(event, event_score)),
        "event_pauroc_0_05": raw_partial_auroc(event, event_score, 0.05),
        "event_balanced_accuracy_at_0_5": float(balanced_accuracy_score(event, prediction)),
        "event_location_mae_bins": float(np.abs(outputs["center"].numpy()[event] - 14.5).mean()),
        "zero_day_novelty_auroc_for_event_vs_no_event": float(roc_auc_score(event, novelty_score)),
        "zero_day_novelty_pauroc_0_05": raw_partial_auroc(event, novelty_score, 0.05),
    }
    if set(np.unique(labels[event])) >= {1, 2}:
        reflective = [4, 6, 7]
        attenuating = [1, 2, 3, 5]
        score = outputs["logits"][:, reflective].amax(1) - outputs["logits"][:, attenuating].amax(1)
        target = labels[event] == 1
        result["reflection_vs_attenuation_auroc_assuming_1_reflection_2_attenuation"] = float(
            roc_auc_score(target, score.numpy()[event])
        )
    result["subgroups"] = {}
    for field in ("wavelength_nm", "pulse_width_ns", "averaging_time_s"):
        values = sorted({row[field] for row in metadata if field in row})
        if values:
            field_rows = {}
            for value in values:
                mask = np.asarray([row.get(field) == value for row in metadata])
                if len(np.unique(event[mask])) == 2:
                    field_rows[str(value)] = {"n": int(mask.sum()), "event_auroc": float(roc_auc_score(event[mask], event_score[mask]))}
            result["subgroups"][field] = field_rows
    return result


def _calibration_transfer(
    calibrator: NormalOnlyCalibrator,
    score: np.ndarray,
    snr: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    event = labels > 0
    rows = {}
    for far in (0.01, 0.02, 0.05):
        rejected = score > calibrator.threshold(snr, far)
        rows[f"far_{far:.3f}"] = {
            "target_far": far,
            "observed_no_event_far": float(rejected[~event].mean()),
            "event_recall": float(rejected[event].mean()),
        }
    return rows


def _external_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for approach, payload in result["approaches"].items():
        for cohort in ("simulated", "measured"):
            metrics = payload[cohort]
            rows.append({
                "row_type": "performance", "approach": approach, "cohort": cohort,
                "examples": metrics["examples"], "event_examples": metrics["event_examples"],
                "no_event_examples": metrics["no_event_examples"], "event_auroc": metrics["event_auroc"],
                "event_pauroc_0_05": metrics["event_pauroc_0_05"],
                "event_balanced_accuracy_at_0_5": metrics["event_balanced_accuracy_at_0_5"],
                "event_location_mae_bins": metrics["event_location_mae_bins"],
                "zero_day_novelty_auroc": metrics["zero_day_novelty_auroc_for_event_vs_no_event"],
                "zero_day_novelty_pauroc_0_05": metrics["zero_day_novelty_pauroc_0_05"],
                "reflection_vs_attenuation_auroc": metrics.get(
                    "reflection_vs_attenuation_auroc_assuming_1_reflection_2_attenuation"
                ),
            })
        for cohort, operating_points in payload["local_normal_calibration_transfer"].items():
            for far_name, metrics in operating_points.items():
                rows.append({
                    "row_type": "calibration_transfer", "approach": approach, "cohort": cohort,
                    "calibration_source": "local_normal", "operating_point": far_name,
                    "target_far": metrics["target_far"], "observed_no_event_far": metrics["observed_no_event_far"],
                    "event_recall": metrics["event_recall"],
                })
        for far_name, metrics in payload["synthetic_no_event_to_measured_calibration_transfer"].items():
            rows.append({
                "row_type": "calibration_transfer", "approach": approach, "cohort": "measured",
                "calibration_source": "simulated_no_event", "operating_point": far_name,
                "target_far": metrics["target_far"], "observed_no_event_far": metrics["observed_no_event_far"],
                "event_recall": metrics["event_recall"],
            })
    return rows


def run_external_validation(
    *, frame: pd.DataFrame, data_path: Path, study_root: Path, recipe_path: Path, device: torch.device,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    train, validation, local_test = closed_set_group_split(frame)
    scaler = StandardScaler().fit(train[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    def transform(part: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = scaler.transform(part[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(part["Class"].to_numpy(dtype=np.int64, copy=True))
    train_x, train_y = transform(train)
    validation_x, validation_y = transform(validation)
    test_x, test_y = transform(local_test)
    recipes = load_event_recipes(recipe_path)
    external_root = study_root / "external" / "dataset"
    simulated_x, simulated_y, simulated_meta = load_simulated_external(external_root, scaler)
    measured_x, measured_y, measured_meta = load_measured_external(external_root, scaler)
    approach_results: dict[str, Any] = {}
    prediction_payload: dict[str, np.ndarray] = {
        "simulated_labels": simulated_y.astype(np.int8),
        "measured_labels": measured_y.astype(np.int8),
    }
    for approach, config_class in (("ec", ECConfig), ("pc2", PC2Config)):
        frozen = json.loads((study_root / "configs" / f"{approach}_frozen.json").read_text(encoding="utf-8"))["config"]
        config = config_class(**{**frozen, "seed": 42})
        if approach == "ec":
            model, training = train_ec_czsl(
                train_x, train_y, recipes["means"], recipes["stds"], device=device, config=config
            )
        else:
            model, training = train_pc2_oe(
                train_x, train_y, recipes["means"], recipes["stds"],
                snr_mean=float(scaler.mean_[0]), snr_scale=float(scaler.scale_[0]),
                device=device, config=config,
            )
        torch.save({
            "config": frozen,
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "purpose": "frozen-finalist external transfer evaluation",
        }, study_root / "checkpoints" / f"external_{approach}.pt")
        local_output = infer_event_model(model, test_x, recipes["means"], recipes["stds"], device=device)
        validation_output = infer_event_model(model, validation_x, recipes["means"], recipes["stds"], device=device)
        simulated_output = infer_event_model(
            model, torch.from_numpy(simulated_x), recipes["means"], recipes["stds"], device=device
        )
        measured_output = infer_event_model(
            model, torch.from_numpy(measured_x), recipes["means"], recipes["stds"], device=device
        )
        validation_normal = validation_y.numpy() == 0
        normalizer = ScoreNormalizer.fit(
            validation_output["novelty_components"].numpy()[validation_normal], config.fusion_weights
        )
        validation_score = normalizer.transform(validation_output["novelty_components"].numpy())
        simulated_score = normalizer.transform(simulated_output["novelty_components"].numpy())
        measured_score = normalizer.transform(measured_output["novelty_components"].numpy())
        local_calibrator = NormalOnlyCalibrator(config.calibration).fit(
            validation_score[validation_normal], validation_x[:, 0].numpy()[validation_normal]
        )
        simulated_normal = simulated_y == 0
        synthetic_calibrator = NormalOnlyCalibrator(config.calibration).fit(
            simulated_score[simulated_normal], simulated_x[:, 0][simulated_normal]
        )
        approach_results[approach] = {
            "local_all_class_balanced_accuracy": float(balanced_accuracy_score(
                test_y.numpy(), local_output["logits"].argmax(1).numpy()
            )),
            "simulated": _evaluate_external(simulated_output, simulated_y, simulated_meta, simulated_score),
            "measured": _evaluate_external(measured_output, measured_y, measured_meta, measured_score),
            "local_normal_calibration_transfer": {
                "simulated": _calibration_transfer(local_calibrator, simulated_score, simulated_x[:, 0], simulated_y),
                "measured": _calibration_transfer(local_calibrator, measured_score, measured_x[:, 0], measured_y),
            },
            "synthetic_no_event_to_measured_calibration_transfer": _calibration_transfer(
                synthetic_calibrator, measured_score, measured_x[:, 0], measured_y
            ),
            "training": training,
        }
        prediction_payload.update({
            f"{approach}_simulated_logits": simulated_output["logits"].numpy().astype(np.float16),
            f"{approach}_simulated_novelty_score": simulated_score.astype(np.float32),
            f"{approach}_simulated_centers": simulated_output["center"].numpy().astype(np.float32),
            f"{approach}_measured_logits": measured_output["logits"].numpy().astype(np.float16),
            f"{approach}_measured_novelty_score": measured_score.astype(np.float32),
            f"{approach}_measured_centers": measured_output["center"].numpy().astype(np.float32),
        })
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "selection_use": "none; executed only after EC and PC2 finalist freeze",
        "approaches": approach_results,
        "dataset_sha256": file_sha256(data_path),
        "external_manifest_sha256": file_sha256(study_root / "external" / "manifest.json"),
        "limitations": [
            "External labels are not mapped into local classes 1..7.",
            "Window normalization is frozen and differs from the source dataset's full-trace sequence task.",
            "Numeric TSV label interpretation 1=reflection and 2=attenuation is explicit and reported, not silently assumed.",
        ],
    }
    atomic_json(study_root / "tables" / "external_validation.json", result)
    pd.DataFrame(_external_summary_rows(result)).to_csv(
        study_root / "tables" / "external_validation.csv", index=False
    )
    atomic_json(study_root / "predictions" / "external_predictions_index.json",
                {"simulated": simulated_meta, "measured": measured_meta})
    np.savez_compressed(study_root / "predictions" / "external_predictions.npz", **prediction_payload)
    return result
