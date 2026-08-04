from __future__ import annotations

"""Resumable fold experiments for the feature-assisted OTDR lifecycle study."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .event_openworld_data import write_exact_group_manifest
from .lifecycle_data import (
    FeatureRegime,
    deterministic_support_indices,
    fit_lifecycle_fold,
    lifecycle_split_manifest,
    split_known_calibration,
    transform_lifecycle,
)
from .lifecycle_baselines import NearestNeighborReference
from .lifecycle_enrollment import EnrollmentSession
from .lifecycle_enrollment import sequential_orders
from .lifecycle_physics import event_grammar_residual
from .lifecycle_metrics import (
    classification_metrics,
    gate_diagnostics,
    hard_prediction_metrics,
    open_world_ranking_metrics,
)
from .lifecycle_scod import (
    DistanceReference,
    EmpiricalCDFNormalizer,
    PrototypeBank,
    assemble_components,
    evaluate_joint_operating_point,
    fit_joint_threshold,
    fuse_scores,
)
from .lifecycle_training import (
    LifecycleTrainingConfig,
    infer_lifecycle_model,
    train_lifecycle_model,
)
from .model_functions.lifecycle import LifecycleModelConfig
from .model_functions.event_openworld import load_event_recipes
from .model_functions.zero_shot import require_cuda
from .study_state import (
    StudyState,
    atomic_json,
    config_hash,
    environment_metadata,
    file_sha256,
    stable_run_id,
    validate_run,
    write_manifest,
)


@dataclass(frozen=True)
class SCODConfig:
    prototypes_per_class: int = 4
    prototype_metric: str = "cosine"
    knn_k: int = 10
    fusion: str = "robust_regret"
    fusion_weights: tuple[float, ...] | None = None
    calibration_mode: str = "empirical"
    normal_far_cap: float = 0.0125
    known_acceptance_floor: float = 0.95


@dataclass(frozen=True)
class CFEConfig:
    prototype_method: str = "mean"
    metric: str = "cosine"
    teen_alpha: float = 0.2
    teen_temperature: float = 0.5
    shots: tuple[int, ...] = (1, 3, 5)
    draws: int = 20


@dataclass(frozen=True)
class FoldExperimentConfig:
    holdout: tuple[int, int]
    seed: int
    regime: FeatureRegime = "full"
    model: LifecycleModelConfig = LifecycleModelConfig()
    training: LifecycleTrainingConfig = LifecycleTrainingConfig()
    scod: SCODConfig = SCODConfig()
    cfe: CFEConfig = CFEConfig()
    device: str = "cuda:0"
    stage: str = "pilot"


def _git_metadata(root: Path) -> dict[str, object]:
    def command(*args: str) -> bytes:
        try:
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return b""

    revision = command("rev-parse", "HEAD").decode().strip() or None
    patch = command("diff", "--binary", "HEAD")
    untracked = command("ls-files", "--others", "--exclude-standard")
    source_digest = hashlib.sha256(patch + b"\0")
    hashed_untracked = []
    for relative in untracked.decode(errors="replace").splitlines():
        normalized = relative.replace("\\", "/")
        suffix = Path(relative).suffix.lower()
        in_declared_scope = (
            "OTDR_CLI/OTDR/src/" in normalized
            or "OTDR_CLI/OTDR/tests/" in normalized
            or "otdr_feature_assisted_lifecycle_study/configs/" in normalized
        )
        generated_subtree = any(
            segment in normalized
            for segment in (
                "OTDR_CLI/OTDR/src/wandb/",
                "OTDR_CLI/OTDR/src/outputs/",
                "OTDR_CLI/OTDR/src/models/",
                "__pycache__/",
            )
        )
        if (
            not in_declared_scope
            or generated_subtree
            or suffix not in {".py", ".json", ".md", ".toml", ".txt"}
        ):
            continue
        path = root / relative
        try:
            is_small_file = (
                path.is_file() and path.stat().st_size <= 5 * 1024 * 1024
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not hash in-scope provenance file {relative!r}."
            ) from exc
        if is_small_file:
            source_digest.update(relative.encode())
            source_digest.update(b"\0")
            try:
                source_digest.update(path.read_bytes())
            except OSError as exc:
                raise RuntimeError(
                    f"Could not hash in-scope provenance file {relative!r}."
                ) from exc
            source_digest.update(b"\0")
            hashed_untracked.append(relative)
    return {
        "revision": revision,
        "dirty": bool(patch or untracked),
        "dirty_patch_sha256": hashlib.sha256(patch + b"\0" + untracked).hexdigest(),
        "dirty_source_content_sha256": source_digest.hexdigest(),
        "content_hashed_untracked_files": len(hashed_untracked),
    }


def _frame_batch(frame: pd.DataFrame, tensor_fold) -> Any:
    return transform_lifecycle(frame, tensor_fold.scaler)


def _components(
    output: dict[str, torch.Tensor],
    *,
    batch,
    reference: DistanceReference,
    prototypes: PrototypeBank,
    config: SCODConfig,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    known_class_ids: tuple[int, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    residual = event_grammar_residual(
        batch.trace, batch.context, recipe_means, recipe_stds,
        known_class_ids=known_class_ids,
    )
    return assemble_components(
        logits=output["logits"].numpy(),
        embeddings=output["embedding"].numpy(),
        distance_reference=reference,
        prototype_bank=prototypes,
        physics_residual=residual,
        knn_k=config.knn_k,
    )


def _group_record(frame: pd.DataFrame) -> dict[str, object]:
    groups = sorted(frame["_input_group"].astype(str).unique())
    return {
        "rows": len(frame),
        "unique_groups": len(groups),
        "group_list_sha256": hashlib.sha256(
            "\n".join(groups).encode()
        ).hexdigest(),
    }


def _evaluate_cfe(
    *,
    tensor_fold,
    outputs: dict[str, dict[str, torch.Tensor]],
    config: CFEConfig,
) -> tuple[
    list[dict[str, object]],
    dict[str, np.ndarray],
    list[dict[str, object]],
]:
    train_embedding = outputs["train"]["embedding"].numpy()
    train_labels = tensor_fold.batches["train"].labels.numpy()
    base_ids = tuple(sorted(int(value) for value in np.unique(train_labels)))
    reference_embedding = outputs["reference_pool"]["embedding"].numpy()
    reference_competence = torch.sigmoid(outputs["reference_pool"]["competence"]).numpy()
    query_embedding = np.vstack((outputs["seen_test"]["embedding"].numpy(), outputs["query"]["embedding"].numpy()))
    query_labels = np.r_[
        tensor_fold.batches["seen_test"].labels.numpy(),
        tensor_fold.batches["query"].labels.numpy(),
    ]
    train_raw = np.c_[
        tensor_fold.batches["train"].trace.numpy(),
        tensor_fold.batches["train"].context.numpy(),
        tensor_fold.batches["train"].context_missing.numpy(),
    ]
    reference_raw = np.c_[
        tensor_fold.batches["reference_pool"].trace.numpy(),
        tensor_fold.batches["reference_pool"].context.numpy(),
        tensor_fold.batches["reference_pool"].context_missing.numpy(),
    ]
    query_raw = np.c_[
        np.vstack((
            tensor_fold.batches["seen_test"].trace.numpy(),
            tensor_fold.batches["query"].trace.numpy(),
        )),
        np.vstack((
            tensor_fold.batches["seen_test"].context.numpy(),
            tensor_fold.batches["query"].context.numpy(),
        )),
        np.vstack((
            tensor_fold.batches["seen_test"].context_missing.numpy(),
            tensor_fold.batches["query"].context_missing.numpy(),
        )),
    ]
    reference_frame = tensor_fold.split.reference_pool
    rows: list[dict[str, object]] = []
    support_manifest: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {"cfe_labels": query_labels.astype(np.int8)}
    nn_references = {
        "raw_cosine_1nn": (NearestNeighborReference.fit(train_raw, train_labels, metric="cosine"), query_raw),
        "raw_euclidean_1nn": (NearestNeighborReference.fit(train_raw, train_labels, metric="euclidean"), query_raw),
        "raw_mahalanobis_1nn": (
            NearestNeighborReference.fit(train_raw, train_labels, metric="diagonal_mahalanobis"), query_raw
        ),
        "encoder_cosine_1nn": (
            NearestNeighborReference.fit(train_embedding, train_labels, metric="cosine"), query_embedding
        ),
    }
    nn_base = {
        name: reference.nearest(query)
        for name, (reference, query) in nn_references.items()
    }

    def metric_row(prediction: np.ndarray, *, method: str, shots: int, draw: int, **extra: object) -> dict[str, object]:
        metrics = hard_prediction_metrics(
            query_labels,
            prediction,
            base_class_ids=base_ids,
            enrolled_class_ids=tensor_fold.split.holdout,
        )
        normal = query_labels == 0
        metrics["normal_far_after_enrollment"] = float(
            np.isin(prediction[normal], tensor_fold.split.holdout).mean()
        )
        return {"shots": shots, "draw": draw, "method": method, **extra, **metrics}

    for shots in config.shots:
        for draw in range(config.draws):
            selected_indices = deterministic_support_indices(
                reference_frame,
                class_ids=tensor_fold.split.holdout,
                shots=shots,
                seed=tensor_fold.split.seed,
                draw=draw,
            )
            positions = reference_frame.index.get_indexer(selected_indices)
            if np.any(positions < 0):
                raise AssertionError("Support indices did not map to the frozen reference embeddings.")
            selected_labels = reference_frame.loc[selected_indices, "Class"].to_numpy(dtype=int)
            support_manifest.append({
                "shots": int(shots),
                "draw": int(draw),
                "classes": {
                    str(class_id): reference_frame.loc[
                        selected_indices[selected_labels == class_id],
                        "_input_group",
                    ].astype(str).tolist()
                    for class_id in tensor_fold.split.holdout
                },
                "selection_inputs": (
                    "seed, draw, shot count, class ID, and exact group ID only"
                ),
                "query_used": False,
            })
            for baseline_name, (reference, query) in nn_references.items():
                support_values = reference_raw[positions] if baseline_name.startswith("raw_") else reference_embedding[positions]
                prediction = reference.combine_support(
                    query,
                    *nn_base[baseline_name],
                    support_values,
                    selected_labels,
                )
                rows.append(metric_row(
                    prediction, method=baseline_name, shots=shots, draw=draw,
                    metric=reference.metric, training_free=True,
                ))

            prototype_variants = [
                ("uncalibrated_mean", "mean", "cosine", 0.0),
                ("mean_euclidean", "mean", "euclidean", 0.0),
                ("mean_mahalanobis", "mean", "diagonal_mahalanobis", 0.0),
                ("medoid_cosine", "medoid", "cosine", 0.0),
                ("finalist", config.prototype_method, config.metric, config.teen_alpha),
            ]
            if shots > 1:
                prototype_variants.extend([
                    ("median_cosine", "median", "cosine", 0.0),
                    ("quality_weighted_cosine", "quality_weighted", "cosine", 0.0),
                ])
            for variant_name, prototype_method, metric, teen_alpha in prototype_variants:
                session = EnrollmentSession.from_base(train_embedding, train_labels, metric=metric)
                base_prediction = session.predict(query_embedding)
                enrollment_started = time.perf_counter()
                for class_id in tensor_fold.split.holdout:
                    class_mask = selected_labels == class_id
                    class_positions = positions[class_mask]
                    support_groups = tuple(reference_frame.loc[selected_indices[class_mask], "_input_group"].astype(str))
                    session = session.enroll(
                        class_id,
                        reference_embedding[class_positions],
                        method=prototype_method,
                        quality=reference_competence[class_positions] if prototype_method == "quality_weighted" else None,
                        teen_alpha=teen_alpha,
                        teen_temperature=config.teen_temperature,
                        support_group_ids=support_groups,
                    )
                enrollment_latency_ms = (
                    time.perf_counter() - enrollment_started
                ) * 1000
                prediction = session.predict(query_embedding)
                row = metric_row(
                    prediction, method=variant_name, shots=shots, draw=draw,
                    prototype_method=prototype_method, metric=metric,
                    teen_alpha=teen_alpha, storage_bytes=session.storage_bytes,
                    training_free=True,
                    enrollment_latency_ms=enrollment_latency_ms,
                )
                base_mask = np.isin(query_labels, base_ids)
                before_accuracy = float((base_prediction[base_mask] == query_labels[base_mask]).mean())
                row["base_accuracy_before"] = before_accuracy
                row["forgetting"] = max(0.0, before_accuracy - float(row["base_accuracy"]))
                row["backward_transfer"] = float(row["base_accuracy"]) - before_accuracy
                row["retention_ratio"] = float(row["base_accuracy"]) / before_accuracy if before_accuracy else None
                rows.append(row)
                if variant_name == "finalist":
                    prediction_payload[f"cfe_prediction_shot{shots}_draw{draw}"] = prediction.astype(np.int8)

            # Required sequential evaluation in both class orders for the finalist.
            for order in sequential_orders(tensor_fold.split.holdout):
                session = EnrollmentSession.from_base(train_embedding, train_labels, metric=config.metric)
                for session_index, class_id in enumerate(order, start=1):
                    class_mask = selected_labels == class_id
                    class_positions = positions[class_mask]
                    session = session.enroll(
                        class_id,
                        reference_embedding[class_positions],
                        method=config.prototype_method,
                        quality=reference_competence[class_positions] if config.prototype_method == "quality_weighted" else None,
                        teen_alpha=config.teen_alpha,
                        teen_temperature=config.teen_temperature,
                        support_group_ids=tuple(reference_frame.loc[selected_indices[class_mask], "_input_group"].astype(str)),
                    )
                    prediction = session.predict(query_embedding)
                    enrolled_mask = np.isin(query_labels, order[:session_index])
                    base_mask = np.isin(query_labels, base_ids)
                    retained = float((prediction[base_mask] == query_labels[base_mask]).mean())
                    enrolled_accuracy = float((prediction[enrolled_mask] == query_labels[enrolled_mask]).mean())
                    harmonic = 0.0 if retained + enrolled_accuracy == 0 else 2 * retained * enrolled_accuracy / (
                        retained + enrolled_accuracy
                    )
                    rows.append({
                        "shots": shots, "draw": draw, "method": "finalist_sequential",
                        "order": list(order), "session": session_index,
                        "classes_enrolled": list(order[:session_index]),
                        "base_accuracy": retained,
                        "enrolled_accuracy": enrolled_accuracy,
                        "harmonic_mean": harmonic,
                        "normal_far_after_enrollment": float(
                            np.isin(prediction[query_labels == 0], order[:session_index]).mean()
                        ),
                        "storage_bytes": session.storage_bytes,
                        "training_free": True,
                    })
    return rows, prediction_payload, support_manifest


def execute_fold(
    *,
    frame: pd.DataFrame,
    data_path: Path,
    study_root: Path,
    repository_root: Path,
    config: FoldExperimentConfig,
) -> dict[str, Any]:
    device = require_cuda(config.device)
    run_id = stable_run_id("lifecycle", config.holdout, config.seed, asdict(config))
    run_dir = study_root / config.stage / run_id
    valid, reason = validate_run(run_dir, expected={"run_id": run_id})
    if valid:
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    state = StudyState(study_root)
    metadata = {
        "run_id": run_id,
        "holdout": list(config.holdout),
        "seed": config.seed,
        "regime": config.regime,
        "stage": config.stage,
        "config_hash": config_hash(asdict(config)),
        "dataset_sha256": file_sha256(data_path),
        "resume_validation": reason,
    }
    with state.run(run_id, run_dir, metadata):
        started = time.perf_counter()
        tensor_fold = fit_lifecycle_fold(
            frame, holdout=config.holdout, seed=config.seed, regime=config.regime
        )
        recipe_path = (
            Path(__file__).resolve().parents[1]
            / "experiments" / "otdr_event_openworld_study" / "configs" / "event_recipes.json"
        )
        recipes = load_event_recipes(recipe_path)
        known_class_ids = tuple(sorted(set(range(8)) - set(config.holdout)))
        selector_frame, calibration_frame = split_known_calibration(
            tensor_fold.split.validation, seed=config.seed
        )
        model, training_metadata = train_lifecycle_model(
            tensor_fold.batches["train"],
            _frame_batch(selector_frame, tensor_fold),
            device=device,
            model_config=config.model,
            training_config=config.training,
        )
        checkpoint_path = run_dir / "checkpoint.pt"
        torch.save(
            {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "model_config": asdict(model.config),
                "training_config": asdict(config.training),
                "scaler": tensor_fold.scaler.payload(),
            },
            checkpoint_path,
        )
        inference_seconds = 0.0
        inference_examples = 0

        def timed_infer(batch):
            nonlocal inference_seconds, inference_examples
            torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            output = infer_lifecycle_model(model, batch, device=device)
            torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_started
            inference_examples += len(batch)
            return output

        outputs = {
            name: timed_infer(batch)
            for name, batch in tensor_fold.batches.items()
        }
        selector_output = timed_infer(_frame_batch(selector_frame, tensor_fold))
        calibration_output = timed_infer(_frame_batch(calibration_frame, tensor_fold))

        train_embedding = outputs["train"]["embedding"].numpy()
        train_labels = tensor_fold.batches["train"].labels.numpy()
        reference = DistanceReference.fit(train_embedding, train_labels)
        prototypes = PrototypeBank.fit(
            train_embedding,
            train_labels,
            prototypes_per_class=config.scod.prototypes_per_class,
            metric=config.scod.prototype_metric,
            seed=config.seed,
        )
        component_names, selector_components = _components(
            selector_output, batch=_frame_batch(selector_frame, tensor_fold),
            reference=reference, prototypes=prototypes, config=config.scod,
            recipe_means=recipes["means"], recipe_stds=recipes["stds"],
            known_class_ids=known_class_ids,
        )
        normalizer = EmpiricalCDFNormalizer.fit(selector_components, component_names)
        _, calibration_components = _components(
            calibration_output, batch=_frame_batch(calibration_frame, tensor_fold),
            reference=reference, prototypes=prototypes, config=config.scod,
            recipe_means=recipes["means"], recipe_stds=recipes["stds"],
            known_class_ids=known_class_ids,
        )
        normalized_calibration = normalizer.transform(calibration_components)
        calibration_score = fuse_scores(
            normalized_calibration,
            method=config.scod.fusion,
            weights=config.scod.fusion_weights,
        )
        threshold = fit_joint_threshold(
            calibration_score,
            calibration_frame["Class"].to_numpy(dtype=int),
            normal_far_cap=config.scod.normal_far_cap,
            known_acceptance_floor=config.scod.known_acceptance_floor,
            mode=config.scod.calibration_mode,
        )

        outer_frame = pd.concat((tensor_fold.split.seen_test, tensor_fold.split.query), ignore_index=True)
        outer_batch = _frame_batch(outer_frame, tensor_fold)
        outer_output = timed_infer(outer_batch)
        _, outer_components = _components(
            outer_output, batch=outer_batch,
            reference=reference, prototypes=prototypes, config=config.scod,
            recipe_means=recipes["means"], recipe_stds=recipes["stds"],
            known_class_ids=known_class_ids,
        )
        normalized_outer = normalizer.transform(outer_components)
        outer_score = fuse_scores(
            normalized_outer,
            method=config.scod.fusion,
            weights=config.scod.fusion_weights,
        )
        outer_labels = outer_batch.labels.numpy()
        outer_prediction = outer_output["logits"].argmax(1).numpy()
        scod_metrics = {
            **evaluate_joint_operating_point(
                outer_score,
                outer_labels,
                outer_prediction,
                holdout=config.holdout,
                calibration=threshold,
            ),
            **open_world_ranking_metrics(
                outer_score, outer_labels, outer_prediction, holdout=config.holdout
            ),
            "component_names": list(component_names),
            "fusion": config.scod.fusion,
            "calibration": asdict(threshold),
        }
        known_metrics = classification_metrics(
            outputs["seen_test"]["logits"].numpy(),
            tensor_fold.batches["seen_test"].labels.numpy(),
            positions=tensor_fold.batches["seen_test"].position.numpy(),
            predicted_positions=outputs["seen_test"]["position"].numpy(),
        )
        cfe_rows, cfe_predictions, support_manifest = _evaluate_cfe(
            tensor_fold=tensor_fold,
            outputs=outputs,
            config=config.cfe,
        )
        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "holdout": list(config.holdout),
            "seed": config.seed,
            "regime": config.regime,
            "known_closed_set": known_metrics,
            "fusion_gate": gate_diagnostics(
                outer_output["gate"].numpy(), outer_labels
            ),
            "kpsc": scod_metrics,
            "cfe": cfe_rows,
            "training": training_metadata,
            "inference": {
                "seconds": inference_seconds,
                "examples": inference_examples,
                "milliseconds_per_trace_including_transfer": (
                    inference_seconds * 1000 / max(inference_examples, 1)
                ),
                "device": str(device),
            },
            "duration_seconds": time.perf_counter() - started,
        }
        atomic_json(run_dir / "config.json", asdict(config))
        atomic_json(run_dir / "scaler.json", tensor_fold.scaler.payload())
        atomic_json(
            run_dir / "split_manifest.json",
            lifecycle_split_manifest(
                tensor_fold.split, data_path=data_path, regime=config.regime
            ),
        )
        atomic_json(
            run_dir / "selection_manifest.json",
            {
                "schema_version": 1,
                "selector": _group_record(selector_frame),
                "threshold_calibration": _group_record(calibration_frame),
                "support": support_manifest,
                "adaptation": {
                    "used": False,
                    "partition": "adaptation_pool",
                    **_group_record(tensor_fold.split.adaptation_pool),
                },
                "query": {
                    "used_for_fitting_or_selection": False,
                    **_group_record(tensor_fold.split.query),
                },
            },
        )
        write_exact_group_manifest(tensor_fold.split, run_dir / "exact_groups.npz")
        atomic_json(run_dir / "metrics.json", metrics)
        np.savez_compressed(
            run_dir / "predictions.npz",
            labels=outer_labels.astype(np.int8),
            predicted=outer_prediction.astype(np.int8),
            position=outer_batch.position.numpy().astype(np.float32),
            predicted_position=outer_output["position"].numpy().astype(np.float32),
            kpsc_raw_components=outer_components.astype(np.float32),
            kpsc_normalized_components=normalized_outer.astype(np.float32),
            # Preserve threshold-side decisions. Empirical-CDF fusion creates
            # ties, so float32 quantization can otherwise move boundary examples.
            kpsc_score=outer_score.astype(np.float64),
            kpsc_rejected=(outer_score > threshold.threshold),
            calibration_labels=calibration_frame["Class"].to_numpy(dtype=np.int8),
            calibration_group_ids=np.asarray(_frame_batch(
                calibration_frame, tensor_fold
            ).group_ids),
            kpsc_calibration_score=calibration_score.astype(np.float64),
            kpsc_normalized_calibration_components=normalized_calibration.astype(np.float32),
            component_names=np.asarray(component_names),
            group_ids=np.asarray(outer_batch.group_ids),
            train_embedding=train_embedding.astype(np.float32),
            train_labels=train_labels.astype(np.int8),
            reference_embedding=outputs["reference_pool"]["embedding"].numpy().astype(np.float32),
            reference_competence=torch.sigmoid(
                outputs["reference_pool"]["competence"]
            ).numpy().astype(np.float32),
            reference_labels=tensor_fold.batches["reference_pool"].labels.numpy().astype(np.int8),
            reference_group_ids=np.asarray(tensor_fold.batches["reference_pool"].group_ids),
            calibration_embedding=calibration_output["embedding"].numpy().astype(np.float32),
            outer_embedding=outer_output["embedding"].numpy().astype(np.float32),
            outer_gate=outer_output["gate"].numpy().astype(np.float32),
            **cfe_predictions,
        )
        manifest = write_manifest(
            run_dir,
            {
                **metadata,
                "run_id": run_id,
                "source": _git_metadata(repository_root),
                "environment": environment_metadata(device),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "completed": True,
            },
        )
    return metrics
