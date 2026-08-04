from __future__ import annotations

"""Resumable group-aware calibration and multi-FAR score enrichment."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .lifecycle_analysis import discover_runs
from .lifecycle_data import (
    deterministic_support_indices,
    fit_lifecycle_fold,
    split_known_calibration,
    transform_lifecycle,
)
from .lifecycle_enrollment import EnrollmentSession, fit_distance_temperature
from .lifecycle_metrics import (
    expected_calibration_error,
    gate_diagnostics,
    open_world_ranking_metrics,
)
from .lifecycle_physics import event_grammar_residual
from .lifecycle_scod import (
    DistanceReference,
    EmpiricalCDFNormalizer,
    PrototypeBank,
    assemble_components,
    evaluate_grouped_operating_point,
    evaluate_joint_operating_point,
    fit_joint_threshold,
    fit_joint_threshold_grouped,
    fuse_scores,
)
from .lifecycle_training import infer_lifecycle_model
from .model_functions.event_openworld import load_event_recipes
from .model_functions.lifecycle import FeatureAssistedOTDR, LifecycleModelConfig
from .model_functions.zero_shot import require_cuda
from .study_state import (
    StudyState,
    atomic_json,
    environment_metadata,
    file_sha256,
    validate_run,
    write_manifest,
)

ENRICHMENT_VERSION = 2


def _saved_scores(run_dir: Path) -> dict[str, Any] | None:
    with np.load(run_dir / "predictions.npz") as payload:
        required = {
            "kpsc_calibration_score",
            "calibration_labels",
            "calibration_group_ids",
            "kpsc_score",
            "labels",
            "predicted",
            "group_ids",
        }
        if not required.issubset(payload.files):
            return None
        result = {
            "calibration_score": payload["kpsc_calibration_score"].astype(float),
            "calibration_labels": payload["calibration_labels"].astype(int),
            "calibration_group_ids": payload["calibration_group_ids"].astype(str),
            "outer_score": payload["kpsc_score"].astype(float),
            "outer_labels": payload["labels"].astype(int),
            "outer_predicted": payload["predicted"].astype(int),
            "outer_group_ids": payload["group_ids"].astype(str),
            "source": "persisted_exact_scores",
        }
        cfe_required = {
            "train_embedding",
            "train_labels",
            "reference_embedding",
            "reference_competence",
            "reference_labels",
            "reference_group_ids",
            "calibration_embedding",
            "outer_embedding",
        }
        if cfe_required.issubset(payload.files):
            result.update({
                name: payload[name].copy()
                for name in cfe_required
            })
        if "outer_gate" in payload.files:
            result["outer_gate"] = payload["outer_gate"].copy()
        return result


def _selection_manifest(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    fold=None,
) -> dict[str, Any]:
    fold = fold or fit_lifecycle_fold(
        frame,
        holdout=tuple(config["holdout"]),
        seed=int(config["seed"]),
        regime=config["regime"],
    )
    selector, calibration = split_known_calibration(
        fold.split.validation, seed=int(config["seed"])
    )

    def record(part: pd.DataFrame) -> dict[str, Any]:
        groups = sorted(part["_input_group"].astype(str).unique())
        return {
            "rows": len(part),
            "unique_groups": len(groups),
            "group_list_sha256": hashlib.sha256(
                "\n".join(groups).encode()
            ).hexdigest(),
        }

    support = []
    for shots in config["cfe"]["shots"]:
        for draw in range(int(config["cfe"]["draws"])):
            indices = deterministic_support_indices(
                fold.split.reference_pool,
                class_ids=tuple(config["holdout"]),
                shots=int(shots),
                seed=int(config["seed"]),
                draw=draw,
            )
            labels = fold.split.reference_pool.loc[
                indices, "Class"
            ].to_numpy(dtype=int)
            support.append({
                "shots": int(shots),
                "draw": draw,
                "classes": {
                    str(class_id): fold.split.reference_pool.loc[
                        indices[labels == class_id], "_input_group"
                    ].astype(str).tolist()
                    for class_id in config["holdout"]
                },
                "query_used": False,
            })
    return {
        "schema_version": 1,
        "selector": record(selector),
        "threshold_calibration": record(calibration),
        "support": support,
        "adaptation": {
            "used": False,
            "partition": "adaptation_pool",
            **record(fold.split.adaptation_pool),
        },
        "query": {
            "used_for_fitting_or_selection": False,
            **record(fold.split.query),
        },
    }


def _reconstruct_scores(
    run_dir: Path,
    *,
    frame: pd.DataFrame,
    device: torch.device,
    tensor_fold=None,
) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tensor_fold = tensor_fold or fit_lifecycle_fold(
        frame,
        holdout=tuple(config["holdout"]),
        seed=int(config["seed"]),
        regime=config["regime"],
    )
    selector_frame, calibration_frame = split_known_calibration(
        tensor_fold.split.validation, seed=int(config["seed"])
    )
    checkpoint = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = FeatureAssistedOTDR(
        LifecycleModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    frames = {
        "train": tensor_fold.split.train,
        "selector": selector_frame,
        "calibration": calibration_frame,
        "reference_pool": tensor_fold.split.reference_pool,
        "outer": pd.concat(
            (tensor_fold.split.seen_test, tensor_fold.split.query),
            ignore_index=True,
        ),
    }
    batches = {
        name: transform_lifecycle(part, tensor_fold.scaler)
        for name, part in frames.items()
    }
    outputs = {}
    inference_rows = {}
    for name, batch in batches.items():
        torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        outputs[name] = infer_lifecycle_model(model, batch, device=device)
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - inference_started
        inference_rows[name] = {
            "seconds": duration,
            "examples": len(batch),
            "milliseconds_per_trace_including_transfer": (
                duration * 1000 / max(len(batch), 1)
            ),
        }
    train_embedding = outputs["train"]["embedding"].numpy()
    train_labels = batches["train"].labels.numpy()
    scod = config["scod"]
    reference = DistanceReference.fit(train_embedding, train_labels)
    bank = PrototypeBank.fit(
        train_embedding,
        train_labels,
        prototypes_per_class=int(scod["prototypes_per_class"]),
        metric=scod["prototype_metric"],
        seed=int(config["seed"]),
    )
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "otdr_event_openworld_study"
        / "configs"
        / "event_recipes.json"
    )
    recipes = load_event_recipes(recipe_path)
    known_ids = tuple(sorted(set(range(8)) - set(config["holdout"])))

    def components(name: str) -> tuple[tuple[str, ...], np.ndarray]:
        residual = event_grammar_residual(
            batches[name].trace,
            batches[name].context,
            recipes["means"],
            recipes["stds"],
            known_class_ids=known_ids,
        )
        return assemble_components(
            logits=outputs[name]["logits"].numpy(),
            embeddings=outputs[name]["embedding"].numpy(),
            distance_reference=reference,
            prototype_bank=bank,
            physics_residual=residual,
            knn_k=int(scod["knn_k"]),
        )

    names, selector_components = components("selector")
    normalizer = EmpiricalCDFNormalizer.fit(selector_components, names)
    _, calibration_components = components("calibration")
    _, outer_components = components("outer")
    calibration_score = fuse_scores(
        normalizer.transform(calibration_components),
        method=scod["fusion"],
        weights=scod.get("fusion_weights"),
    )
    outer_score = fuse_scores(
        normalizer.transform(outer_components),
        method=scod["fusion"],
        weights=scod.get("fusion_weights"),
    )
    return {
        "calibration_score": calibration_score,
        "calibration_labels": batches["calibration"].labels.numpy(),
        "calibration_group_ids": np.asarray(batches["calibration"].group_ids),
        "outer_score": outer_score,
        "outer_labels": batches["outer"].labels.numpy(),
        "outer_predicted": outputs["outer"]["logits"].argmax(1).numpy(),
        "outer_group_ids": np.asarray(batches["outer"].group_ids),
        "source": "cuda_checkpoint_reconstruction",
        "component_names": list(names),
        "train_embedding": train_embedding,
        "train_labels": train_labels,
        "reference_embedding": outputs["reference_pool"]["embedding"].numpy(),
        "reference_competence": torch.sigmoid(
            outputs["reference_pool"]["competence"]
        ).numpy(),
        "reference_labels": batches["reference_pool"].labels.numpy(),
        "reference_group_ids": np.asarray(batches["reference_pool"].group_ids),
        "calibration_embedding": outputs["calibration"]["embedding"].numpy(),
        "outer_embedding": outputs["outer"]["embedding"].numpy(),
        "outer_gate": outputs["outer"]["gate"].numpy(),
        "inference_benchmark": inference_rows,
    }


def _support_positions(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    holdout: tuple[int, int],
    shots: int,
    seed: int,
    draw: int,
) -> np.ndarray:
    chosen = []
    for class_id in holdout:
        candidates = np.flatnonzero(labels == class_id)
        first_by_group: dict[str, int] = {}
        for index in candidates:
            first_by_group.setdefault(str(groups[index]), int(index))
        ranked = sorted(
            first_by_group.values(),
            key=lambda index: hashlib.sha256(
                f"lifecycle-support:{seed}:{draw}:{shots}:{class_id}:{groups[index]}".encode()
            ).hexdigest(),
        )
        if len(ranked) < shots:
            raise ValueError(f"Class {class_id} lacks {shots} support groups.")
        chosen.extend(ranked[:shots])
    return np.asarray(chosen, dtype=int)


def _cfe_calibration(
    scores: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    required = {
        "train_embedding",
        "train_labels",
        "reference_embedding",
        "reference_competence",
        "reference_labels",
        "reference_group_ids",
        "calibration_embedding",
        "outer_embedding",
    }
    if not required.issubset(scores):
        return None
    cfe = config["cfe"]
    holdout = tuple(config["holdout"])
    train_embedding = np.asarray(scores["train_embedding"], dtype=float)
    train_labels = np.asarray(scores["train_labels"], dtype=int)
    reference_embedding = np.asarray(scores["reference_embedding"], dtype=float)
    reference_competence = np.asarray(scores["reference_competence"], dtype=float)
    reference_labels = np.asarray(scores["reference_labels"], dtype=int)
    reference_groups = np.asarray(scores["reference_group_ids"]).astype(str)
    outer_embedding = np.asarray(scores["outer_embedding"], dtype=float)
    outer_labels = np.asarray(scores["outer_labels"], dtype=int)
    base = EnrollmentSession.from_base(
        train_embedding, train_labels, metric=cfe["metric"]
    )
    temperature = fit_distance_temperature(
        base,
        np.asarray(scores["calibration_embedding"], dtype=float),
        np.asarray(scores["calibration_labels"], dtype=int),
    )
    rows = []
    for shots in cfe["shots"]:
        for draw in range(int(cfe["draws"])):
            positions = _support_positions(
                reference_labels,
                reference_groups,
                holdout=holdout,
                shots=int(shots),
                seed=int(config["seed"]),
                draw=draw,
            )
            session = base
            enrollment_started = time.perf_counter()
            for class_id in holdout:
                mask = reference_labels[positions] == class_id
                class_positions = positions[mask]
                session = session.enroll(
                    class_id,
                    reference_embedding[class_positions],
                    method=cfe["prototype_method"],
                    quality=reference_competence[class_positions]
                    if cfe["prototype_method"] == "quality_weighted"
                    else None,
                    teen_alpha=float(cfe["teen_alpha"]),
                    teen_temperature=float(cfe["teen_temperature"]),
                    support_group_ids=tuple(reference_groups[class_positions]),
                )
            enrollment_latency_ms = (
                time.perf_counter() - enrollment_started
            ) * 1000
            probability = session.predict_proba(
                outer_embedding, temperature=temperature
            )
            prediction = probability.argmax(1)
            one_hot = np.eye(8)[outer_labels]
            rows.append({
                "shots": int(shots),
                "draw": draw,
                "nll": float(
                    -np.log(
                        np.clip(
                            probability[np.arange(len(outer_labels)), outer_labels],
                            1e-12,
                            1,
                        )
                    ).mean()
                ),
                "brier": float(np.square(probability - one_hot).sum(1).mean()),
                "ece_15": expected_calibration_error(
                    probability, outer_labels
                ),
                "accuracy": float((prediction == outer_labels).mean()),
                "normal_far_after_enrollment": float(
                    np.isin(
                        prediction[outer_labels == 0],
                        holdout,
                    ).mean()
                ),
                "enrollment_latency_ms": enrollment_latency_ms,
                "prototype_storage_bytes": session.storage_bytes,
            })
    return {
        "distance_temperature": temperature,
        "temperature_fit_partition": "known group-disjoint calibration only",
        "rows": rows,
    }


def enrich_run(
    run_dir: Path,
    *,
    frame: pd.DataFrame,
    study_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    source_checkpoint_hash = file_sha256(run_dir / "checkpoint.pt")
    output_dir = (
        study_root
        / "posthoc_calibration"
        / config["regime"]
        / run_dir.name
    )
    valid, _ = validate_run(
        output_dir,
        expected={
            "run_id": f"posthoc-{run_dir.name}",
            "source_checkpoint_sha256": source_checkpoint_hash,
            "enrichment_version": ENRICHMENT_VERSION,
        },
    )
    if valid:
        return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    state = StudyState(study_root)
    run_id = f"posthoc-{run_dir.name}"
    with state.run(
        run_id,
        output_dir,
        {
            "stage": "posthoc_group_calibration",
            "source_run_id": run_dir.name,
            "regime": config["regime"],
        },
    ):
        started = time.perf_counter()
        source_selection_path = run_dir / "selection_manifest.json"
        protocol_fold = None
        if not source_selection_path.exists():
            protocol_fold = fit_lifecycle_fold(
                frame,
                holdout=tuple(config["holdout"]),
                seed=int(config["seed"]),
                regime=config["regime"],
            )
        scores = _saved_scores(run_dir)
        if scores is None:
            scores = _reconstruct_scores(
                run_dir,
                frame=frame,
                device=device,
                tensor_fold=protocol_fold,
            )
        scod = config["scod"]
        holdout = tuple(config["holdout"])
        operating_points: dict[str, Any] = {}
        grouped_operating_points: dict[str, Any] = {}
        for far in (0.01, 0.02, 0.05):
            row_threshold = fit_joint_threshold(
                scores["calibration_score"],
                scores["calibration_labels"],
                normal_far_cap=far,
                known_acceptance_floor=float(scod["known_acceptance_floor"]),
                mode=scod["calibration_mode"],
            )
            group_threshold = fit_joint_threshold_grouped(
                scores["calibration_score"],
                scores["calibration_labels"],
                scores["calibration_group_ids"],
                normal_far_cap=far,
                known_acceptance_floor=float(scod["known_acceptance_floor"]),
                mode=scod["calibration_mode"],
            )
            key = f"far_{far:.3f}"
            operating_points[key] = {
                "calibration": asdict(row_threshold),
                **evaluate_joint_operating_point(
                    scores["outer_score"],
                    scores["outer_labels"],
                    scores["outer_predicted"],
                    holdout=holdout,
                    calibration=row_threshold,
                ),
            }
            grouped_operating_points[key] = {
                "calibration": asdict(group_threshold),
                **evaluate_grouped_operating_point(
                    scores["outer_score"],
                    scores["outer_labels"],
                    scores["outer_predicted"],
                    scores["outer_group_ids"],
                    holdout=holdout,
                    calibration=group_threshold,
                    seed=int(config["seed"]) + 100 * holdout[0] + holdout[1],
                ),
            }
        metrics = {
            "schema_version": 1,
            "enrichment_version": ENRICHMENT_VERSION,
            "source_run_id": run_dir.name,
            "holdout": list(holdout),
            "seed": config["seed"],
            "regime": config["regime"],
            "score_source": scores["source"],
            "source_checkpoint_sha256": source_checkpoint_hash,
            "operating_points": operating_points,
            "group_equal_weight_operating_points": grouped_operating_points,
            "ranking": open_world_ranking_metrics(
                scores["outer_score"],
                scores["outer_labels"],
                scores["outer_predicted"],
                holdout=holdout,
            ),
            "fusion_gate": (
                gate_diagnostics(
                    scores["outer_gate"], scores["outer_labels"]
                )
                if "outer_gate" in scores
                else None
            ),
            "cfe_probability_calibration": _cfe_calibration(scores, config),
            "inference_benchmark": scores.get("inference_benchmark"),
            "checkpoint_size_bytes": (run_dir / "checkpoint.pt").stat().st_size,
            "duration_seconds": time.perf_counter() - started,
            "environment": environment_metadata(device),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        selection_manifest = (
            json.loads(source_selection_path.read_text(encoding="utf-8"))
            if source_selection_path.exists()
            else _selection_manifest(frame, config, fold=protocol_fold)
        )
        atomic_json(
            output_dir / "source_selection_manifest.json",
            selection_manifest,
        )
        np.savez_compressed(
            output_dir / "scores.npz",
            calibration_score=np.asarray(scores["calibration_score"], dtype=np.float64),
            calibration_labels=np.asarray(scores["calibration_labels"], dtype=np.int8),
            calibration_group_ids=np.asarray(scores["calibration_group_ids"]),
            outer_score=np.asarray(scores["outer_score"], dtype=np.float64),
            outer_labels=np.asarray(scores["outer_labels"], dtype=np.int8),
            outer_predicted=np.asarray(scores["outer_predicted"], dtype=np.int8),
            outer_group_ids=np.asarray(scores["outer_group_ids"]),
        )
        atomic_json(output_dir / "metrics.json", metrics)
        write_manifest(output_dir, {
            "run_id": run_id,
            "completed": True,
            "device": str(device),
            "source_checkpoint_sha256": source_checkpoint_hash,
            "source_run_id": run_dir.name,
            "group_equal_weight": True,
            "enrichment_version": ENRICHMENT_VERSION,
        })
    return metrics


def run_calibration_enrichment(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    regime: str,
    device: torch.device | str,
    require_complete: bool = True,
    expected_runs: int | None = None,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    runs = discover_runs(study_root, regime=regime)
    expected = 105 if expected_runs is None else expected_runs
    if require_complete and len(runs) != expected:
        raise RuntimeError(
            f"{regime!r} enrichment requires {expected} source runs; found {len(runs)}."
        )
    results = [
        enrich_run(
            run_dir,
            frame=frame,
            study_root=study_root,
            device=device,
        )
        for run_dir in runs
    ]
    summary = {
        "schema_version": 1,
        "regime": regime,
        "source_runs": len(runs),
        "expected_runs": expected,
        "complete": len(runs) == expected,
        "cuda_reconstructed_runs": sum(
            row["score_source"] == "cuda_checkpoint_reconstruction"
            for row in results
        ),
        "persisted_score_runs": sum(
            row["score_source"] == "persisted_exact_scores"
            for row in results
        ),
    }
    atomic_json(
        study_root / "posthoc_calibration" / regime / "summary.json",
        summary,
    )
    return summary
