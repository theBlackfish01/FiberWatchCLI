from __future__ import annotations

"""Nested, group-safe staged selection for the three lifecycle families."""

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .event_openworld_data import attach_input_groups, build_event_openworld_fold
from .event_openworld_sweep import _inner_task
from .lifecycle_data import fit_lifecycle_scaler, split_known_calibration, transform_lifecycle
from .lifecycle_enrollment import EnrollmentSession
from .lifecycle_metrics import hard_prediction_metrics
from .lifecycle_scod import (
    DistanceReference,
    EmpiricalCDFNormalizer,
    PrototypeBank,
    assemble_components,
    evaluate_joint_operating_point,
    fit_joint_threshold,
    fuse_scores,
)
from .lifecycle_training import LifecycleTrainingConfig, infer_lifecycle_model, train_lifecycle_model
from .model_functions.lifecycle import FeatureAssistedOTDR, LifecycleModelConfig
from .model_functions.zero_shot import require_cuda
from .study_state import append_jsonl, atomic_json, config_hash, utc_now


PILOT_PAIRS = ((1, 2), (3, 5), (6, 7))
STAGES = {
    "short": {"epochs": 1, "steps": 3, "survivors": 8, "pairs": PILOT_PAIRS[:1]},
    "intermediate": {"epochs": 3, "steps": 8, "survivors": 3, "pairs": PILOT_PAIRS},
    "full": {"epochs": 8, "steps": 20, "survivors": 1, "pairs": PILOT_PAIRS},
}


@dataclass(frozen=True)
class RepresentationCandidate:
    model: LifecycleModelConfig
    training: LifecycleTrainingConfig

    @property
    def candidate_id(self) -> str:
        payload = {"model": asdict(self.model), "training": {**asdict(self.training), "seed": None}}
        return config_hash(payload)


def generate_representation_candidates(count: int = 24) -> list[RepresentationCandidate]:
    rng = np.random.default_rng(260725)
    result: list[RepresentationCandidate] = []
    seen: set[str] = set()
    modes = ["late_fusion"] * 6 + ["morphology_only", "context_only"]
    while len(result) < count:
        model = LifecycleModelConfig(
            width=int(rng.choice([32, 48, 64])),
            embedding_dim=int(rng.choice([32, 48, 64])),
            context_width=int(rng.choice([16, 24, 32])),
            blocks=int(rng.choice([2, 3])),
            kernel_size=int(rng.choice([3, 5])),
            dropout=float(rng.choice([0.0, 0.1, 0.2])),
            fusion=str(rng.choice(["gated", "concat"])),
            mode=str(rng.choice(modes)),
            canonicalize=bool(rng.choice([False, True])),
        )
        training = LifecycleTrainingConfig(
            learning_rate=float(rng.choice([2e-4, 4e-4, 7e-4])),
            weight_decay=float(rng.choice([1e-5, 1e-4, 5e-4])),
            supcon_weight=float(rng.choice([0.0, 0.05, 0.15])),
            localization_weight=float(rng.choice([0.0, 0.03, 0.08])),
            scalar_dropout=float(rng.choice([0.0, 0.1, 0.25])),
            trace_noise_std=float(rng.choice([0.0, 0.01, 0.025])),
            context_noise_std=float(rng.choice([0.0, 0.02, 0.05])),
            seed=42,
        )
        candidate = RepresentationCandidate(model, training)
        if candidate.candidate_id not in seen:
            result.append(candidate)
            seen.add(candidate.candidate_id)
    return result


def generate_kpsc_candidates() -> list[dict[str, object]]:
    result = []
    for prototypes, fusion, calibration in itertools.product(
        (1, 4), ("confidence", "best_single", "weighted", "sirc", "meta_p", "robust_regret"), ("empirical", "conformal")
    ):
        result.append({
            "prototypes_per_class": prototypes,
            "prototype_metric": "cosine",
            "knn_k": 10,
            "fusion": fusion,
            "fusion_weights": None,
            "calibration_mode": calibration,
            "normal_far_cap": 0.0125,
            "known_acceptance_floor": 0.95,
        })
    return result


def generate_cfe_candidates() -> list[dict[str, object]]:
    result = []
    for method, metric, teen_alpha in itertools.product(
        ("mean", "medoid", "median", "quality_weighted"),
        ("cosine", "euclidean", "diagonal_mahalanobis"),
        (0.0, 0.2),
    ):
        result.append({
            "prototype_method": method,
            "metric": metric,
            "teen_alpha": teen_alpha,
            "teen_temperature": 0.5,
            "shots": [1, 3, 5],
            "draws": 20,
        })
    return result


def _pseudo_pairs(outer_holdout: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    available = [value for value in range(1, 8) if value not in outer_holdout]
    candidates = list(itertools.combinations(available, 2))
    return candidates[0], candidates[len(candidates) // 2], candidates[-1]


def _stress_batch(batch, *, seed: int):
    generator = torch.Generator().manual_seed(seed)
    trace = batch.trace * 1.2 + torch.randn(batch.trace.shape, generator=generator) * 0.08
    context = batch.context.clone()
    missing = batch.context_missing.clone()
    context[:, 1] += torch.randn(len(context), generator=generator) * 0.25
    drop = torch.rand(len(context), generator=generator) < 0.3
    context[drop, 2] = 0
    missing[drop, 2] = 1
    return replace(batch, trace=trace, context=context, context_missing=missing)


def _evaluate_representation_task(
    task: dict[str, pd.DataFrame],
    *,
    pseudo_classes: tuple[int, int],
    candidate: RepresentationCandidate,
    training: LifecycleTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    scaler = fit_lifecycle_scaler(task["train"], regime="full")
    batches = {name: transform_lifecycle(part, scaler) for name, part in task.items()}
    selector_frame, threshold_frame = split_known_calibration(
        task["calibration"], seed=training.seed
    )
    batches["selector"] = transform_lifecycle(selector_frame, scaler)
    batches["threshold"] = transform_lifecycle(threshold_frame, scaler)
    model, _ = train_lifecycle_model(
        batches["train"], batches["selector"], device=device,
        model_config=candidate.model, training_config=training,
    )
    output = {name: infer_lifecycle_model(model, batch, device=device) for name, batch in batches.items()}
    seen_labels = batches["seen_test"].labels.numpy()
    seen_prediction = output["seen_test"]["logits"].argmax(1).numpy()
    known_balanced = float(balanced_accuracy_score(seen_labels, seen_prediction))
    stressed_output = infer_lifecycle_model(model, _stress_batch(batches["seen_test"], seed=training.seed + 77), device=device)
    stress_balanced = float(balanced_accuracy_score(seen_labels, stressed_output["logits"].argmax(1).numpy()))

    train_embedding = output["train"]["embedding"].numpy()
    train_labels = batches["train"].labels.numpy()
    distance = DistanceReference.fit(train_embedding, train_labels)
    bank = PrototypeBank.fit(train_embedding, train_labels, prototypes_per_class=4, seed=training.seed)
    names, selector_components = assemble_components(
        logits=output["selector"]["logits"].numpy(),
        embeddings=output["selector"]["embedding"].numpy(),
        distance_reference=distance,
        prototype_bank=bank,
    )
    normalizer = EmpiricalCDFNormalizer.fit(selector_components, names)
    _, threshold_components = assemble_components(
        logits=output["threshold"]["logits"].numpy(),
        embeddings=output["threshold"]["embedding"].numpy(),
        distance_reference=distance,
        prototype_bank=bank,
    )
    seen_names, seen_components = assemble_components(
        logits=output["seen_test"]["logits"].numpy(),
        embeddings=output["seen_test"]["embedding"].numpy(),
        distance_reference=distance,
        prototype_bank=bank,
    )
    _, pseudo_components = assemble_components(
        logits=output["pseudo_query"]["logits"].numpy(),
        embeddings=output["pseudo_query"]["embedding"].numpy(),
        distance_reference=distance,
        prototype_bank=bank,
    )
    calibration_score = fuse_scores(normalizer.transform(threshold_components), method="robust_regret")
    threshold = fit_joint_threshold(calibration_score, batches["threshold"].labels.numpy())
    combined_score = fuse_scores(
        normalizer.transform(np.vstack((seen_components, pseudo_components))), method="robust_regret"
    )
    combined_labels = np.r_[seen_labels, batches["pseudo_query"].labels.numpy()]
    combined_prediction = np.r_[
        seen_prediction, output["pseudo_query"]["logits"].argmax(1).numpy()
    ]
    kpsc = evaluate_joint_operating_point(
        combined_score, combined_labels, combined_prediction,
        holdout=pseudo_classes, calibration=threshold,
    )

    session = EnrollmentSession.from_base(train_embedding, train_labels, metric="cosine")
    support_embedding = output["support"]["embedding"].numpy()
    support_labels = batches["support"].labels.numpy()
    for class_id in pseudo_classes:
        selected = np.flatnonzero(support_labels == class_id)[:1]
        session = session.enroll(class_id, support_embedding[selected], teen_alpha=0.2)
    query_embedding = np.vstack((output["seen_test"]["embedding"].numpy(), output["pseudo_query"]["embedding"].numpy()))
    cfe_labels = np.r_[seen_labels, batches["pseudo_query"].labels.numpy()]
    cfe = hard_prediction_metrics(
        cfe_labels, session.predict(query_embedding),
        base_class_ids=tuple(sorted(np.unique(train_labels))),
        enrolled_class_ids=pseudo_classes,
    )
    feasible = float(bool(kpsc["constraints_met"]))
    return {
        "known_balanced_accuracy": known_balanced,
        "stress_balanced_accuracy": stress_balanced,
        "stress_retention": stress_balanced / max(known_balanced, 1e-8),
        "kpsc_unknown_recall": float(kpsc["unknown_recall"]),
        "kpsc_worst_recall": float(kpsc["worst_fault_recall"]),
        "kpsc_feasible": feasible,
        "cfe_harmonic_mean": float(cfe["harmonic_mean"]),
        "cfe_worst_recall": float(cfe["worst_enrolled_recall"]),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    result = {f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in keys}
    result["representation_objective"] = (
        0.35 * result["mean_known_balanced_accuracy"]
        + 0.25 * result["mean_stress_balanced_accuracy"]
        + 0.20 * result["mean_cfe_harmonic_mean"]
        + 0.20 * result["mean_kpsc_worst_recall"]
        - (1 - result["mean_kpsc_feasible"]) * 0.10
    )
    return result


def run_representation_sweep(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    device: torch.device | str,
) -> dict[str, object]:
    device = require_cuda(str(device))
    grouped = attach_input_groups(frame)
    candidates = generate_representation_candidates()
    survivors = candidates
    all_rows: list[dict[str, object]] = []
    output_path = study_root / "sweeps" / "representation_trials.jsonl"
    for stage, budget in STAGES.items():
        stage_rows = []
        for candidate in survivors:
            task_metrics = []
            for outer_pair in budget["pairs"]:
                outer = build_event_openworld_fold(grouped, holdout=outer_pair, seed=42)
                for inner_index, pseudo_pair in enumerate(_pseudo_pairs(outer_pair)):
                    task = _inner_task(outer.train, outer.validation, pseudo_pair)
                    training = replace(
                        candidate.training,
                        epochs=budget["epochs"],
                        steps_per_epoch=budget["steps"],
                        patience=budget["epochs"],
                        seed=42 + inner_index,
                    )
                    metrics = _evaluate_representation_task(
                        task, pseudo_classes=pseudo_pair, candidate=candidate,
                        training=training, device=device,
                    )
                    task_metrics.append(metrics)
            aggregate = _aggregate(task_metrics)
            row = {
                "timestamp": utc_now(),
                "stage": stage,
                "candidate_id": candidate.candidate_id,
                "model": asdict(candidate.model),
                "training": asdict(candidate.training),
                "tasks": len(task_metrics),
                **aggregate,
            }
            append_jsonl(output_path, row)
            all_rows.append(row)
            stage_rows.append((aggregate["representation_objective"], candidate, row))
        stage_rows.sort(key=lambda value: value[0], reverse=True)
        survivors = [value[1] for value in stage_rows[: budget["survivors"]]]
        atomic_json(study_root / "sweeps" / f"representation_{stage}_selection.json", {
            "stage": stage,
            "candidates": len(stage_rows),
            "survivors": [candidate.candidate_id for candidate in survivors],
            "ranking": [value[2] for value in stage_rows],
        })
    winner = survivors[0]
    result = {
        "schema_version": 1,
        "selection": "nested_group_safe_successive_halving_pseudo_unseen_only",
        "candidate_count": len(candidates),
        "pilot_pairs": [list(value) for value in PILOT_PAIRS],
        "inner_folds_per_pair": 3,
        "winner_id": winner.candidate_id,
        "shared_backbone": asdict(winner.model),
        "training": asdict(winner.training),
        "kpsc_candidates": len(generate_kpsc_candidates()),
        "cfe_candidates": len(generate_cfe_candidates()),
    }
    atomic_json(study_root / "sweeps" / "representation_finalist.json", result)
    return result


def freeze_default_posthoc_finalists(study_root: Path, representation: dict[str, object]) -> dict[str, object]:
    """Predeclare post-hoc candidates; selection values are filled by the evaluation stage."""
    payload = {
        "schema_version": 1,
        "selection_status": "representation_frozen_posthoc_selection_pending",
        "shared_backbone": representation["shared_backbone"],
        "training": representation["training"],
        "kpsc": generate_kpsc_candidates()[0],
        "cfe": generate_cfe_candidates()[0],
        "candidate_counts": {"representation_mcf": 24, "kpsc": 24, "cfe": 24},
    }
    atomic_json(study_root / "configs" / "finalists.pending.json", payload)
    return payload


def _ranked_support_positions(
    frame: pd.DataFrame,
    *,
    class_id: int,
    shots: int,
    draw: int,
) -> np.ndarray:
    candidates = np.flatnonzero(frame["Class"].to_numpy(dtype=int) == class_id)
    ranked = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"inner-support:{draw}:{shots}:{class_id}:{frame.iloc[index]['_input_group']}".encode()
        ).hexdigest(),
    )
    return np.asarray(ranked[:shots], dtype=int)


def _posthoc_task(
    task: dict[str, pd.DataFrame],
    *,
    pseudo_classes: tuple[int, int],
    representation: RepresentationCandidate,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scaler = fit_lifecycle_scaler(task["train"], regime="full")
    selector_frame, threshold_frame = split_known_calibration(task["calibration"], seed=seed)
    frames = {**task, "selector": selector_frame, "threshold": threshold_frame}
    batches = {name: transform_lifecycle(part, scaler) for name, part in frames.items()}
    training = replace(
        representation.training, epochs=8, steps_per_epoch=20, patience=8, seed=seed
    )
    model, _ = train_lifecycle_model(
        batches["train"], batches["selector"], device=device,
        model_config=representation.model, training_config=training,
    )
    outputs = {name: infer_lifecycle_model(model, batch, device=device) for name, batch in batches.items()}
    train_embedding = outputs["train"]["embedding"].numpy()
    train_labels = batches["train"].labels.numpy()
    reference = DistanceReference.fit(train_embedding, train_labels)
    query_labels = np.r_[batches["seen_test"].labels.numpy(), batches["pseudo_query"].labels.numpy()]
    query_prediction = np.r_[
        outputs["seen_test"]["logits"].argmax(1).numpy(),
        outputs["pseudo_query"]["logits"].argmax(1).numpy(),
    ]

    kpsc_rows: list[dict[str, object]] = []
    component_cache: dict[int, tuple[tuple[str, ...], dict[str, np.ndarray]]] = {}
    for count in (1, 4):
        bank = PrototypeBank.fit(
            train_embedding, train_labels, prototypes_per_class=count, seed=seed
        )
        values: dict[str, np.ndarray] = {}
        names: tuple[str, ...] | None = None
        for name in ("selector", "threshold", "seen_test", "pseudo_query"):
            names, values[name] = assemble_components(
                logits=outputs[name]["logits"].numpy(),
                embeddings=outputs[name]["embedding"].numpy(),
                distance_reference=reference,
                prototype_bank=bank,
            )
        component_cache[count] = (names, values)
    for candidate in generate_kpsc_candidates():
        names, components = component_cache[int(candidate["prototypes_per_class"])]
        normalizer = EmpiricalCDFNormalizer.fit(components["selector"], names)
        calibration_score = fuse_scores(
            normalizer.transform(components["threshold"]),
            method=str(candidate["fusion"]),
        )
        threshold = fit_joint_threshold(
            calibration_score,
            batches["threshold"].labels.numpy(),
            normal_far_cap=float(candidate["normal_far_cap"]),
            known_acceptance_floor=float(candidate["known_acceptance_floor"]),
            mode=str(candidate["calibration_mode"]),
        )
        score = fuse_scores(
            normalizer.transform(np.vstack((components["seen_test"], components["pseudo_query"]))),
            method=str(candidate["fusion"]),
        )
        metric = evaluate_joint_operating_point(
            score, query_labels, query_prediction,
            holdout=pseudo_classes, calibration=threshold,
        )
        kpsc_rows.append({
            "candidate_id": config_hash(candidate),
            "candidate": candidate,
            "pseudo_pair": list(pseudo_classes),
            **{key: metric[key] for key in (
                "normal_far", "known_fault_acceptance", "unknown_recall",
                "worst_fault_recall", "accepted_known_accuracy", "constraints_met"
            )},
        })

    cfe_rows: list[dict[str, object]] = []
    query_embedding = np.vstack((
        outputs["seen_test"]["embedding"].numpy(),
        outputs["pseudo_query"]["embedding"].numpy(),
    ))
    support_embedding = outputs["support"]["embedding"].numpy()
    support_quality = torch.sigmoid(outputs["support"]["competence"]).numpy()
    base_ids = tuple(sorted(int(value) for value in np.unique(train_labels)))
    for candidate in generate_cfe_candidates():
        draw_rows = []
        for shots in (1, 3, 5):
            for draw in range(3):
                session = EnrollmentSession.from_base(
                    train_embedding, train_labels, metric=str(candidate["metric"])
                )
                for class_id in pseudo_classes:
                    positions = _ranked_support_positions(
                        task["support"], class_id=class_id, shots=shots, draw=draw
                    )
                    session = session.enroll(
                        class_id,
                        support_embedding[positions],
                        method=str(candidate["prototype_method"]),
                        quality=support_quality[positions] if candidate["prototype_method"] == "quality_weighted" else None,
                        teen_alpha=float(candidate["teen_alpha"]),
                        teen_temperature=float(candidate["teen_temperature"]),
                        support_group_ids=tuple(task["support"].iloc[positions]["_input_group"].astype(str)),
                    )
                metric = hard_prediction_metrics(
                    query_labels, session.predict(query_embedding),
                    base_class_ids=base_ids, enrolled_class_ids=pseudo_classes,
                )
                draw_rows.append((shots, metric))
        cfe_rows.append({
            "candidate_id": config_hash(candidate),
            "candidate": candidate,
            "pseudo_pair": list(pseudo_classes),
            "mean_harmonic_mean": float(np.mean([value["harmonic_mean"] for _, value in draw_rows])),
            "mean_base_accuracy": float(np.mean([value["base_accuracy"] for _, value in draw_rows])),
            "mean_enrolled_accuracy": float(np.mean([value["enrolled_accuracy"] for _, value in draw_rows])),
            "worst_enrolled_recall": float(min(value["worst_enrolled_recall"] for _, value in draw_rows)),
            "one_shot_harmonic_mean": float(np.mean([value["harmonic_mean"] for shot, value in draw_rows if shot == 1])),
            "five_shot_harmonic_mean": float(np.mean([value["harmonic_mean"] for shot, value in draw_rows if shot == 5])),
        })
    return kpsc_rows, cfe_rows


def run_posthoc_sweeps(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    representation_result: dict[str, object],
    device: torch.device | str,
) -> dict[str, object]:
    """Select KPSC and CFE using only pseudo-unseen inner tasks."""
    device = require_cuda(str(device))
    grouped = attach_input_groups(frame)
    representation = RepresentationCandidate(
        LifecycleModelConfig(**representation_result["shared_backbone"]),
        LifecycleTrainingConfig(**representation_result["training"]),
    )
    kpsc_rows: list[dict[str, object]] = []
    cfe_rows: list[dict[str, object]] = []
    for outer_index, outer_pair in enumerate(PILOT_PAIRS):
        outer = build_event_openworld_fold(grouped, holdout=outer_pair, seed=42)
        for inner_index, pseudo_pair in enumerate(_pseudo_pairs(outer_pair)):
            task = _inner_task(outer.train, outer.validation, pseudo_pair)
            left, right = _posthoc_task(
                task, pseudo_classes=pseudo_pair, representation=representation,
                seed=4200 + outer_index * 10 + inner_index, device=device,
            )
            kpsc_rows.extend(left)
            cfe_rows.extend(right)
    for row in kpsc_rows:
        append_jsonl(study_root / "sweeps" / "kpsc_trials.jsonl", row)
    for row in cfe_rows:
        append_jsonl(study_root / "sweeps" / "cfe_trials.jsonl", row)

    kpsc_summary = []
    for candidate in generate_kpsc_candidates():
        candidate_id = config_hash(candidate)
        rows = [row for row in kpsc_rows if row["candidate_id"] == candidate_id]
        feasibility = float(np.mean([bool(row["constraints_met"]) for row in rows]))
        summary = {
            "candidate_id": candidate_id,
            "candidate": candidate,
            "tasks": len(rows),
            "feasibility_rate": feasibility,
            "mean_unknown_recall": float(np.mean([row["unknown_recall"] for row in rows])),
            "worst_fault_recall": float(min(row["worst_fault_recall"] for row in rows)),
            "mean_known_acceptance": float(np.mean([row["known_fault_acceptance"] for row in rows])),
            "mean_normal_far": float(np.mean([row["normal_far"] for row in rows])),
            "mean_accepted_known_accuracy": float(np.mean([row["accepted_known_accuracy"] for row in rows])),
        }
        summary["selection_score"] = (
            2 * feasibility + summary["worst_fault_recall"] + summary["mean_unknown_recall"]
            + 0.25 * summary["mean_accepted_known_accuracy"]
        )
        kpsc_summary.append(summary)
    kpsc_summary.sort(key=lambda row: row["selection_score"], reverse=True)

    cfe_summary = []
    for candidate in generate_cfe_candidates():
        candidate_id = config_hash(candidate)
        rows = [row for row in cfe_rows if row["candidate_id"] == candidate_id]
        summary = {
            "candidate_id": candidate_id,
            "candidate": candidate,
            "tasks": len(rows),
            "mean_harmonic_mean": float(np.mean([row["mean_harmonic_mean"] for row in rows])),
            "mean_base_accuracy": float(np.mean([row["mean_base_accuracy"] for row in rows])),
            "mean_enrolled_accuracy": float(np.mean([row["mean_enrolled_accuracy"] for row in rows])),
            "worst_enrolled_recall": float(min(row["worst_enrolled_recall"] for row in rows)),
            "one_shot_harmonic_mean": float(np.mean([row["one_shot_harmonic_mean"] for row in rows])),
            "five_shot_harmonic_mean": float(np.mean([row["five_shot_harmonic_mean"] for row in rows])),
        }
        summary["selection_score"] = (
            summary["mean_harmonic_mean"] + 0.5 * summary["worst_enrolled_recall"]
            + 0.25 * summary["mean_base_accuracy"]
        )
        cfe_summary.append(summary)
    cfe_summary.sort(key=lambda row: row["selection_score"], reverse=True)
    atomic_json(study_root / "sweeps" / "kpsc_selection.json", {"ranking": kpsc_summary})
    atomic_json(study_root / "sweeps" / "cfe_selection.json", {"ranking": cfe_summary})
    finalists = {
        "schema_version": 1,
        "selection_status": "frozen_before_outer_query",
        "selection_protocol": "nested_group_safe_pseudo_unseen_only",
        "shared_backbone": representation_result["shared_backbone"],
        "training": {
            **representation_result["training"],
            "epochs": 8,
            "steps_per_epoch": 20,
            "patience": 8,
        },
        "kpsc": kpsc_summary[0]["candidate"],
        "cfe": cfe_summary[0]["candidate"],
        "candidate_counts": {"representation_mcf": 24, "kpsc": 24, "cfe": 24},
        "winner_ids": {
            "representation": representation_result["winner_id"],
            "kpsc": kpsc_summary[0]["candidate_id"],
            "cfe": cfe_summary[0]["candidate_id"],
        },
    }
    atomic_json(study_root / "configs" / "finalists.json", finalists)
    return finalists
