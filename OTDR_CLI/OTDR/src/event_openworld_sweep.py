from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .event_openworld_data import attach_input_groups, build_event_openworld_fold
from .event_openworld_graph import class_prototypes, prototype_predict, seeded_graph_enrollment
from .event_openworld_metrics import ScoreNormalizer, evaluate_zero_day
from .event_openworld_training import ECConfig, PC2Config, SGMEConfig, infer_event_model, train_ec_czsl, train_pc2_oe
from .model_functions.event_openworld import load_event_recipes
from .model_functions.zero_shot import require_cuda
from .study_metrics import post_enrollment_metrics
from .study_state import append_jsonl, atomic_json, config_hash, utc_now
from .zero_shot_data import INPUT_COLUMNS


PILOT_PAIRS = ((1, 2), (3, 5), (6, 7))
STAGES = {
    "short": {"epochs": 1, "steps_per_epoch": 4, "survivors": 12},
    "intermediate": {"epochs": 3, "steps_per_epoch": 12, "survivors": 4},
    "full": {"epochs": 6, "steps_per_epoch": 24, "survivors": 1},
}


def _candidate_id(config: ECConfig | PC2Config) -> str:
    payload = asdict(config)
    payload.pop("seed", None)
    return config_hash(payload)


def generate_ec_candidates(count: int = 36) -> list[ECConfig]:
    rng = np.random.default_rng(1001)
    values: list[ECConfig] = []
    while len(values) < count:
        config = ECConfig(
            learning_rate=float(rng.choice([2e-4, 4e-4, 7e-4, 1e-3])),
            weight_decay=float(rng.choice([0.0, 1e-5, 1e-4, 5e-4])),
            width=int(rng.choice([48, 64, 72, 96])),
            latent_dim=int(rng.choice([40, 56, 72, 96])),
            patch_size=int(rng.choice([11, 15, 19])),
            dropout=float(rng.choice([0.0, 0.1, 0.2])),
            factor_weight=float(rng.choice([0.25, 0.50, 0.75, 1.0])),
            uncertainty_weight=float(rng.choice([0.0, 0.03, 0.08])),
            residual_penalty=float(rng.choice([0.0, 0.01, 0.03])),
            class_dropout_count=int(rng.choice([1, 2])),
            calibration=str(rng.choice(["global", "mondrian", "normalized"])),
            fusion_weights=tuple(float(x) for x in [
                [0.0, 1.0, 0.0, 0.0], [0.1, 0.7, 0.1, 0.1], [0.25, 0.55, 0.1, 0.1], [0.2, 0.6, 0.0, 0.2]
            ][int(rng.integers(4))]),
            seed=42,
        )
        if _candidate_id(config) not in {_candidate_id(item) for item in values}:
            values.append(config)
    return values


def generate_pc2_candidates(count: int = 36) -> list[PC2Config]:
    rng = np.random.default_rng(2002)
    values: list[PC2Config] = []
    while len(values) < count:
        config = PC2Config(
            learning_rate=float(rng.choice([2e-4, 4e-4, 7e-4, 1e-3])),
            weight_decay=float(rng.choice([0.0, 1e-5, 1e-4, 5e-4])),
            width=int(rng.choice([48, 64, 80, 96])),
            latent_dim=int(rng.choice([40, 56, 72, 96])),
            patch_size=int(rng.choice([11, 15, 19])),
            dropout=float(rng.choice([0.0, 0.1, 0.2])),
            factor_weight=float(rng.choice([0.25, 0.5, 0.75])),
            named_weight=float(rng.choice([0.2, 0.4, 0.7])),
            oe_weight=float(rng.choice([0.15, 0.35, 0.6, 0.9])),
            cvar_weight=float(rng.choice([0.0, 0.2, 0.4, 0.7])),
            energy_margin=float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            cvar_fraction=float(rng.choice([0.1, 0.2, 0.4])),
            synthetic_fraction=float(rng.choice([0.2, 0.35, 0.5])),
            calibration=str(rng.choice(["global", "mondrian", "normalized"])),
            fusion_weights=tuple(float(x) for x in [
                [1.0, 0.0, 0.0, 0.0], [0.5, 0.4, 0.1, 0.0], [0.35, 0.5, 0.1, 0.05], [0.6, 0.25, 0.1, 0.05]
            ][int(rng.integers(4))]),
            seed=42,
        )
        if _candidate_id(config) not in {_candidate_id(item) for item in values}:
            values.append(config)
    return values


def _transform(frame: pd.DataFrame, scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    x = scaler.transform(frame[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)).astype(np.float32)
    y = frame["Class"].to_numpy(dtype=np.int64, copy=True)
    return torch.from_numpy(x), torch.from_numpy(y)


def _hash_split(frame: pd.DataFrame, fraction: float, namespace: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = sorted(range(len(frame)), key=lambda index: hashlib.sha256(
        f"{namespace}:{frame.iloc[index]['_input_group']}".encode()
    ).hexdigest())
    cut = max(1, min(len(ranked) - 1, int(len(ranked) * fraction)))
    return frame.iloc[ranked[:cut]].copy(), frame.iloc[ranked[cut:]].copy()


def _inner_task(outer_train: pd.DataFrame, outer_validation: pd.DataFrame, pseudo_classes: tuple[int, int]) -> dict[str, pd.DataFrame]:
    train = outer_train[~outer_train["Class"].isin(pseudo_classes)].copy()
    support_parts, adaptation_parts, query_parts = [], [], []
    for pseudo_class in pseudo_classes:
        pseudo = outer_validation[outer_validation["Class"] == pseudo_class].copy()
        enrollment, pseudo_query = _hash_split(pseudo, 0.35, f"pseudo:{pseudo_classes}:{pseudo_class}")
        support, adaptation = _hash_split(enrollment, 2 / 7, f"pseudo-adaptation:{pseudo_classes}:{pseudo_class}")
        support_parts.append(support); adaptation_parts.append(adaptation); query_parts.append(pseudo_query)
    calibration_parts, test_parts = [], []
    for class_id, class_frame in outer_validation[~outer_validation["Class"].isin(pseudo_classes)].groupby("Class"):
        calibration, test = _hash_split(class_frame, 0.5, f"cal:{pseudo_classes}:{class_id}")
        calibration_parts.append(calibration)
        test_parts.append(test)
    return {
        "train": train,
        "calibration": pd.concat(calibration_parts, ignore_index=True),
        "seen_test": pd.concat(test_parts, ignore_index=True),
        "support": pd.concat(support_parts, ignore_index=True),
        "adaptation": pd.concat(adaptation_parts, ignore_index=True),
        "pseudo_query": pd.concat(query_parts, ignore_index=True),
    }


def _inner_sgme_task(
    outer_train: pd.DataFrame,
    outer_validation: pd.DataFrame,
    pseudo_classes: tuple[int, int],
) -> dict[str, pd.DataFrame]:
    """Inner task with ample, disjoint SGME buffers and untouched query groups."""
    base = _inner_task(outer_train, outer_validation, pseudo_classes)
    unused_pseudo = pd.concat([
        outer_train[outer_train["Class"].isin(pseudo_classes)],
        outer_validation[outer_validation["Class"].isin(pseudo_classes)],
    ], ignore_index=True)
    support_parts, adaptation_parts, query_parts = [], [], []
    for pseudo_class in pseudo_classes:
        values = unused_pseudo[unused_pseudo["Class"] == pseudo_class].copy()
        support, remainder = _hash_split(values, 0.05, f"sgme-support:{pseudo_classes}:{pseudo_class}")
        adaptation, query = _hash_split(remainder, 0.20 / 0.95, f"sgme-adaptation:{pseudo_classes}:{pseudo_class}")
        support_parts.append(support)
        adaptation_parts.append(adaptation)
        query_parts.append(query)
    base["support"] = pd.concat(support_parts, ignore_index=True)
    base["adaptation"] = pd.concat(adaptation_parts, ignore_index=True)
    base["pseudo_query"] = pd.concat(query_parts, ignore_index=True)
    group_sets = {name: set(part["_input_group"]) for name, part in base.items()}
    names = list(group_sets)
    for left, left_name in enumerate(names):
        for right_name in names[left + 1:]:
            if group_sets[left_name] & group_sets[right_name]:
                raise AssertionError(f"SGME inner group leakage: {left_name}/{right_name}")
    return base


def _evaluate_inner(
    *,
    approach: str,
    task: dict[str, pd.DataFrame],
    pseudo_classes: tuple[int, int],
    config: ECConfig | PC2Config,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    scaler = StandardScaler().fit(task["train"][INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    tensors = {name: _transform(value, scaler) for name, value in task.items()}
    train_x, train_y = tensors["train"]
    if approach == "ec":
        model, metadata = train_ec_czsl(train_x, train_y, recipe_means, recipe_stds, device=device, config=config)
    else:
        model, metadata = train_pc2_oe(
            train_x, train_y, recipe_means, recipe_stds,
            snr_mean=float(scaler.mean_[0]), snr_scale=float(scaler.scale_[0]), device=device, config=config,
        )
    known_class_ids = sorted(int(value) for value in train_y.unique())
    outputs = {name: infer_event_model(
                   model, value[0], recipe_means, recipe_stds, device=device,
                   known_class_ids=known_class_ids,
               )
               for name, value in tensors.items()}
    normal = tensors["calibration"][1].numpy() == 0
    normalizer = ScoreNormalizer.fit(outputs["calibration"]["novelty_components"].numpy()[normal], config.fusion_weights)
    score = {name: normalizer.transform(value["novelty_components"].numpy()) for name, value in outputs.items()}
    test_y = torch.cat([tensors["seen_test"][1], tensors["pseudo_query"][1]]).numpy()
    test_score = np.concatenate([score["seen_test"], score["pseudo_query"]])
    test_snr = torch.cat([tensors["seen_test"][0][:, 0], tensors["pseudo_query"][0][:, 0]]).numpy()
    test_logits = torch.cat([outputs["seen_test"]["logits"], outputs["pseudo_query"]["logits"]]).numpy()
    operational_predicted = np.asarray(known_class_ids)[test_logits[:, known_class_ids].argmax(1)]
    zero = evaluate_zero_day(
        validation_normal_score=score["calibration"][normal],
        validation_normal_snr=tensors["calibration"][0][:, 0].numpy()[normal],
        test_score=test_score, test_snr=test_snr, true_labels=test_y, predicted=operational_predicted,
        holdout=pseudo_classes, calibration=config.calibration,
    )
    op = zero["operating_points"]["far_0.010"]
    query_y = tensors["pseudo_query"][1].numpy()
    strict_local = outputs["pseudo_query"]["logits"][:, list(pseudo_classes)].argmax(1).numpy()
    strict_pred = np.asarray(pseudo_classes)[strict_local]
    strict = float(np.mean([np.mean(strict_pred[query_y == class_id] == class_id) for class_id in pseudo_classes]))
    support_indices = torch.cat([torch.nonzero(tensors["support"][1] == class_id, as_tuple=False).flatten()[:1]
                                 for class_id in pseudo_classes])
    embeddings = torch.cat([outputs["train"]["embedding"], outputs["support"]["embedding"][support_indices]])
    enrollment_y = torch.cat([train_y, tensors["support"][1][support_indices]])
    prototypes, _ = class_prototypes(embeddings, enrollment_y)
    post_embeddings = torch.cat([outputs["seen_test"]["embedding"], outputs["pseudo_query"]["embedding"]])
    post_y = torch.cat([tensors["seen_test"][1], tensors["pseudo_query"][1]])
    post_pred, _ = prototype_predict(post_embeddings, prototypes)
    post = post_enrollment_metrics(post_y.numpy(), post_pred.numpy(),
                                   seen_ids=sorted(int(value) for value in train_y.unique()), unseen_ids=pseudo_classes)
    return {
        "unknown_recall": float(op["unknown_recall"]),
        "worst_fault_recall": float(op["worst_fault_recall"]),
        "known_acceptance": float(op["known_acceptance"]),
        "observed_normal_far": float(op["observed_normal_far"]),
        "strict_accuracy": strict,
        "post_h": float(post["harmonic_mean"]),
        "parameters": float(metadata["parameter_count"]),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float | bool]:
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    worst = float(min(row["worst_fault_recall"] for row in rows))
    result: dict[str, float | bool] = {
        "unknown_recall": mean("unknown_recall"),
        "worst_fault_recall": worst,
        "known_acceptance": mean("known_acceptance"),
        "observed_normal_far": mean("observed_normal_far"),
        "strict_accuracy": mean("strict_accuracy"),
        "post_h": mean("post_h"),
        "parameters": mean("parameters"),
    }
    result["feasible"] = result["observed_normal_far"] <= 0.0125 and result["known_acceptance"] >= 0.95
    result["calibration_error"] = abs(float(result["observed_normal_far"]) - 0.01)
    return result


def _rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(row["aggregate"]["feasible"])),
        float(row["aggregate"]["unknown_recall"]),
        float(row["aggregate"]["worst_fault_recall"]),
        float(row["aggregate"]["strict_accuracy"]),
        float(row["aggregate"]["post_h"]),
        -float(row["aggregate"]["calibration_error"]),
        -float(row["aggregate"]["parameters"]),
    )


def run_neural_sweep(
    *,
    approach: str,
    frame: pd.DataFrame,
    study_root: Path,
    recipe_path: Path,
    device: torch.device,
    resume: bool = True,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    if approach not in {"ec", "pc2"}:
        raise ValueError("Neural sweep approach must be ec or pc2")
    frame = attach_input_groups(frame) if "_input_group" not in frame else frame
    candidates: list[ECConfig | PC2Config] = generate_ec_candidates() if approach == "ec" else generate_pc2_candidates()
    recipes = load_event_recipes(recipe_path)
    pilot_folds = {pair: build_event_openworld_fold(frame, holdout=pair, seed=42) for pair in PILOT_PAIRS}
    selector_version = "v6" if approach == "ec" else "v4"
    sweep_root = study_root / "sweeps" / f"{approach}_{selector_version}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[dict[str, Any]]] = {}
    survivors = candidates
    for stage_name, budget in STAGES.items():
        stage_rows: list[dict[str, Any]] = []
        for base_config in survivors:
            config = replace(base_config, epochs=budget["epochs"], steps_per_epoch=budget["steps_per_epoch"])
            candidate_id = _candidate_id(base_config)
            path = sweep_root / stage_name / f"{candidate_id}.json"
            if resume and path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("candidate_id") == candidate_id and row.get("stage") == stage_name and len(row.get("tasks", [])) == 9:
                    stage_rows.append(row)
                    continue
            tasks: list[dict[str, Any]] = []
            for pilot_index, pair in enumerate(PILOT_PAIRS):
                fold = pilot_folds[pair]
                seen_faults = sorted(set(range(1, 8)) - set(pair))
                pseudo_pairs = [(seen_faults[0], seen_faults[1]), (seen_faults[2], seen_faults[3]),
                                (seen_faults[4], seen_faults[1])]
                for inner_index, pseudo_pair in enumerate(pseudo_pairs):
                    task_config = replace(config, seed=42 + pilot_index * 100 + inner_index)
                    try:
                        metrics = _evaluate_inner(
                            approach=approach,
                            task=_inner_task(fold.train, fold.validation, pseudo_pair),
                            pseudo_classes=pseudo_pair,
                            config=task_config,
                            recipe_means=recipes["means"], recipe_stds=recipes["stds"], device=device,
                        )
                    except Exception as exc:
                        append_jsonl(study_root / "failures.jsonl", {
                            "event": "sweep_failed", "timestamp": utc_now(), "approach": approach,
                            "stage": stage_name, "candidate_id": candidate_id, "pilot_pair": list(pair),
                            "inner_fold": inner_index, "pseudo_classes": list(pseudo_pair),
                            "exception_type": type(exc).__name__, "exception": str(exc),
                        })
                        raise
                    tasks.append({"pilot_pair": list(pair), "inner_fold": inner_index, "pseudo_classes": list(pseudo_pair), **metrics})
                    torch.cuda.empty_cache()
            row = {
                "schema_version": 1, "approach": approach, "stage": stage_name,
                "candidate_id": candidate_id, "base_config": asdict(base_config), "budget_config": asdict(config),
                "tasks": tasks, "aggregate": _aggregate(tasks),
                "outer_heldout_real_evaluated": False,
            }
            atomic_json(path, row)
            append_jsonl(study_root / "experiment_registry.jsonl", {
                "event": "sweep_candidate_completed", "timestamp": utc_now(), "approach": approach,
                "stage": stage_name, "candidate_id": candidate_id, "aggregate": row["aggregate"],
                "outer_heldout_real_evaluated": False,
            })
            stage_rows.append(row)
        ranked = sorted(stage_rows, key=_rank_key, reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        atomic_json(sweep_root / f"{stage_name}_ranking.json", ranked)
        history[stage_name] = ranked
        survivor_ids = {row["candidate_id"] for row in ranked[:budget["survivors"]]}
        survivors = [candidate for candidate in survivors if _candidate_id(candidate) in survivor_ids]
    winner = history["full"][0]
    finalist_class = ECConfig if approach == "ec" else PC2Config
    finalist_payload = winner["base_config"].copy()
    finalist_payload.update({"epochs": 8 if approach == "ec" else 9, "steps_per_epoch": 48 if approach == "ec" else 56})
    finalist = finalist_class(**finalist_payload)
    frozen = {
        "schema_version": 1, "approach": approach, "candidate_id": winner["candidate_id"],
        "config": asdict(finalist), "selection": "nested_group_safe_successive_halving_inner_only",
        "selection_protocol": "two_pseudo_unseen_classes_v2",
        "selector_implementation": selector_version,
        "selection_metrics": winner["aggregate"], "outer_heldout_real_evaluated": False,
    }
    atomic_json(study_root / "configs" / f"{approach}_frozen.json", frozen)
    append_jsonl(study_root / "experiment_registry.jsonl", {
        "event": "finalist_frozen", "timestamp": utc_now(), "approach": approach,
        "candidate_id": winner["candidate_id"], "selection_metrics": winner["aggregate"],
        "selector_implementation": selector_version, "outer_heldout_real_evaluated": False,
    })
    return frozen


def generate_sgme_candidates(count: int = 36) -> list[SGMEConfig]:
    rng = np.random.default_rng(3003)
    values: list[SGMEConfig] = []
    while len(values) < count:
        config = SGMEConfig(
            k_neighbors=int(rng.choice([5, 8, 12, 16, 24])),
            graph_iterations=int(rng.choice([6, 10, 16, 24])),
            graph_temperature=float(rng.choice([0.06, 0.10, 0.16, 0.24])),
            propagation_alpha=float(rng.choice([0.65, 0.75, 0.85, 0.92])),
            confidence_threshold=float(rng.choice([0.55, 0.70, 0.82, 0.90])),
            agreement_threshold=float(rng.choice([0.50, 0.65, 0.80])),
            augmentation_threshold=float(rng.choice([0.55, 0.70, 0.82])),
            semantic_threshold=float(rng.choice([0.40, 0.55, 0.70])),
            seen_rejection_threshold=float(rng.choice([-0.10, 0.0, 0.10, 0.20])),
            covariance=bool(rng.choice([False, True])),
            covariance_shrinkage=float(rng.choice([0.1, 0.25, 0.5])),
            abstention_quantile=float(rng.choice([0.0, 0.02, 0.05, 0.10])),
            seed=42,
        )
        if config not in values:
            values.append(config)
    return values


def _load_frozen_ec(study_root: Path) -> ECConfig:
    path = study_root / "configs" / "ec_frozen.json"
    if not path.exists():
        raise FileNotFoundError("EC finalist must be frozen before SGME selection.")
    return ECConfig(**json.loads(path.read_text(encoding="utf-8"))["config"])


def _prepare_sgme_tasks(
    frame: pd.DataFrame,
    study_root: Path,
    recipe_path: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    ec_config = _load_frozen_ec(study_root)
    recipes = load_event_recipes(recipe_path)
    bundles: list[dict[str, Any]] = []
    for pilot_index, pair in enumerate(PILOT_PAIRS):
        fold = build_event_openworld_fold(frame, holdout=pair, seed=42)
        seen_faults = sorted(set(range(1, 8)) - set(pair))
        pseudo_pairs = [(seen_faults[0], seen_faults[1]), (seen_faults[2], seen_faults[3]),
                        (seen_faults[4], seen_faults[1])]
        for inner_index, pseudo_pair in enumerate(pseudo_pairs):
            task = _inner_sgme_task(fold.train, fold.validation, pseudo_pair)
            scaler = StandardScaler().fit(task["train"][INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            tensors = {name: _transform(value, scaler) for name, value in task.items()}
            config = replace(ec_config, seed=42 + 100 * pilot_index + inner_index, epochs=6, steps_per_epoch=24)
            model, _ = train_ec_czsl(tensors["train"][0], tensors["train"][1], recipes["means"], recipes["stds"],
                                     device=device, config=config)
            known_class_ids = sorted(int(value) for value in tensors["train"][1].unique())
            outputs = {name: infer_event_model(
                           model, value[0], recipes["means"], recipes["stds"], device=device,
                           known_class_ids=known_class_ids,
                       )
                       for name, value in tensors.items()}
            generator = torch.Generator().manual_seed(880_000 + pilot_index * 100 + inner_index)
            noisy = tensors["adaptation"][0] + torch.randn(tensors["adaptation"][0].shape, generator=generator) * 0.015
            augmented = infer_event_model(
                model, noisy, recipes["means"], recipes["stds"], device=device,
                known_class_ids=known_class_ids,
            )
            bundles.append({
                "pilot_pair": pair, "inner_fold": inner_index, "pseudo_classes": pseudo_pair,
                "task": task, "tensors": tensors, "outputs": outputs,
                "augmentation_probabilities": augmented["logits"].softmax(-1),
            })
            torch.cuda.empty_cache()
    return bundles


def _evaluate_sgme_config(bundle: dict[str, Any], config: SGMEConfig, buffer_size: int, device: torch.device) -> dict[str, float]:
    tensors, outputs = bundle["tensors"], bundle["outputs"]
    pseudo_classes = tuple(int(value) for value in bundle["pseudo_classes"])
    train_y = tensors["train"][1]
    anchors = []
    for class_id in sorted(int(value) for value in train_y.unique()):
        candidates = torch.nonzero(train_y == class_id, as_tuple=False).flatten()
        anchors.append(candidates[:32])
    anchor_idx = torch.cat(anchors)
    reference_idx = torch.cat([torch.nonzero(tensors["support"][1] == class_id, as_tuple=False).flatten()[:1]
                               for class_id in pseudo_classes])
    reference_embedding = outputs["support"]["embedding"][reference_idx]
    reference_label = tensors["support"][1][reference_idx]
    adaptation_idx = np.concatenate([
        np.flatnonzero(tensors["adaptation"][1].numpy() == class_id)[:buffer_size] for class_id in pseudo_classes
    ]).astype(np.int64)
    available = {class_id: int((tensors["adaptation"][1] == class_id).sum()) for class_id in pseudo_classes}
    if any(count < buffer_size for count in available.values()):
        raise ValueError(f"SGME inner buffer requested {buffer_size} per class, available={available}")
    result = seeded_graph_enrollment(
        seen_anchor_embeddings=outputs["train"]["embedding"][anchor_idx],
        seen_anchor_labels=train_y[anchor_idx],
        reference_embeddings=reference_embedding,
        reference_labels=reference_label,
        adaptation_embeddings=outputs["adaptation"]["embedding"][adaptation_idx],
        semantic_probabilities=outputs["adaptation"]["logits"].softmax(-1)[adaptation_idx],
        augmentation_probabilities=bundle["augmentation_probabilities"][adaptation_idx],
        holdout=pseudo_classes, device=device, config=config,
    )
    post_embeddings = torch.cat([outputs["seen_test"]["embedding"], outputs["pseudo_query"]["embedding"]])
    post_y = torch.cat([tensors["seen_test"][1], tensors["pseudo_query"][1]])
    prototypes, variances = result.prototypes.to(device), result.variances.to(device)
    _, calibration_confidence = prototype_predict(outputs["calibration"]["embedding"], prototypes, variances,
                                                   covariance=config.covariance)
    abstention = None if config.abstention_quantile == 0 else float(np.quantile(
        calibration_confidence.numpy(), config.abstention_quantile
    ))
    predicted, _ = prototype_predict(post_embeddings, prototypes, variances, covariance=config.covariance,
                                     abstention_threshold=abstention)
    metrics = post_enrollment_metrics(post_y.numpy(), predicted.numpy(),
                                      seen_ids=sorted(int(value) for value in train_y.unique()), unseen_ids=pseudo_classes)
    covered = predicted != -1
    return {
        "post_h": float(metrics["harmonic_mean"]),
        "seen_accuracy": float(metrics["seen_accuracy"]),
        "unseen_accuracy": float(metrics["unseen_accuracy"]),
        "coverage": float(covered.float().mean()),
        "selective_risk": float((predicted[covered] != post_y[covered]).float().mean()) if covered.any() else 1.0,
        "accepted": float(result.metadata["accepted_count"]),
    }


def run_sgme_sweep(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    recipe_path: Path,
    device: torch.device,
    resume: bool = True,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    frame = attach_input_groups(frame) if "_input_group" not in frame else frame
    candidates = generate_sgme_candidates()
    bundles = _prepare_sgme_tasks(frame, study_root, recipe_path, device)
    root = study_root / "sweeps" / "sgme"
    root.mkdir(parents=True, exist_ok=True)
    stages = (("short", 32, 12), ("intermediate", 128, 4), ("full", 512, 1))
    survivors = candidates
    history = {}
    for stage_name, buffer_size, keep in stages:
        rows = []
        for config in survivors:
            candidate_id = config_hash(asdict(config))
            path = root / stage_name / f"{candidate_id}.json"
            if resume and path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("candidate_id") == candidate_id and row.get("stage") == stage_name and len(row.get("tasks", [])) == 9:
                    rows.append(row)
                    continue
            tasks = [_evaluate_sgme_config(bundle, config, buffer_size, device) for bundle in bundles]
            aggregate = {
                "post_h": float(np.mean([row["post_h"] for row in tasks])),
                "worst_unseen_accuracy": float(min(row["unseen_accuracy"] for row in tasks)),
                "seen_accuracy": float(np.mean([row["seen_accuracy"] for row in tasks])),
                "coverage": float(np.mean([row["coverage"] for row in tasks])),
                "selective_risk": float(np.mean([row["selective_risk"] for row in tasks])),
                "accepted": float(np.mean([row["accepted"] for row in tasks])),
            }
            row = {"schema_version": 1, "approach": "sgme", "stage": stage_name, "buffer_size": buffer_size,
                   "candidate_id": candidate_id, "config": asdict(config), "tasks": tasks, "aggregate": aggregate,
                   "outer_heldout_real_evaluated": False}
            atomic_json(path, row)
            append_jsonl(study_root / "experiment_registry.jsonl", {
                "event": "sweep_candidate_completed", "timestamp": utc_now(), "approach": "sgme",
                "stage": stage_name, "candidate_id": candidate_id, "aggregate": aggregate,
                "outer_heldout_real_evaluated": False,
            })
            rows.append(row)
        ranked = sorted(rows, key=lambda row: (
            row["aggregate"]["post_h"], row["aggregate"]["worst_unseen_accuracy"],
            row["aggregate"]["coverage"], -row["aggregate"]["selective_risk"]
        ), reverse=True)
        atomic_json(root / f"{stage_name}_ranking.json", ranked)
        history[stage_name] = ranked
        survivor_ids = {row["candidate_id"] for row in ranked[:keep]}
        survivors = [config for config in survivors if config_hash(asdict(config)) in survivor_ids]
    winner = history["full"][0]
    frozen = {"schema_version": 1, "approach": "sgme", "candidate_id": winner["candidate_id"],
              "config": winner["config"], "selection": "nested_group_safe_successive_halving_inner_only",
              "selection_metrics": winner["aggregate"], "outer_heldout_real_evaluated": False}
    atomic_json(study_root / "configs" / "sgme_frozen.json", frozen)
    append_jsonl(study_root / "experiment_registry.jsonl", {
        "event": "finalist_frozen", "timestamp": utc_now(), "approach": "sgme",
        "candidate_id": winner["candidate_id"], "selection_metrics": winner["aggregate"],
        "outer_heldout_real_evaluated": False,
    })
    return frozen
