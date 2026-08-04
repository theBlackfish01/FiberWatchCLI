from __future__ import annotations

"""Representative-pair KPSC physics-OE and score-fusion ablations."""

from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from .lifecycle_data import (
    deterministic_support_indices,
    fit_lifecycle_fold,
    split_known_calibration,
    transform_lifecycle,
)
from .lifecycle_enrollment import (
    EnrollmentSession,
    ProjectionAdapterConfig,
    projection_adapter_predict,
)
from .lifecycle_metrics import hard_prediction_metrics, open_world_ranking_metrics
from .lifecycle_experiment import _git_metadata
from .lifecycle_physics import PhysicsOEConfig, event_grammar_residual, finetune_physics_oe
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
from .model_functions.event_openworld import load_event_recipes
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


def _prepare_model_evaluation(
    model,
    *,
    tensor_fold,
    selector_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    recipes: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    frames = {
        "train": tensor_fold.split.train,
        "selector": selector_frame,
        "calibration": calibration_frame,
        "seen_test": tensor_fold.split.seen_test,
        "reference_pool": tensor_fold.split.reference_pool,
        "query": tensor_fold.split.query,
    }
    batches = {name: transform_lifecycle(frame, tensor_fold.scaler) for name, frame in frames.items()}
    outputs = {name: infer_lifecycle_model(model, batch, device=device) for name, batch in batches.items()}
    train_embedding = outputs["train"]["embedding"].numpy()
    train_labels = batches["train"].labels.numpy()
    reference = DistanceReference.fit(train_embedding, train_labels)
    known_ids = tuple(sorted(set(range(8)) - set(tensor_fold.split.holdout)))
    return {
        "batches": batches,
        "outputs": outputs,
        "train_embedding": train_embedding,
        "train_labels": train_labels,
        "reference": reference,
        "known_ids": known_ids,
        "recipes": recipes,
        "component_cache": {},
    }


def _evaluate_model(
    model,
    *,
    tensor_fold,
    selector_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    recipes: dict[str, Any],
    device: torch.device,
    prototype_count: int,
    fusion: str,
    calibration_mode: str,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = prepared or _prepare_model_evaluation(
        model,
        tensor_fold=tensor_fold,
        selector_frame=selector_frame,
        calibration_frame=calibration_frame,
        recipes=recipes,
        device=device,
    )
    batches = prepared["batches"]
    outputs = prepared["outputs"]
    cache = prepared["component_cache"]
    if prototype_count not in cache:
        bank = PrototypeBank.fit(
            prepared["train_embedding"],
            prepared["train_labels"],
            prototypes_per_class=prototype_count,
            seed=tensor_fold.split.seed,
        )

        def components(name: str) -> tuple[tuple[str, ...], np.ndarray]:
            residual = event_grammar_residual(
                batches[name].trace,
                batches[name].context,
                prepared["recipes"]["means"],
                prepared["recipes"]["stds"],
                known_class_ids=prepared["known_ids"],
            )
            return assemble_components(
                logits=outputs[name]["logits"].numpy(),
                embeddings=outputs[name]["embedding"].numpy(),
                distance_reference=prepared["reference"],
                prototype_bank=bank,
                physics_residual=residual,
            )

        names, selector_components = components("selector")
        normalizer = EmpiricalCDFNormalizer.fit(selector_components, names)
        _, calibration_components = components("calibration")
        _, seen_components = components("seen_test")
        _, query_components = components("query")
        cache[prototype_count] = {
            "names": names,
            "calibration": normalizer.transform(calibration_components),
            "outer": normalizer.transform(np.vstack((seen_components, query_components))),
        }
    component_payload = cache[prototype_count]
    names = component_payload["names"]
    calibration_score = fuse_scores(component_payload["calibration"], method=fusion)
    threshold = fit_joint_threshold(
        calibration_score, batches["calibration"].labels.numpy(), mode=calibration_mode
    )
    score = fuse_scores(component_payload["outer"], method=fusion)
    labels = np.r_[batches["seen_test"].labels.numpy(), batches["query"].labels.numpy()]
    predicted = np.r_[
        outputs["seen_test"]["logits"].argmax(1).numpy(),
        outputs["query"]["logits"].argmax(1).numpy(),
    ]
    operating = evaluate_joint_operating_point(
        score, labels, predicted, holdout=tensor_fold.split.holdout,
        calibration=threshold,
    )
    ranking = open_world_ranking_metrics(
        score, labels, predicted, holdout=tensor_fold.split.holdout
    )
    return {
        **operating, **ranking,
        "known_balanced_accuracy": float(balanced_accuracy_score(
            batches["seen_test"].labels.numpy(),
            outputs["seen_test"]["logits"].argmax(1).numpy(),
        )),
        "prototype_count": prototype_count,
        "fusion": fusion,
        "calibration_mode": calibration_mode,
        "component_names": list(names),
    }


def run_kpsc_ablation(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    model_config: LifecycleModelConfig,
    training_config: LifecycleTrainingConfig,
    device: torch.device | str,
    pairs: tuple[tuple[int, int], ...] = ((1, 2), (3, 5), (6, 7)),
) -> dict[str, Any]:
    device = require_cuda(str(device))
    root = study_root / "ablations"
    valid, _ = validate_run(
        root,
        expected={"run_id": "lifecycle-kpsc-ablation-v1"},
    )
    if valid:
        return json.loads(
            (root / "kpsc_ablation.json").read_text(encoding="utf-8")
        )
    environment = environment_metadata(device)
    dataset_path = Path(__file__).resolve().parent / "data" / "OTDR_DATA.csv"
    provenance = {
        "dataset_sha256": file_sha256(dataset_path),
        "source": _git_metadata(Path(__file__).resolve().parents[3]),
        "environment": environment,
    }
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "experiments" / "otdr_event_openworld_study" / "configs" / "event_recipes.json"
    )
    recipes = load_event_recipes(recipe_path)
    rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    training_rows: dict[str, Any] = {}
    root.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        unit_root = root / f"pair_{pair[0]}_{pair[1]}"
        unit_run_id = f"lifecycle-kpsc-ablation-{pair[0]}_{pair[1]}"
        unit_valid, _ = validate_run(
            unit_root,
            expected={"run_id": unit_run_id},
        )
        if unit_valid:
            unit = json.loads(
                (unit_root / "metrics.json").read_text(encoding="utf-8")
            )
            rows.extend(unit["rows"])
            projection_rows.extend(unit.get("cfe_projection_adapter", []))
            training_rows.update(unit["training"])
            continue
        unit_root.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            study_root / "experiment_registry.jsonl",
            {
                "event": "started",
                "run_id": unit_run_id,
                "stage": "kpsc_cfe_ablation",
                "timestamp": utc_now(),
                "device": str(device),
            },
        )
        row_start = len(rows)
        projection_start = len(projection_rows)
        training_keys_before = set(training_rows)
        tensor_fold = fit_lifecycle_fold(frame, holdout=pair, seed=42, regime="full")
        selector_frame, calibration_frame = split_known_calibration(
            tensor_fold.split.validation, seed=42
        )
        base, metadata = train_lifecycle_model(
            tensor_fold.batches["train"],
            transform_lifecycle(selector_frame, tensor_fold.scaler),
            device=device, model_config=model_config,
            training_config=replace(training_config, seed=42),
        )
        training_rows[f"base_{pair}"] = metadata
        models = {"no_outlier_exposure": base}
        for mode in ("generic", "pc2_physics", "diverse_physics", "diverse_anchor"):
            model = copy.deepcopy(base)
            model, oe_metadata = finetune_physics_oe(
                model, tensor_fold.batches["train"], device=device,
                config=PhysicsOEConfig(mode=mode, seed=42),
            )
            models[mode] = model
            training_rows[f"{mode}_{pair}"] = oe_metadata
        prepared_models = {
            mode: _prepare_model_evaluation(
                model,
                tensor_fold=tensor_fold,
                selector_frame=selector_frame,
                calibration_frame=calibration_frame,
                recipes=recipes,
                device=device,
            )
            for mode, model in models.items()
        }
        for mode, model in models.items():
            # Primary finalist scoring for every OE arm.
            metrics = _evaluate_model(
                model, tensor_fold=tensor_fold,
                selector_frame=selector_frame, calibration_frame=calibration_frame,
                recipes=recipes, device=device, prototype_count=1,
                fusion="robust_regret", calibration_mode="conformal",
                prepared=prepared_models[mode],
            )
            rows.append({"pair": list(pair), "oe_mode": mode, "score_ablation": "finalist", **metrics})
            torch.save({
                "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                "model_config": asdict(model_config), "oe_mode": mode,
            }, unit_root / f"kpsc_{mode}_{pair[0]}_{pair[1]}.pt")
        # Score, prototype, and calibration ablations on the unmodified encoder.
        for prototypes in (1, 4):
            for fusion in ("confidence", "best_single", "weighted", "sirc", "meta_p", "robust_regret"):
                for calibration_mode in ("empirical", "conformal"):
                    if prototypes == 1 and fusion == "robust_regret" and calibration_mode == "conformal":
                        continue
                    metrics = _evaluate_model(
                        base, tensor_fold=tensor_fold,
                        selector_frame=selector_frame, calibration_frame=calibration_frame,
                        recipes=recipes, device=device, prototype_count=prototypes,
                        fusion=fusion, calibration_mode=calibration_mode,
                        prepared=prepared_models["no_outlier_exposure"],
                    )
                    rows.append({
                        "pair": list(pair), "oe_mode": "no_outlier_exposure",
                        "score_ablation": "component_fusion", **metrics,
                    })
        prepared = prepared_models["no_outlier_exposure"]
        train_embedding = prepared["outputs"]["train"]["embedding"].numpy()
        train_labels = prepared["batches"]["train"].labels.numpy()
        reference_embedding = prepared["outputs"]["reference_pool"][
            "embedding"
        ].numpy()
        reference_frame = tensor_fold.split.reference_pool
        query_embedding = np.vstack((
            prepared["outputs"]["seen_test"]["embedding"].numpy(),
            prepared["outputs"]["query"]["embedding"].numpy(),
        ))
        query_labels = np.r_[
            prepared["batches"]["seen_test"].labels.numpy(),
            prepared["batches"]["query"].labels.numpy(),
        ]
        base_ids = tuple(sorted(int(value) for value in np.unique(train_labels)))
        for shots in (1, 3, 5):
            for draw in range(5):
                selected = deterministic_support_indices(
                    reference_frame,
                    class_ids=pair,
                    shots=shots,
                    seed=42,
                    draw=draw,
                    namespace="projection-adapter-support",
                )
                positions = reference_frame.index.get_indexer(selected)
                support_labels = reference_frame.loc[
                    selected, "Class"
                ].to_numpy(dtype=int)
                support_embedding = reference_embedding[positions]
                baseline = EnrollmentSession.from_base(
                    train_embedding, train_labels, metric="cosine"
                )
                base_mask = np.isin(query_labels, base_ids)
                base_accuracy_before = float(
                    (
                        baseline.predict(query_embedding)[base_mask]
                        == query_labels[base_mask]
                    ).mean()
                )
                for class_id in pair:
                    mask = support_labels == class_id
                    baseline = baseline.enroll(
                        class_id,
                        support_embedding[mask],
                        method="mean",
                        support_group_ids=tuple(
                            reference_frame.loc[
                                selected[mask], "_input_group"
                            ].astype(str)
                        ),
                    )
                for method, prediction, adapter_metadata in (
                    (
                        "training_free_mean",
                        baseline.predict(query_embedding),
                        None,
                    ),
                    (
                        "projection_only_adapter",
                        *projection_adapter_predict(
                            train_embedding,
                            train_labels,
                            support_embedding,
                            support_labels,
                            query_embedding,
                            device=device,
                            config=ProjectionAdapterConfig(
                                seed=42 + 100 * shots + draw
                            ),
                        ),
                    ),
                ):
                    metrics = hard_prediction_metrics(
                        query_labels,
                        prediction,
                        base_class_ids=base_ids,
                        enrolled_class_ids=pair,
                    )
                    normal = query_labels == 0
                    projection_rows.append({
                        "pair": list(pair),
                        "shots": shots,
                        "draw": draw,
                        "method": method,
                        "normal_far_after_enrollment": float(
                            np.isin(prediction[normal], pair).mean()
                        ),
                        "base_accuracy_before": base_accuracy_before,
                        "forgetting": max(
                            0.0,
                            base_accuracy_before - float(
                                metrics["base_accuracy"]
                            ),
                        ),
                        "backward_transfer": float(metrics["base_accuracy"])
                        - base_accuracy_before,
                        "retention_ratio": (
                            float(metrics["base_accuracy"])
                            / base_accuracy_before
                            if base_accuracy_before
                            else None
                        ),
                        "adapter": adapter_metadata,
                        **metrics,
                    })
        unit = {
            "schema_version": 1,
            "pair": list(pair),
            "rows": rows[row_start:],
            "cfe_projection_adapter": projection_rows[projection_start:],
            "training": {
                key: value
                for key, value in training_rows.items()
                if key not in training_keys_before
            },
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
                "stage": "kpsc_cfe_ablation",
                "timestamp": utc_now(),
                "device": str(device),
            },
        )
    result = {
        "schema_version": 1,
        "representative_pairs_only": True,
        "rows": rows,
        "cfe_projection_adapter": projection_rows,
        "training": training_rows,
        **provenance,
    }
    atomic_json(root / "kpsc_ablation.json", result)
    pd.DataFrame([
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        | {"pair": "-".join(str(value) for value in row["pair"])}
        for row in rows
    ]).to_csv(root / "kpsc_ablation.csv", index=False)
    write_manifest(root, {
        "run_id": "lifecycle-kpsc-ablation-v1",
        "completed": True, "device": str(device),
        "representative_pairs_only": True,
        **provenance,
    })
    return result
