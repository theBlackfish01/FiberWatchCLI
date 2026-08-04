from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .event_openworld_baselines import (
    OpenMaxEVT,
    deterministic_recipe_scores,
    run_outer_closed_encoder_open_set_baselines,
)
from .event_openworld_data import (
    attach_input_groups,
    build_event_openworld_fold,
    deterministic_group_sample,
    fit_tensor_fold,
    split_manifest,
    write_exact_group_manifest,
)
from .event_openworld_graph import class_prototypes, prototype_predict, seeded_graph_enrollment
from .event_openworld_metrics import ScoreNormalizer, evaluate_zero_day, semantic_metrics
from .event_openworld_training import (
    ECConfig,
    PC2Config,
    SGMEConfig,
    infer_event_model,
    train_ec_czsl,
    train_pc2_oe,
)
from .model_functions.event_openworld import PhysicsEventRenderer, load_event_recipes, robust_linear_detrend
from .model_functions.zero_shot import require_cuda
from .study_metrics import post_enrollment_metrics
from .study_state import atomic_json, file_sha256, write_manifest
from .zero_shot_data import INPUT_COLUMNS


def event_openworld_source_manifest() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    files = sorted(source_root.glob("event_openworld*.py"))
    files.append(source_root / "model_functions" / "event_openworld.py")
    hashes = {
        path.relative_to(source_root).as_posix(): file_sha256(path)
        for path in files
    }
    digest = hashlib.sha256("\n".join(f"{name}:{value}" for name, value in hashes.items()).encode()).hexdigest()
    runtime_names = {
        "event_openworld_baselines.py",
        "event_openworld_data.py",
        "event_openworld_experiment.py",
        "event_openworld_graph.py",
        "event_openworld_metrics.py",
        "event_openworld_training.py",
        "model_functions/event_openworld.py",
    }
    runtime_hashes = {name: value for name, value in hashes.items() if name in runtime_names}
    runtime_digest = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in runtime_hashes.items()).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "source_sha256": digest,
        "runtime_source_sha256": runtime_digest,
        "files": hashes,
        "runtime_files": sorted(runtime_names),
    }


def _score_bundle(
    outputs: dict[str, dict[str, torch.Tensor]],
    labels: dict[str, torch.Tensor],
    config: ECConfig | PC2Config,
) -> tuple[ScoreNormalizer, dict[str, np.ndarray]]:
    normal_mask = labels["validation"].numpy() == 0
    validation_components = outputs["validation"]["novelty_components"].numpy()
    normalizer = ScoreNormalizer.fit(validation_components[normal_mask], config.fusion_weights)
    scores = {name: normalizer.transform(output["novelty_components"].numpy()) for name, output in outputs.items()}
    return normalizer, scores


def _post_metrics(labels: torch.Tensor, predicted: torch.Tensor, holdout: tuple[int, int]) -> dict[str, object]:
    return post_enrollment_metrics(
        labels.numpy(), predicted.numpy(), seen_ids=sorted(set(range(8)) - set(holdout)), unseen_ids=holdout
    )


def _raw_seen_1nn(
    train_x: torch.Tensor, train_y: torch.Tensor, query_x: torch.Tensor, *, gallery_per_class: int = 128,
) -> dict[str, torch.Tensor]:
    indices = torch.cat([torch.nonzero(train_y == class_id, as_tuple=False).flatten()[:gallery_per_class]
                         for class_id in sorted(int(value) for value in train_y.unique())])
    gallery, gallery_y = train_x[indices].float(), train_y[indices]
    cosine_value, cosine_label, euclidean_value, euclidean_label = [], [], [], []
    normalized_gallery = F.normalize(gallery, dim=-1)
    for start in range(0, len(query_x), 4096):
        query = query_x[start:start + 4096].float()
        cosine = F.normalize(query, dim=-1) @ normalized_gallery.T
        value, nearest = cosine.max(1)
        cosine_value.append(value); cosine_label.append(gallery_y[nearest])
        euclidean = -torch.cdist(query, gallery)
        value, nearest = euclidean.max(1)
        euclidean_value.append(value); euclidean_label.append(gallery_y[nearest])
    return {"cosine_value": torch.cat(cosine_value), "cosine_label": torch.cat(cosine_label),
            "euclidean_value": torch.cat(euclidean_value), "euclidean_label": torch.cat(euclidean_label)}


def _raw_1nn_baselines(
    seen_nearest: dict[str, torch.Tensor],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    holdout: tuple[int, int],
) -> dict[str, object]:
    cosine_support = F.normalize(query_x.float(), dim=-1) @ F.normalize(support_x.float(), dim=-1).T
    cosine_value, cosine_index = cosine_support.max(1)
    cosine_pred = seen_nearest["cosine_label"].clone()
    replace_cosine = cosine_value > seen_nearest["cosine_value"]
    cosine_pred[replace_cosine] = support_y[cosine_index[replace_cosine]]
    euclidean_support = -torch.cdist(query_x.float(), support_x.float())
    euclidean_value, euclidean_index = euclidean_support.max(1)
    euclidean_pred = seen_nearest["euclidean_label"].clone()
    replace_euclidean = euclidean_value > seen_nearest["euclidean_value"]
    euclidean_pred[replace_euclidean] = support_y[euclidean_index[replace_euclidean]]
    return {
        "raw_cosine_1nn": _post_metrics(query_y, cosine_pred, holdout),
        "raw_euclidean_1nn": _post_metrics(query_y, euclidean_pred, holdout),
    }


def _select_rows(frame: pd.DataFrame, per_class: int, holdout: tuple[int, int], seed: int, draw: int, namespace: str) -> np.ndarray:
    local = frame.reset_index(drop=True)
    return deterministic_group_sample(local, class_ids=holdout, per_class=per_class, seed=seed, draw=draw, namespace=namespace)


def _evaluate_baseline_score(
    validation_score: np.ndarray,
    seen_score: np.ndarray,
    query_score: np.ndarray,
    *,
    tensor_fold: Any,
    labels: dict[str, torch.Tensor],
    predicted: np.ndarray,
    holdout: tuple[int, int],
    calibration: str,
) -> dict[str, object]:
    normal = labels["validation"].numpy() == 0
    true = torch.cat([labels["seen_test"], labels["query"]]).numpy()
    return evaluate_zero_day(
        validation_normal_score=validation_score[normal],
        validation_normal_snr=tensor_fold.tensors["validation"][0][:, 0].numpy()[normal],
        test_score=np.concatenate([seen_score, query_score]),
        test_snr=torch.cat([tensor_fold.tensors["seen_test"][0][:, 0], tensor_fold.tensors["query"][0][:, 0]]).numpy(),
        true_labels=true, predicted=predicted, holdout=holdout, calibration=calibration,
    )


def _anchor_indices(labels: torch.Tensor, *, per_class: int = 32) -> torch.Tensor:
    rows = []
    for class_id in sorted(int(value) for value in labels.unique()):
        indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        rows.append(indices[:per_class])
    return torch.cat(rows)


def _stratified_example_indices(labels: torch.Tensor, *, per_class: int) -> torch.Tensor:
    by_class = [torch.nonzero(labels == class_id, as_tuple=False).flatten()[:per_class]
                for class_id in sorted(int(value) for value in labels.unique())]
    rows = [values[offset] for offset in range(per_class) for values in by_class if offset < len(values)]
    return torch.stack(rows) if rows else torch.empty(0, dtype=torch.long)


def run_event_openworld_fold(
    *,
    approach: str,
    frame: pd.DataFrame,
    data_path: Path,
    run_dir: Path,
    holdout: tuple[int, int],
    seed: int,
    config: ECConfig | PC2Config,
    sgme_config: SGMEConfig,
    device: torch.device,
    recipe_path: Path,
    support_draws: int = 20,
    shots: tuple[int, ...] = (1, 3, 5),
    adaptation_buffers: tuple[int, ...] = (0, 32, 128, 512),
    run_sgme: bool = True,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    if approach not in {"ec", "pc2"}:
        raise ValueError("approach must be ec or pc2")
    run_dir.mkdir(parents=True, exist_ok=True)
    source = event_openworld_source_manifest()
    fold = build_event_openworld_fold(frame, holdout=holdout, seed=seed)
    tensor_fold = fit_tensor_fold(fold)
    study_root = recipe_path.parents[1]
    exact_path = write_exact_group_manifest(
        fold, study_root / "manifests" / "splits" / f"pair_{holdout[0]}_{holdout[1]}_seed_{seed}.npz"
    )
    split_payload = split_manifest(fold, data_path=data_path)
    split_payload["dataset_sha256"] = file_sha256(data_path)
    split_payload["exact_group_manifest"] = str(exact_path.relative_to(study_root).as_posix())
    split_payload["exact_group_manifest_sha256"] = file_sha256(exact_path)
    atomic_json(run_dir / "split_manifest.json", split_payload)
    atomic_json(run_dir / "config.json", {
        "approach": approach,
        "model": asdict(config),
        "sgme": asdict(sgme_config),
        "source_sha256": source["source_sha256"],
        "runtime_source_sha256": source["runtime_source_sha256"],
    })
    recipes = load_event_recipes(recipe_path)
    means, stds = recipes["means"], recipes["stds"]
    train_x, train_y = tensor_fold.tensors["train"]
    seen_ids = sorted(int(value) for value in train_y.unique())
    if approach == "ec":
        model, training = train_ec_czsl(train_x, train_y, means, stds, device=device, config=config)
    else:
        model, training = train_pc2_oe(
            train_x, train_y, means, stds,
            snr_mean=float(tensor_fold.scaler.mean_[0]), snr_scale=float(tensor_fold.scaler.scale_[0]),
            device=device, config=config,
        )
    checkpoint = {
        "approach": approach,
        "holdout": list(holdout),
        "seed": seed,
        "config": asdict(config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "recipe_sha256": file_sha256(recipe_path),
        "dataset_sha256": file_sha256(data_path),
        "cuda_device": training["cuda_device"],
        "source_sha256": source["source_sha256"],
        "runtime_source_sha256": source["runtime_source_sha256"],
    }
    torch.save(checkpoint, run_dir / "model.pt")
    atomic_json(run_dir / "training.json", training)
    infer_started = time.perf_counter()
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    labels: dict[str, torch.Tensor] = {}
    for name in ("train", "validation", "seen_test", "reference_pool", "adaptation_pool", "query"):
        x, y = tensor_fold.tensors[name]
        outputs[name] = infer_event_model(model, x, means, stds, device=device, known_class_ids=seen_ids)
        labels[name] = y
    inference_seconds = time.perf_counter() - infer_started
    normalizer, scores = _score_bundle(outputs, labels, config)
    test_labels = torch.cat([labels["seen_test"], labels["query"]]).numpy()
    test_logits = torch.cat([outputs["seen_test"]["logits"], outputs["query"]["logits"]]).numpy()
    operational_predicted = np.asarray(seen_ids)[test_logits[:, seen_ids].argmax(1)]
    test_score = np.concatenate([scores["seen_test"], scores["query"]])
    test_snr = torch.cat([tensor_fold.tensors["seen_test"][0][:, 0], tensor_fold.tensors["query"][0][:, 0]]).numpy()
    validation_normal = labels["validation"].numpy() == 0
    zero_day = evaluate_zero_day(
        validation_normal_score=scores["validation"][validation_normal],
        validation_normal_snr=tensor_fold.tensors["validation"][0][:, 0].numpy()[validation_normal],
        test_score=test_score, test_snr=test_snr, true_labels=test_labels, predicted=operational_predicted,
        holdout=holdout, calibration=config.calibration,
    )
    semantic = semantic_metrics(test_logits, test_labels, holdout)
    energy_baseline = _evaluate_baseline_score(
        outputs["validation"]["novelty_components"][:, 0].numpy(),
        outputs["seen_test"]["novelty_components"][:, 0].numpy(),
        outputs["query"]["novelty_components"][:, 0].numpy(),
        tensor_fold=tensor_fold, labels=labels, predicted=operational_predicted, holdout=holdout, calibration=config.calibration,
    )
    distance_baseline = _evaluate_baseline_score(
        outputs["validation"]["novelty_components"][:, 1].numpy(),
        outputs["seen_test"]["novelty_components"][:, 1].numpy(),
        outputs["query"]["novelty_components"][:, 1].numpy(),
        tensor_fold=tensor_fold, labels=labels, predicted=operational_predicted, holdout=holdout, calibration=config.calibration,
    )
    openmax = OpenMaxEVT.fit(outputs["train"]["embedding"], labels["train"], seen_ids)
    def openmax_score(name: str) -> np.ndarray:
        return openmax.novelty(outputs[name]["embedding"])
    openmax_baseline = _evaluate_baseline_score(
        openmax_score("validation"), openmax_score("seen_test"), openmax_score("query"),
        tensor_fold=tensor_fold, labels=labels, predicted=operational_predicted, holdout=holdout, calibration=config.calibration,
    )
    deterministic_outputs = {}
    for name in ("validation", "seen_test", "query"):
        deterministic_outputs[name] = deterministic_recipe_scores(
            tensor_fold.tensors[name][0], means, stds, seen_ids
        )
    deterministic_logits = torch.cat([deterministic_outputs["seen_test"][0], deterministic_outputs["query"][0]]).numpy()
    deterministic_operational_predicted = np.asarray(seen_ids)[deterministic_logits[:, seen_ids].argmax(1)]
    deterministic_baseline = {
        "zero_day": _evaluate_baseline_score(
            deterministic_outputs["validation"][1].numpy(), deterministic_outputs["seen_test"][1].numpy(),
            deterministic_outputs["query"][1].numpy(), tensor_fold=tensor_fold, labels=labels,
            predicted=deterministic_operational_predicted, holdout=holdout, calibration=config.calibration,
        ),
        "semantic": semantic_metrics(deterministic_logits, test_labels, holdout),
    }
    closed_encoder_baselines = run_outer_closed_encoder_open_set_baselines(
        fold=fold,
        tensor_fold=tensor_fold,
        holdout=holdout,
        seed=seed,
        study_root=study_root,
        device=device,
    )
    with torch.no_grad():
        train_detrended, _ = robust_linear_detrend(train_x[:, 1:].float())
        renderer_trace_rms = float(train_detrended.square().mean(1).sqrt().median().clamp_min(1e-3))
    renderer = PhysicsEventRenderer(
        means.to(device), stds.to(device),
        snr_mean=float(tensor_fold.scaler.mean_[0]),
        snr_scale=float(tensor_fold.scaler.scale_[0]),
        trace_rms_target=renderer_trace_rms,
    )
    boundary_x, boundary_factors = renderer.render_boundary(512, generator=torch.Generator(device=device).manual_seed(seed + 6_700_001))
    boundary_output = infer_event_model(
        model, boundary_x.cpu(), means, stds, device=device, known_class_ids=seen_ids
    )
    finalists_frozen = all(
        (study_root / "configs" / f"{name}_frozen.json").is_file()
        for name in ("ec", "pc2", "sgme")
    )
    posthoc_position = np.concatenate([
        fold.seen_test["Position"].to_numpy(dtype=np.float32, copy=True),
        fold.query["Position"].to_numpy(dtype=np.float32, copy=True),
    ]) if (finalists_frozen and "Position" in fold.seen_test and "Position" in fold.query) else np.full(
        len(test_labels), np.nan, dtype=np.float32
    )
    combined_feature_tensor = torch.cat([
        tensor_fold.tensors["seen_test"][0], tensor_fold.tensors["query"][0]
    ])
    combined_label_tensor = torch.cat([labels["seen_test"], labels["query"]])
    combined_embedding_tensor = torch.cat([
        outputs["seen_test"]["embedding"], outputs["query"]["embedding"]
    ])
    combined_center_tensor = torch.cat([outputs["seen_test"]["center"], outputs["query"]["center"]])
    feature_examples = _stratified_example_indices(combined_label_tensor, per_class=16)
    embedding_examples = _stratified_example_indices(combined_label_tensor, per_class=256)
    base_predictions = {
        "labels": test_labels.astype(np.int8),
        "logits": test_logits.astype(np.float16),
        "novelty_score": test_score.astype(np.float32),
        "snr": test_snr.astype(np.float32),
        "validation_labels": labels["validation"].numpy().astype(np.int8),
        "validation_novelty_score": scores["validation"].astype(np.float32),
        "validation_snr": tensor_fold.tensors["validation"][0][:, 0].numpy().astype(np.float32),
        "factor_mean": torch.cat([outputs["seen_test"]["factor_mean"], outputs["query"]["factor_mean"]]).numpy().astype(np.float16),
        "event_center": combined_center_tensor.numpy().astype(np.float16),
        "example_features": combined_feature_tensor[feature_examples].numpy().astype(np.float32),
        "example_labels": combined_label_tensor[feature_examples].numpy().astype(np.int8),
        "example_event_center": combined_center_tensor[feature_examples].numpy().astype(np.float16),
        "example_embedding": combined_embedding_tensor[embedding_examples].numpy().astype(np.float16),
        "example_embedding_labels": combined_label_tensor[embedding_examples].numpy().astype(np.int8),
        "boundary_features": boundary_x.cpu().numpy().astype(np.float32),
        "boundary_embedding": boundary_output["embedding"].numpy().astype(np.float16),
        "boundary_factors": boundary_factors.cpu().numpy().astype(np.float16),
        "posthoc_position": posthoc_position,
    }
    np.savez_compressed(run_dir / "predictions.npz", **base_predictions)

    train_embeddings = outputs["train"]["embedding"]
    seen_test_embeddings = outputs["seen_test"]["embedding"]
    reference_embeddings = outputs["reference_pool"]["embedding"]
    adaptation_embeddings = outputs["adaptation_pool"]["embedding"]
    query_embeddings = outputs["query"]["embedding"]
    combined_embeddings = torch.cat([seen_test_embeddings, query_embeddings])
    combined_labels = torch.cat([labels["seen_test"], labels["query"]])
    reference_frame = fold.reference_pool.reset_index(drop=True)
    adaptation_frame = fold.adaptation_pool.reset_index(drop=True)
    seen_prototypes, seen_variances = class_prototypes(train_embeddings, train_y)
    anchor_idx = _anchor_indices(train_y)
    anchor_embeddings, anchor_labels = train_embeddings[anchor_idx], train_y[anchor_idx]
    adaptation_x = tensor_fold.tensors["adaptation_pool"][0]
    augmentation_generator = torch.Generator().manual_seed(seed + 8_000_003)
    augmented_x = adaptation_x + torch.randn(adaptation_x.shape, generator=augmentation_generator) * 0.015
    augmentation_output = infer_event_model(
        model, augmented_x, means, stds, device=device, known_class_ids=seen_ids
    )
    adaptation_probabilities = outputs["adaptation_pool"]["logits"].softmax(-1)
    augmentation_probabilities = augmentation_output["logits"].softmax(-1)
    enrollment_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    inductive_prediction_settings: list[dict[str, object]] = []
    inductive_prediction_values: list[np.ndarray] = []
    graph_prediction_settings: list[dict[str, object]] = []
    graph_prediction_values: list[np.ndarray] = []
    enrollment_group_arrays: dict[str, np.ndarray] = {}
    enrollment_group_index: list[dict[str, object]] = []
    graph_seconds = 0.0
    enrollment_started = time.perf_counter()
    combined_raw_x = torch.cat([tensor_fold.tensors["seen_test"][0], tensor_fold.tensors["query"][0]])
    raw_seen_nearest = _raw_seen_1nn(train_x, train_y, combined_raw_x)
    for draw in range(support_draws):
        for shot in shots:
            support_idx = _select_rows(reference_frame, shot, holdout, seed, draw, "reference")
            support_key = f"support_d{draw}_k{shot}"
            enrollment_group_arrays[support_key] = np.asarray(
                [bytes.fromhex(value) for value in reference_frame.iloc[support_idx]["_input_group"]], dtype="V32"
            )
            enrollment_group_index.append({"draw": draw, "shots": shot, "support_key": support_key})
            support_embeddings = reference_embeddings[support_idx]
            support_labels = labels["reference_pool"][support_idx]
            enrolled_embeddings = torch.cat([train_embeddings, support_embeddings])
            enrolled_labels = torch.cat([train_y, support_labels])
            prototypes, variances = class_prototypes(enrolled_embeddings, enrolled_labels)
            prototype_pred, _ = prototype_predict(combined_embeddings, prototypes)
            row = {"draw": draw, "shots": shot, **_post_metrics(combined_labels, prototype_pred, holdout)}
            enrollment_rows.append(row)
            inductive_prediction_settings.append({"draw": draw, "shots": shot, "support_key": support_key})
            inductive_prediction_values.append(prototype_pred.numpy().astype(np.int8))
            raw = _raw_1nn_baselines(
                raw_seen_nearest, tensor_fold.tensors["reference_pool"][0][support_idx], support_labels,
                combined_raw_x, combined_labels, holdout,
            )
            raw_rows.append({"draw": draw, "shots": shot, **raw})
            if not run_sgme:
                continue
            for buffer_size in adaptation_buffers:
                if buffer_size:
                    adaptation_idx = _select_rows(adaptation_frame, buffer_size, holdout, seed, draw, "adaptation")
                else:
                    adaptation_idx = np.empty(0, dtype=np.int64)
                adaptation_key = f"adaptation_d{draw}_n{buffer_size}"
                if adaptation_key not in enrollment_group_arrays:
                    enrollment_group_arrays[adaptation_key] = np.asarray(
                        [bytes.fromhex(value) for value in adaptation_frame.iloc[adaptation_idx]["_input_group"]], dtype="V32"
                    )
                graph_started = time.perf_counter()
                result = seeded_graph_enrollment(
                    seen_anchor_embeddings=anchor_embeddings,
                    seen_anchor_labels=anchor_labels,
                    reference_embeddings=support_embeddings,
                    reference_labels=support_labels,
                    adaptation_embeddings=adaptation_embeddings[adaptation_idx],
                    semantic_probabilities=adaptation_probabilities[adaptation_idx],
                    augmentation_probabilities=augmentation_probabilities[adaptation_idx],
                    holdout=holdout, device=device, config=sgme_config,
                )
                graph_seconds += time.perf_counter() - graph_started
                result_prototypes = result.prototypes.to(device)
                result_variances = result.variances.to(device)
                _, validation_confidence = prototype_predict(
                    outputs["validation"]["embedding"], result_prototypes, result_variances,
                    covariance=sgme_config.covariance,
                )
                abstention = float(np.quantile(validation_confidence.numpy(), sgme_config.abstention_quantile))
                graph_pred, graph_conf = prototype_predict(
                    combined_embeddings, result_prototypes, result_variances,
                    covariance=sgme_config.covariance, abstention_threshold=abstention,
                )
                graph_rows.append({
                    "draw": draw, "shots": shot, "buffer_per_class": buffer_size,
                    **_post_metrics(combined_labels, graph_pred, holdout),
                    "selective_risk": float((graph_pred[graph_pred != -1] != combined_labels[graph_pred != -1]).float().mean()) if (graph_pred != -1).any() else 1.0,
                    "coverage": float((graph_pred != -1).float().mean()),
                    "graph": result.metadata,
                })
                graph_prediction_settings.append({"draw": draw, "shots": shot, "buffer_per_class": buffer_size,
                                                   "support_key": support_key, "adaptation_key": adaptation_key})
                graph_prediction_values.append(graph_pred.numpy().astype(np.int8))
    atomic_json(run_dir / "inductive_predictions_index.json", inductive_prediction_settings)
    atomic_json(run_dir / "enrollment_predictions_index.json", graph_prediction_settings)
    atomic_json(run_dir / "enrollment_group_index.json", enrollment_group_index)
    np.savez_compressed(run_dir / "enrollment_groups.npz", **enrollment_group_arrays)
    np.savez_compressed(
        run_dir / "enrollment_predictions.npz",
        labels=combined_labels.numpy().astype(np.int8),
        inductive_predicted=np.stack(inductive_prediction_values),
        sgme_predicted=(np.stack(graph_prediction_values) if graph_prediction_values
                        else np.empty((0, len(combined_labels)), dtype=np.int8)),
    )
    enrollment_seconds = time.perf_counter() - enrollment_started
    metrics = {
        "schema_version": 1,
        "approach": approach,
        "holdout": list(holdout),
        "seed": seed,
        "zero_day": zero_day,
        "semantic": semantic,
        "open_set_baselines": {
            "energy": energy_baseline,
            "recipe_distance": distance_baseline,
            "openmax_evt": openmax_baseline,
            "deterministic_physics": deterministic_baseline,
            "strongest_closed_set_encoder": closed_encoder_baselines,
            "previous_multi_similarity_frozen": {
                "source": "otdr_three_approach_study/FINAL_REPORT.md",
                "pre_auroc": 0.757, "unknown_recall_at_approximately_1pct_far": 0.128,
                "one_shot_h": 0.422,
            },
        },
        "inductive_enrollment": enrollment_rows,
        "sgme_enrollment": graph_rows,
        "raw_baselines": raw_rows,
        "score_normalizer": {"mean": normalizer.mean.tolist(), "scale": normalizer.scale.tolist(), "weights": normalizer.weights.tolist()},
        "efficiency": {"training_seconds": training["duration_seconds"], "inference_seconds": inference_seconds,
                       "inference_ms_per_trace": inference_seconds / max(sum(len(value) for value in labels.values()), 1) * 1000,
                       "enrollment_seconds": enrollment_seconds, "graph_seconds": graph_seconds,
                       "graph_update_ms_mean": graph_seconds / max(len(graph_rows), 1) * 1000,
                       "parameters": training["parameter_count"], "checkpoint_bytes": (run_dir / "model.pt").stat().st_size,
                       "prediction_bytes": (run_dir / "predictions.npz").stat().st_size,
                       "enrollment_prediction_bytes": (run_dir / "enrollment_predictions.npz").stat().st_size,
                       "enrollment_group_manifest_bytes": (run_dir / "enrollment_groups.npz").stat().st_size,
                       "peak_allocated_bytes": training["peak_allocated_bytes"]},
    }
    localization_mask = np.isfinite(posthoc_position) & (posthoc_position > 0) & (test_labels != 0)
    if localization_mask.any():
        expected_bin = np.clip(posthoc_position[localization_mask] * 100 - 1, 0, 29)
        predicted_bin = np.concatenate([outputs["seen_test"]["center"].numpy(), outputs["query"]["center"].numpy()])[localization_mask]
        correlation = (float(np.corrcoef(predicted_bin, expected_bin)[0, 1])
                       if np.std(predicted_bin) > 0 and np.std(expected_bin) > 0 else np.nan)
        metrics["posthoc_localization"] = {
            "policy": "Position inspected only after EC/PC2/SGME finalists were frozen; never used for training or selection.",
            "samples": int(localization_mask.sum()),
            "mae_bins": float(np.abs(predicted_bin - expected_bin).mean()),
            "median_absolute_error_bins": float(np.median(np.abs(predicted_bin - expected_bin))),
            "pearson_correlation": correlation if np.isfinite(correlation) else None,
        }
    atomic_json(run_dir / "metrics.json", metrics)
    manifest = write_manifest(run_dir, {
        "run_id": run_dir.name, "approach": approach, "holdout": list(holdout), "seed": seed,
        "cuda_verified": True, "cuda_device": training["cuda_device"],
        "source_sha256": source["source_sha256"],
        "runtime_source_sha256": source["runtime_source_sha256"],
    })
    return {"metrics": metrics, "manifest": manifest}
