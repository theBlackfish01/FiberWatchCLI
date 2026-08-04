from __future__ import annotations

"""CUDA TabPFN-v2 enrollment pilot with balanced, deterministic contexts."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .lifecycle_data import deterministic_support_indices, fit_lifecycle_fold
from .lifecycle_experiment import _git_metadata
from .lifecycle_metrics import hard_prediction_metrics
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


def _features(batch, regime: str) -> np.ndarray:
    if regime == "full":
        return np.c_[batch.trace.numpy(), batch.context.numpy()]
    if regime == "trace_only":
        return np.c_[batch.trace.numpy(), batch.context.numpy()[:, :1]]
    raise ValueError("TabPFN pilot supports full and trace_only.")


def _ranked_indices(labels: np.ndarray, groups: tuple[str, ...], class_id: int, count: int, namespace: str) -> np.ndarray:
    candidates = np.flatnonzero(labels == class_id)
    first_by_group: dict[str, int] = {}
    for index in candidates:
        first_by_group.setdefault(str(groups[index]), int(index))
    ranked = sorted(
        first_by_group.values(),
        key=lambda index: hashlib.sha256(f"{namespace}:{groups[index]}".encode()).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(
            f"Class {class_id} has {len(ranked)} unique groups; {count} required."
        )
    return np.asarray(ranked[:count], dtype=int)


def _balanced_query_indices(labels: np.ndarray, groups: tuple[str, ...], per_class: int = 100) -> np.ndarray:
    return np.concatenate([
        _ranked_indices(
            labels,
            groups,
            int(class_id),
            min(
                per_class,
                len({
                    str(groups[index])
                    for index in np.flatnonzero(labels == class_id)
                }),
            ),
            "tabpfn-query",
        )
        for class_id in sorted(np.unique(labels))
    ])


def _weight_files(model: object) -> list[Path]:
    paths: list[Path] = []
    for value in vars(model).values():
        if isinstance(value, Path) and value.is_file():
            paths.append(value)
        elif isinstance(value, str) and Path(value).is_file():
            paths.append(Path(value))
    return sorted(set(paths))


def run_tabpfn_pilot(
    *,
    frame: pd.DataFrame,
    study_root: Path,
    device: torch.device | str,
    pairs: tuple[tuple[int, int], ...] = ((1, 2), (3, 5), (6, 7)),
    regimes: tuple[str, ...] = ("full", "trace_only"),
    draws: int = 20,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    root = study_root / "baselines" / "tabpfn_v2"
    valid, _ = validate_run(
        root,
        expected={"run_id": "tabpfn-v2-representative-pilot"},
    )
    if valid:
        result = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        if int(result.get("requested_draws", -1)) == int(draws):
            return result
    root.mkdir(parents=True, exist_ok=True)
    environment = environment_metadata(device)
    dataset_path = Path(__file__).resolve().parent / "data" / "OTDR_DATA.csv"
    provenance = {
        "dataset_sha256": file_sha256(dataset_path),
        "source": _git_metadata(Path(__file__).resolve().parents[3]),
        "environment": environment,
    }
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.model_loading import resolve_model_path
    except ImportError as exc:
        raise RuntimeError("TabPFN-v2 is not installed in the declared isolated environment.") from exc
    rows = []
    resolved_model_path, _, resolved_model_name, _ = resolve_model_path(
        None, "classifier", "v2"
    )
    weight_paths: list[Path] = [resolved_model_path]
    started = time.perf_counter()
    for regime in regimes:
        for pair in pairs:
            unit_root = root / f"{regime}_{pair[0]}_{pair[1]}"
            unit_run_id = (
                f"tabpfn-v2-{regime}-{pair[0]}_{pair[1]}-draws{draws}"
            )
            unit_valid, _ = validate_run(
                unit_root,
                expected={
                    "run_id": unit_run_id,
                    "requested_draws": int(draws),
                },
            )
            if unit_valid:
                unit = json.loads(
                    (unit_root / "metrics.json").read_text(encoding="utf-8")
                )
                rows.extend(unit["rows"])
                weight_paths.extend(
                    Path(path) for path in unit.get("weight_hashes", {})
                )
                continue
            unit_started = time.perf_counter()
            unit_row_start = len(rows)
            append_jsonl(
                study_root / "experiment_registry.jsonl",
                {
                    "event": "started",
                    "run_id": unit_run_id,
                    "stage": "tabpfn_v2",
                    "timestamp": utc_now(),
                    "device": str(device),
                },
            )
            tensor_fold = fit_lifecycle_fold(frame, holdout=pair, seed=42, regime=regime)
            train_x = _features(tensor_fold.batches["train"], regime)
            train_y = tensor_fold.batches["train"].labels.numpy()
            train_groups = tensor_fold.batches["train"].group_ids
            reference_x = _features(tensor_fold.batches["reference_pool"], regime)
            reference_frame = tensor_fold.split.reference_pool
            query_x_all = np.vstack((
                _features(tensor_fold.batches["seen_test"], regime),
                _features(tensor_fold.batches["query"], regime),
            ))
            query_y_all = np.r_[
                tensor_fold.batches["seen_test"].labels.numpy(),
                tensor_fold.batches["query"].labels.numpy(),
            ]
            query_groups = (
                *tensor_fold.batches["seen_test"].group_ids,
                *tensor_fold.batches["query"].group_ids,
            )
            query_indices = _balanced_query_indices(query_y_all, query_groups)
            query_x, query_y = query_x_all[query_indices], query_y_all[query_indices]
            base_ids = tuple(sorted(int(value) for value in np.unique(train_y)))
            context_seeds = (42, 123, 2026)
            # Reuse the loaded CUDA models across all contexts for this pair.
            # Repeated fit() calls replace only the small task-specific cache.
            ensemble_models = [
                TabPFNClassifier(
                    n_estimators=1,
                    model_path=resolved_model_path,
                    device=str(device),
                    ignore_pretraining_limits=True,
                    fit_mode="fit_with_cache",
                    random_state=context_seed,
                    n_jobs=1,
                )
                for context_seed in context_seeds
            ]
            for shots in (1, 3, 5):
                for draw in range(draws):
                    selected = deterministic_support_indices(
                        reference_frame, class_ids=pair, shots=shots,
                        seed=42, draw=draw, namespace="tabpfn-support",
                    )
                    support_positions = reference_frame.index.get_indexer(selected)
                    support_x = reference_x[support_positions]
                    support_y = reference_frame.loc[selected, "Class"].to_numpy(dtype=int)
                    probabilities = []
                    inference_started = time.perf_counter()
                    for ensemble_index, (context_seed, model) in enumerate(
                        zip(context_seeds, ensemble_models, strict=True)
                    ):
                        base_indices = np.concatenate([
                            _ranked_indices(
                                train_y, train_groups, class_id, shots,
                                f"tabpfn-context:{context_seed}:{draw}:{shots}",
                            )
                            for class_id in base_ids
                        ])
                        context_x = np.vstack((train_x[base_indices], support_x))
                        context_y = np.r_[train_y[base_indices], support_y]
                        model.fit(context_x, context_y)
                        probability = model.predict_proba(query_x)
                        # Align any estimator-specific class ordering to local IDs 0..7.
                        aligned = np.zeros((len(query_x), 8), dtype=float)
                        aligned[:, np.asarray(model.classes_, dtype=int)] = probability
                        probabilities.append(aligned)
                        weight_paths.extend(_weight_files(model))
                    mean_probability = np.mean(probabilities, axis=0)
                    prediction = mean_probability.argmax(1)
                    metrics = hard_prediction_metrics(
                        query_y, prediction, base_class_ids=base_ids,
                        enrolled_class_ids=pair,
                    )
                    rows.append({
                        "regime": regime,
                        "pair": list(pair),
                        "shots": shots,
                        "draw": draw,
                        "context_examples_per_class": shots,
                        "context_examples": int(8 * shots),
                        "ensemble_contexts": 3,
                        "query_examples": len(query_y),
                        "device": str(device),
                        "inference_seconds": time.perf_counter() - inference_started,
                        **metrics,
                    })
            del ensemble_models
            torch.cuda.empty_cache()
            unit_weight_hashes = {
                str(path): file_sha256(path)
                for path in sorted(set(weight_paths))
                if path.is_file()
            }
            if (
                not resolved_model_path.is_file()
                or str(resolved_model_path) not in unit_weight_hashes
            ):
                raise RuntimeError(
                    "TabPFN unit completed inference without a hashable "
                    "resolved model checkpoint."
                )
            unit = {
                "schema_version": 1,
                "run_id": unit_run_id,
                "regime": regime,
                "pair": list(pair),
                "requested_draws": int(draws),
                "rows": rows[unit_row_start:],
                "weight_hashes": unit_weight_hashes,
                "duration_seconds": time.perf_counter() - unit_started,
                **provenance,
            }
            unit_root.mkdir(parents=True, exist_ok=True)
            atomic_json(unit_root / "metrics.json", unit)
            write_manifest(
                unit_root,
                {
                    "run_id": unit_run_id,
                    "completed": True,
                    "device": str(device),
                    "requested_draws": int(draws),
                    "regime": regime,
                    "pair": list(pair),
                    "weight_hashes": unit_weight_hashes,
                    **provenance,
                },
            )
            append_jsonl(
                study_root / "experiment_registry.jsonl",
                {
                    "event": "completed",
                    "run_id": unit_run_id,
                    "stage": "tabpfn_v2",
                    "timestamp": utc_now(),
                    "device": str(device),
                    "duration_seconds": unit["duration_seconds"],
                },
            )
    weight_hashes = {str(path): file_sha256(path) for path in sorted(set(weight_paths)) if path.is_file()}
    if not resolved_model_path.is_file() or str(resolved_model_path) not in weight_hashes:
        raise RuntimeError("TabPFN completed inference without a hashable resolved model checkpoint.")
    result = {
        "schema_version": 1,
        "status": "bounded_representative_pilot",
        "requested_draws": int(draws),
        "package_version": importlib.metadata.version("tabpfn"),
        "model": "TabPFNClassifier",
        "resolved_model_name": resolved_model_name,
        "resolved_model_path": str(resolved_model_path),
        "rows": rows,
        **provenance,
        "weight_hashes": weight_hashes,
        "cache_source": "local_tabpfn_or_huggingface_cache",
        "duration_seconds": time.perf_counter() - started,
        "balanced_context_policy": (
            "Each base and enrolled class contributes exactly the declared shot count; "
            "three deterministic base-context draws are ensembled."
        ),
    }
    atomic_json(root / "metrics.json", result)
    pd.DataFrame([
        {**row, "pair": "-".join(str(value) for value in row["pair"])}
        for row in rows
    ]).to_csv(root / "metrics.csv", index=False)
    write_manifest(root, {
        "run_id": "tabpfn-v2-representative-pilot",
        "completed": True,
        "device": str(device),
        "package_version": result["package_version"],
        **provenance,
    })
    return result
