from __future__ import annotations

"""Cached-base-memory TabPFN enrollment studies for OTDR faults.

The confirmatory TabPFN study builds a fresh balanced context with the same
number of examples for every class.  This post-confirmatory sensitivity instead
establishes a fixed memory for the six base classes and appends only support
examples from the two held-out faults. The same implementation also executes
the subsequently frozen multi-seed replication without changing the original
pilot artifacts.
"""

import argparse
import importlib.metadata
import itertools
import json
from pathlib import Path
import time
import traceback
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

from .lifecycle_analysis import (
    fault_aware_leave_one_out,
    hierarchical_bootstrap,
    holm_adjust,
    paired_sign_flip,
)
from .lifecycle_data import (
    deterministic_support_indices,
    fit_lifecycle_fold,
    lifecycle_split_manifest,
)
from .lifecycle_experiment import _git_metadata
from .lifecycle_metrics import hard_prediction_metrics
from .lifecycle_tabpfn import _balanced_query_indices, _ranked_indices
from .model_functions.zero_shot import require_cuda
from .study_state import (
    append_jsonl,
    atomic_json,
    environment_metadata,
    file_sha256,
    utc_now,
    validate_run,
    write_manifest,
)
from .tabpfn_full_analysis import reconstruct_metrics
from .tabpfn_full_study import (
    DATA_PATH,
    FROZEN_PROTOCOL_SHA256,
    N_CLASSES,
    REPOSITORY_ROOT,
    STUDY_ROOT,
    _aligned_probability,
    _features,
    _softmax,
    _weight_paths,
    load_protocol,
    metric_row,
)


EXPLORATORY_ROOT = STUDY_ROOT / "incremental_memory_pilot"
CONFIRMATORY_ROOT = STUDY_ROOT / "incremental_memory_confirmatory"
EXPLORATORY_CONFIG_SHA256 = "9ec2e53831c947e05aa35d45e233b1db4e93cf8e58b440e6ff18f65083bb4ee6"
CONFIRMATORY_CONFIG_SHA256 = "d6cd76a9a85d644455b66bd9972fa614fc54fbe64ddb3646a7df48dee7e14f6f"

# These aliases deliberately default to the original pilot so existing imports
# and reproduction commands remain backward compatible. The CLI activates the
# confirmatory root before any run or analysis function is called.
PILOT_ROOT = EXPLORATORY_ROOT
CONFIG_PATH = PILOT_ROOT / "config.json"
CONFIG_SHA256 = EXPLORATORY_CONFIG_SHA256
ACTIVE_STUDY = "pilot"
EVIDENCE_SCHEMA = 1
SVM_MAX_ITER = 1000


def activate_study(study: str) -> None:
    """Select one immutable study root for the current process."""
    global ACTIVE_STUDY, PILOT_ROOT, CONFIG_PATH, CONFIG_SHA256
    if study == "pilot":
        ACTIVE_STUDY = study
        PILOT_ROOT = EXPLORATORY_ROOT
        CONFIG_SHA256 = EXPLORATORY_CONFIG_SHA256
    elif study == "confirmatory":
        ACTIVE_STUDY = study
        PILOT_ROOT = CONFIRMATORY_ROOT
        CONFIG_SHA256 = CONFIRMATORY_CONFIG_SHA256
    else:
        raise ValueError(f"Unknown incremental-memory study: {study}")
    CONFIG_PATH = PILOT_ROOT / "config.json"


def load_config() -> dict[str, Any]:
    if file_sha256(CONFIG_PATH) != CONFIG_SHA256:
        raise RuntimeError("Incremental-memory study config hash mismatch.")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_type = (
        "exploratory_post_confirmatory"
        if ACTIVE_STUDY == "pilot"
        else "confirmatory_fixed_base_memory_replication"
    )
    if config.get("study_type") != expected_type:
        raise RuntimeError(f"Study type must be {expected_type!r}.")
    if config.get("device") != "cuda:0":
        raise RuntimeError("Incremental-memory config must require CUDA.")
    return config


def freeze_source_snapshot() -> dict[str, Any]:
    """Freeze execution inputs before opening a new study's outer results."""
    from tabpfn.model_loading import resolve_model_path

    existing_units = list((PILOT_ROOT / "units").glob("pair_*/seed_*"))
    snapshot_path = PILOT_ROOT / "SOURCE_SNAPSHOT.json"
    if existing_units and not snapshot_path.is_file():
        raise RuntimeError("Cannot create a retrospective execution snapshot.")
    checkpoint_path, _, model_name, _ = resolve_model_path(None, "classifier", "v2")
    source_paths = [
        Path(__file__),
        REPOSITORY_ROOT / "OTDR_CLI/OTDR/src/tabpfn_full_study.py",
        REPOSITORY_ROOT / "OTDR_CLI/OTDR/src/lifecycle_data.py",
        REPOSITORY_ROOT / "OTDR_CLI/OTDR/src/lifecycle_metrics.py",
        REPOSITORY_ROOT / "OTDR_CLI/OTDR/src/lifecycle_analysis.py",
        REPOSITORY_ROOT / "OTDR_CLI/OTDR/tests/test_tabpfn_incremental_memory.py",
    ]
    snapshot = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "study": ACTIVE_STUDY,
        "config_sha256": CONFIG_SHA256,
        "frozen_parent_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "dataset_path": str(DATA_PATH),
        "dataset_sha256": file_sha256(DATA_PATH),
        "tabpfn_package_version": importlib.metadata.version("tabpfn"),
        "tabpfn_model_name": model_name,
        "tabpfn_checkpoint_path": str(checkpoint_path),
        "tabpfn_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_file_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in source_paths
        },
        "git": _git_metadata(REPOSITORY_ROOT),
        "outer_results_opened": bool(existing_units),
    }
    if snapshot_path.is_file():
        existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "study",
            "config_sha256",
            "frozen_parent_protocol_sha256",
            "dataset_sha256",
            "tabpfn_checkpoint_sha256",
            "source_file_sha256",
        )
        if any(existing.get(key) != snapshot.get(key) for key in immutable_keys):
            raise RuntimeError("Execution source snapshot already exists and differs.")
        return existing
    atomic_json(snapshot_path, snapshot)
    return snapshot


def _unit_dir(pair: tuple[int, int], seed: int) -> Path:
    return PILOT_ROOT / "units" / f"pair_{pair[0]:02d}_{pair[1]:02d}" / f"seed_{seed}"


def _unit_id(pair: tuple[int, int], seed: int) -> str:
    return f"incremental-memory-pair-{pair[0]}-{pair[1]}-seed-{seed}"


def _expected_manifest(
    pair: tuple[int, int],
    seed: int,
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_config()
    return {
        "run_id": _unit_id(pair, seed),
        "evidence_schema": EVIDENCE_SCHEMA,
        "pilot_config_sha256": config_sha256 or CONFIG_SHA256,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "requested_draws": int(config["draws"]),
        "requested_shots": [int(value) for value in config["shots"]],
    }


def _append_registry(event: str, **payload: Any) -> None:
    append_jsonl(
        PILOT_ROOT / "experiment_registry.jsonl",
        {"event": event, "timestamp": utc_now(), **payload},
    )


def _heartbeat(run_id: str, stage: str, **payload: Any) -> None:
    """Append diagnostic progress without changing prediction artifacts."""
    append_jsonl(
        PILOT_ROOT / "heartbeats.jsonl",
        {
            "run_id": run_id,
            "stage": stage,
            "timestamp": utc_now(),
            **payload,
        },
    )


def _update_state(
    *,
    run_id: str,
    completed: bool,
    failure: dict[str, Any] | None = None,
) -> None:
    """Persist resumable unit state when the selected study has a state file."""
    path = PILOT_ROOT / "state.json"
    if not path.is_file():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    completed_units = set(state.get("completed_local_units", []))
    failed_units = {
        str(item.get("run_id")): item
        for item in state.get("failed_units", [])
        if isinstance(item, dict) and item.get("run_id")
    }
    if completed:
        completed_units.add(run_id)
        failed_units.pop(run_id, None)
    elif failure is not None:
        failed_units[run_id] = failure
    state["completed_local_units"] = sorted(completed_units)
    state["failed_units"] = list(failed_units.values())
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def _base_memory_indices(
    labels: np.ndarray,
    groups: tuple[str, ...],
    *,
    base_ids: tuple[int, ...],
    count: int,
    context_seed: int,
    namespace: str,
) -> np.ndarray:
    return np.concatenate(
        [
            _ranked_indices(
                labels,
                groups,
                class_id,
                count,
                f"{namespace}:{context_seed}",
            )
            for class_id in base_ids
        ]
    )


def _bounded_linear_svm_probability(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the secondary SVM control with a guard against libsvm nontermination."""
    model = SVC(
        C=1.0,
        kernel="linear",
        probability=False,
        decision_function_shape="ovr",
        max_iter=SVM_MAX_ITER,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(context_x, context_y)
    score = model.decision_function(query_x)
    probability = _aligned_probability(_softmax(score), model.classes_)
    return probability, {
        "solver": "sklearn.svm.SVC",
        "kernel": "linear",
        "max_iter": SVM_MAX_ITER,
        "fit_status": int(model.fit_status_),
        "converged": int(model.fit_status_) == 0,
        "n_iter_by_binary_problem": [int(value) for value in model.n_iter_],
        "maximum_binary_iterations": int(np.max(model.n_iter_)),
        "convergence_warning": any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }


def run_unit(
    *,
    frame: pd.DataFrame,
    pair: tuple[int, int],
    seed: int,
    device: str = "cuda:0",
) -> dict[str, Any]:
    config = load_config()
    protocol = load_protocol()
    pair = tuple(sorted(int(value) for value in pair))
    if list(pair) not in config["pairs"]:
        raise ValueError(f"Pair {pair} is outside the pilot config.")
    if int(seed) not in config["seeds"]:
        raise ValueError(f"Seed {seed} is outside the pilot config.")

    root = _unit_dir(pair, seed)
    expected = _expected_manifest(pair, seed)
    valid, _ = validate_run(root, expected=expected)
    if valid:
        _update_state(run_id=_unit_id(pair, seed), completed=True)
        return json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    cuda = require_cuda(device)
    if str(cuda) != config["device"]:
        raise RuntimeError(f"Configured device is {config['device']}, got {cuda}.")
    torch.cuda.synchronize(cuda)
    torch.cuda.reset_peak_memory_stats(cuda)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run_id = _unit_id(pair, seed)
    _append_registry(
        "started",
        run_id=run_id,
        pair=list(pair),
        seed=int(seed),
        device=str(cuda),
        pilot_config_sha256=CONFIG_SHA256,
    )

    try:
        fold = fit_lifecycle_fold(frame, holdout=pair, seed=seed, regime="summary_only")
        _heartbeat(
            run_id,
            "fold_ready",
            elapsed_seconds=time.perf_counter() - started,
        )
        batches = fold.batches
        train_x = _features(batches["train"], "summary_only")
        train_y = batches["train"].labels.numpy()
        train_groups = batches["train"].group_ids
        reference_x = _features(batches["reference_pool"], "summary_only")
        reference_frame = fold.split.reference_pool

        outer_x_all = np.vstack(
            (
                _features(batches["seen_test"], "summary_only"),
                _features(batches["query"], "summary_only"),
            )
        )
        outer_y_all = np.r_[
            batches["seen_test"].labels.numpy(),
            batches["query"].labels.numpy(),
        ]
        outer_groups = (*batches["seen_test"].group_ids, *batches["query"].group_ids)
        query_indices = _balanced_query_indices(
            outer_y_all,
            outer_groups,
            per_class=int(config["query_examples_per_class"]),
        )
        query_x = outer_x_all[query_indices]
        query_y = outer_y_all[query_indices]
        query_groups = tuple(outer_groups[index] for index in query_indices)
        if len(query_groups) != len(set(query_groups)):
            raise AssertionError("Query groups are not unique.")
        _heartbeat(
            run_id,
            "query_ready",
            query_examples=len(query_y),
            elapsed_seconds=time.perf_counter() - started,
        )

        base_ids = tuple(sorted(int(value) for value in np.unique(train_y)))
        if set(base_ids) & set(pair) or set(base_ids) | set(pair) != set(range(N_CLASSES)):
            raise AssertionError("Base and enrolled classes do not partition the label space.")

        context_seeds = tuple(int(value) for value in config["context_seeds"])
        base_count = int(config["base_context_per_class"])
        base_memories: list[dict[str, Any]] = []
        context_manifest: list[dict[str, Any]] = []
        for context_seed in context_seeds:
            indices = _base_memory_indices(
                train_y,
                train_groups,
                base_ids=base_ids,
                count=base_count,
                context_seed=context_seed,
                namespace=str(config["base_context_namespace"]),
            )
            groups = tuple(train_groups[index] for index in indices)
            if len(groups) != len(set(groups)):
                raise AssertionError("Cached base memory is not group-distinct.")
            if set(groups) & set(query_groups):
                raise AssertionError("Cached base memory overlaps the query.")
            base_memories.append(
                {
                    "context_seed": context_seed,
                    "indices": indices,
                    "features": train_x[indices],
                    "labels": train_y[indices],
                    "groups": groups,
                }
            )
            context_manifest.append(
                {
                    "context_seed": context_seed,
                    "fixed_across_shots_and_draws": True,
                    "examples_per_base_class": base_count,
                    "base_groups_by_class": {
                        str(class_id): [
                            groups[index]
                            for index in np.flatnonzero(train_y[indices] == class_id)
                        ]
                        for class_id in base_ids
                    },
                }
            )
        _heartbeat(
            run_id,
            "base_memories_ready",
            contexts=len(base_memories),
            examples_per_base_class=base_count,
            elapsed_seconds=time.perf_counter() - started,
        )

        from tabpfn import TabPFNClassifier
        from tabpfn.model_loading import resolve_model_path

        _heartbeat(run_id, "checkpoint_resolution_started")
        checkpoint_started = time.perf_counter()
        checkpoint_path, _, model_name, _ = resolve_model_path(None, "classifier", "v2")
        checkpoint_resolution_seconds = time.perf_counter() - checkpoint_started
        expected_checkpoint = protocol["tabpfn"]["checkpoint_sha256"]
        if file_sha256(checkpoint_path) != expected_checkpoint:
            raise RuntimeError("TabPFN checkpoint differs from the frozen protocol.")
        _heartbeat(
            run_id,
            "checkpoint_resolved",
            elapsed_seconds=checkpoint_resolution_seconds,
        )
        initialization_started = time.perf_counter()
        models = [
            TabPFNClassifier(
                n_estimators=1,
                model_path=checkpoint_path,
                device=str(cuda),
                ignore_pretraining_limits=True,
                fit_mode="fit_with_cache",
                random_state=context_seed,
                n_jobs=1,
            )
            for context_seed in context_seeds
        ]
        model_initialization_seconds = time.perf_counter() - initialization_started
        _heartbeat(
            run_id,
            "models_initialized",
            elapsed_seconds=model_initialization_seconds,
            models=len(models),
        )

        base_query = np.isin(query_y, base_ids)
        pre_probabilities: list[np.ndarray] = []
        pre_context_rows: list[dict[str, Any]] = []
        pre_started = time.perf_counter()
        for model, memory in zip(models, base_memories, strict=True):
            context_started = time.perf_counter()
            _heartbeat(
                run_id,
                "pre_enrollment_context_started",
                context_seed=int(memory["context_seed"]),
            )
            model.fit(memory["features"], memory["labels"])
            _heartbeat(
                run_id,
                "pre_enrollment_fit_completed",
                context_seed=int(memory["context_seed"]),
                elapsed_seconds=time.perf_counter() - context_started,
            )
            probability = _aligned_probability(
                model.predict_proba(query_x[base_query]), model.classes_
            )
            elapsed = time.perf_counter() - context_started
            pre_probabilities.append(probability)
            pre_context_rows.append(
                {
                    "context_seed": int(memory["context_seed"]),
                    "accuracy": float(
                        (probability.argmax(1) == query_y[base_query]).mean()
                    ),
                    "elapsed_seconds": elapsed,
                }
            )
            _heartbeat(
                run_id,
                "pre_enrollment_context_completed",
                context_seed=int(memory["context_seed"]),
                elapsed_seconds=elapsed,
            )
        pre_probability = np.mean(pre_probabilities, axis=0)
        pre_prediction = pre_probability.argmax(1)
        pre_class_recall = {
            str(class_id): float(
                (pre_prediction[query_y[base_query] == class_id] == class_id).mean()
            )
            for class_id in base_ids
        }
        pre_enrollment = {
            "accuracy": float((pre_prediction == query_y[base_query]).mean()),
            "balanced_accuracy": float(np.mean(list(pre_class_recall.values()))),
            "per_class_recall": pre_class_recall,
            "query_examples": int(base_query.sum()),
            "context_examples": int(len(base_ids) * base_count),
            "elapsed_seconds": time.perf_counter() - pre_started,
            "context_rows": pre_context_rows,
        }
        _heartbeat(
            run_id,
            "pre_enrollment_completed",
            elapsed_seconds=pre_enrollment["elapsed_seconds"],
        )

        rows: list[dict[str, Any]] = []
        evidence_row_ids: list[str] = []
        evidence_predictions: list[np.ndarray] = []
        evidence_probabilities: list[np.ndarray] = []
        support_manifest: list[dict[str, Any]] = []
        context_sensitivity_rows: list[dict[str, Any]] = []
        svm_diagnostic_rows: list[dict[str, Any]] = []
        first_context_seconds: float | None = None

        for shots, draw in itertools.product(
            (int(value) for value in config["shots"]),
            range(int(config["draws"])),
        ):
            _heartbeat(run_id, "draw_started", shots=shots, draw=draw)
            selected = deterministic_support_indices(
                reference_frame,
                class_ids=pair,
                shots=shots,
                seed=seed,
                draw=draw,
                namespace=str(config["support_namespace"]),
            )
            positions = reference_frame.index.get_indexer(selected)
            if np.any(positions < 0):
                raise AssertionError("Support rows did not map to the reference pool.")
            support_x = reference_x[positions]
            support_y = reference_frame.loc[selected, "Class"].to_numpy(dtype=int)
            support_groups = tuple(
                reference_frame.loc[selected, "_input_group"].astype(str)
            )
            if len(support_groups) != len(set(support_groups)):
                raise AssertionError("Support groups are not unique.")
            if set(support_groups) & set(query_groups):
                raise AssertionError("Support and query groups overlap.")
            for memory in base_memories:
                if set(support_groups) & set(memory["groups"]):
                    raise AssertionError("Support and cached base memory overlap.")
            support_manifest.append(
                {
                    "shots": shots,
                    "draw": draw,
                    "groups_by_class": {
                        str(class_id): [
                            support_groups[index]
                            for index in np.flatnonzero(support_y == class_id)
                        ]
                        for class_id in pair
                    },
                    "query_used": False,
                }
            )

            probabilities: dict[str, list[np.ndarray]] = {
                "cached_base_tabpfn_v2": [],
                "cached_base_linear_svm": [],
            }
            elapsed = {name: 0.0 for name in probabilities}
            for model, memory in zip(models, base_memories, strict=True):
                context_x = np.vstack((memory["features"], support_x))
                context_y = np.r_[memory["labels"], support_y]

                method_started = time.perf_counter()
                _heartbeat(
                    run_id,
                    "tabpfn_context_started",
                    shots=shots,
                    draw=draw,
                    context_seed=int(memory["context_seed"]),
                )
                model.fit(context_x, context_y)
                _heartbeat(
                    run_id,
                    "tabpfn_fit_completed",
                    shots=shots,
                    draw=draw,
                    context_seed=int(memory["context_seed"]),
                    elapsed_seconds=time.perf_counter() - method_started,
                )
                probability = _aligned_probability(
                    model.predict_proba(query_x), model.classes_
                )
                context_elapsed = time.perf_counter() - method_started
                if first_context_seconds is None:
                    first_context_seconds = context_elapsed
                probabilities["cached_base_tabpfn_v2"].append(probability)
                elapsed["cached_base_tabpfn_v2"] += context_elapsed
                context_metrics = hard_prediction_metrics(
                    query_y,
                    probability.argmax(1),
                    base_class_ids=base_ids,
                    enrolled_class_ids=pair,
                )
                context_sensitivity_rows.append(
                    {
                        "shots": shots,
                        "draw": draw,
                        "context_seed": int(memory["context_seed"]),
                        "elapsed_seconds": context_elapsed,
                        "base_accuracy": context_metrics["base_accuracy"],
                        "enrolled_accuracy": context_metrics["enrolled_accuracy"],
                        "harmonic_mean": context_metrics["harmonic_mean"],
                        "accuracy": context_metrics["accuracy"],
                        "worst_enrolled_recall": context_metrics["worst_enrolled_recall"],
                    }
                )
                _heartbeat(
                    run_id,
                    "tabpfn_context_completed",
                    shots=shots,
                    draw=draw,
                    context_seed=int(memory["context_seed"]),
                    elapsed_seconds=context_elapsed,
                )

                method_started = time.perf_counter()
                _heartbeat(
                    run_id,
                    "svm_context_started",
                    shots=shots,
                    draw=draw,
                    context_seed=int(memory["context_seed"]),
                )
                svm_probability, svm_diagnostic = _bounded_linear_svm_probability(
                    context_x, context_y, query_x
                )
                probabilities["cached_base_linear_svm"].append(svm_probability)
                svm_elapsed = time.perf_counter() - method_started
                elapsed["cached_base_linear_svm"] += svm_elapsed
                svm_diagnostic_rows.append(
                    {
                        "shots": shots,
                        "draw": draw,
                        "context_seed": int(memory["context_seed"]),
                        "elapsed_seconds": svm_elapsed,
                        **svm_diagnostic,
                    }
                )
                _heartbeat(
                    run_id,
                    "svm_context_completed",
                    shots=shots,
                    draw=draw,
                    context_seed=int(memory["context_seed"]),
                    elapsed_seconds=svm_elapsed,
                )

            _heartbeat(
                run_id,
                "draw_contexts_completed",
                shots=shots,
                draw=draw,
            )

            for method, context_probabilities in probabilities.items():
                mean_probability = np.mean(context_probabilities, axis=0)
                mean_probability /= mean_probability.sum(1, keepdims=True)
                row = metric_row(
                    labels=query_y,
                    probability=mean_probability,
                    base_class_ids=base_ids,
                    enrolled_class_ids=pair,
                    method=method,
                    shots=shots,
                    draw=draw,
                    elapsed_seconds=elapsed[method],
                    probability_source=(
                        "native_tabpfn_probability_ensemble"
                        if method == "cached_base_tabpfn_v2"
                        else "fixed_softmax_score_ensemble"
                    ),
                    extra={
                        "base_class_ids": list(base_ids),
                        "enrolled_class_ids": list(pair),
                        "ensemble_contexts": len(context_seeds),
                        "base_context_examples": len(base_ids) * base_count,
                        "appended_novel_examples": len(pair) * shots,
                        "context_examples": len(base_ids) * base_count + len(pair) * shots,
                        "query_examples": len(query_y),
                    },
                )
                rows.append(row)
                evidence_row_ids.append(
                    f"{method}|shot={shots}|draw={draw}"
                )
                evidence_predictions.append(mean_probability.argmax(1).astype(np.uint8))
                evidence_probabilities.append(mean_probability.astype(np.float64))
            _heartbeat(run_id, "draw_completed", shots=shots, draw=draw)

        expected_rows = (
            len(config["methods"]) * len(config["shots"]) * int(config["draws"])
        )
        if len(rows) != expected_rows or {row["method"] for row in rows} != set(config["methods"]):
            raise AssertionError("Pilot result matrix is incomplete.")

        peak_allocated = int(torch.cuda.max_memory_allocated(cuda))
        peak_reserved = int(torch.cuda.max_memory_reserved(cuda))
        if peak_allocated <= 0:
            raise RuntimeError("No CUDA allocation was observed.")
        weight_hashes = {str(checkpoint_path): file_sha256(checkpoint_path)}
        for model in models:
            for path in _weight_paths(model):
                weight_hashes[str(path)] = file_sha256(path)

        result = {
            "schema_version": 1,
            "evidence_schema": EVIDENCE_SCHEMA,
            "run_id": run_id,
            "pilot_config_sha256": CONFIG_SHA256,
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "dataset_sha256": file_sha256(DATA_PATH),
            "pair": list(pair),
            "seed": int(seed),
            "regime": "summary_only",
            "requested_draws": int(config["draws"]),
            "requested_shots": [int(value) for value in config["shots"]],
            "methods": list(config["methods"]),
            "base_class_ids": list(base_ids),
            "enrolled_class_ids": list(pair),
            "query_examples": int(len(query_y)),
            "query_class_counts": {
                str(class_id): int((query_y == class_id).sum())
                for class_id in range(N_CLASSES)
            },
            "pre_enrollment": pre_enrollment,
            "rows": rows,
            "context_sensitivity_rows": context_sensitivity_rows,
            "svm_diagnostic_rows": svm_diagnostic_rows,
            "svm_solver_policy": {
                "status": "post_freeze_operational_safeguard",
                "solver": "sklearn.svm.SVC",
                "kernel": "linear",
                "max_iter": SVM_MAX_ITER,
                "primary_tabpfn_affected": False,
                "uniform_replay_required_before_final_svm_comparison": True,
            },
            "duration_seconds": time.perf_counter() - started,
            "device": str(cuda),
            "environment": environment_metadata(cuda),
            "tabpfn_package_version": importlib.metadata.version("tabpfn"),
            "tabpfn_model_name": model_name,
            "tabpfn_checkpoint_resolution_seconds": checkpoint_resolution_seconds,
            "tabpfn_model_initialization_seconds": model_initialization_seconds,
            "tabpfn_first_context_seconds_including_lazy_load": first_context_seconds,
            "tabpfn_checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "cuda_diagnostics": {
                "actual_device": str(cuda),
                "current_device_index": int(torch.cuda.current_device()),
                "compute_capability": list(torch.cuda.get_device_capability(cuda)),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            },
            "weight_hashes": weight_hashes,
            "source": _git_metadata(REPOSITORY_ROOT),
            "interpretation": config["interpretation"],
        }
        atomic_json(root / "metrics.json", result)
        np.savez_compressed(
            root / "prediction_evidence.npz",
            labels=query_y.astype(np.uint8),
            query_group_ids=np.asarray(query_groups),
            row_ids=np.asarray(evidence_row_ids),
            predictions=np.stack(evidence_predictions),
            probabilities=np.stack(evidence_probabilities),
            pre_enrollment_labels=query_y[base_query].astype(np.uint8),
            pre_enrollment_predictions=pre_prediction.astype(np.uint8),
            pre_enrollment_probabilities=pre_probability.astype(np.float64),
        )
        atomic_json(root / "support_manifest.json", support_manifest)
        atomic_json(root / "context_manifest.json", context_manifest)
        atomic_json(
            root / "query_manifest.json",
            {
                "groups": list(query_groups),
                "labels": query_y.tolist(),
                "selection_namespace": "tabpfn-query",
                "cap_per_class": int(config["query_examples_per_class"]),
                "used_for_fitting_or_selection": False,
            },
        )
        atomic_json(
            root / "split_manifest.json",
            lifecycle_split_manifest(fold.split, data_path=DATA_PATH, regime="summary_only"),
        )
        write_manifest(root, {**expected, "completed": True, "device": str(cuda)})
        _append_registry(
            "completed",
            run_id=run_id,
            pair=list(pair),
            seed=int(seed),
            device=str(cuda),
            duration_seconds=result["duration_seconds"],
            rows=len(rows),
        )
        _update_state(run_id=run_id, completed=True)
        del models
        torch.cuda.empty_cache()
        return result
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "pair": list(pair),
            "seed": int(seed),
            "device": str(cuda),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        append_jsonl(PILOT_ROOT / "failures.jsonl", {"timestamp": utc_now(), **failure})
        _append_registry("failed", **failure)
        _update_state(run_id=run_id, completed=False, failure=failure)
        raise


def run_matrix(*, device: str = "cuda:0", pair_limit: int | None = None) -> dict[str, Any]:
    config = load_config()
    frame = pd.read_csv(DATA_PATH)
    pairs = [tuple(int(value) for value in pair) for pair in config["pairs"]]
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    seeds = config.get("execution_seeds", config["seeds"])
    units = list(itertools.product(pairs, seeds))
    results = []
    for index, (pair, seed) in enumerate(units, start=1):
        print(
            f"[{ACTIVE_STUDY} {index}/{len(units)}] pair={pair[0]}-{pair[1]} seed={seed}",
            flush=True,
        )
        results.append(
            run_unit(frame=frame, pair=pair, seed=int(seed), device=device)
        )
    return {
        "validated_units": len(results),
        "rows": sum(len(result["rows"]) for result in results),
        "device": device,
    }


def discover_units() -> list[Path]:
    units: list[Path] = []
    roots = list((PILOT_ROOT / "units").glob("pair_*/seed_*"))
    if ACTIVE_STUDY == "confirmatory" and load_config().get(
        "include_observed_seed_in_combined_analysis"
    ):
        observed_seed = int(load_config()["previously_observed_seed"])
        roots.extend(
            (EXPLORATORY_ROOT / "units").glob(
                f"pair_*/seed_{observed_seed}"
            )
        )
    for root in sorted(roots, key=str):
        if not (root / "manifest.json").is_file():
            continue
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        origin_sha = (
            EXPLORATORY_CONFIG_SHA256
            if EXPLORATORY_ROOT in root.parents
            else CONFIG_SHA256
        )
        expected = _expected_manifest(
            tuple(metrics["pair"]),
            int(metrics["seed"]),
            config_sha256=origin_sha,
        )
        valid, reason = validate_run(root, expected=expected)
        if not valid:
            raise RuntimeError(f"Invalid pilot unit {root}: {reason}")
        units.append(root)
    return units


def _load_rows(units: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scalars: list[dict[str, Any]] = []
    originals: list[dict[str, Any]] = []
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = "-".join(str(value) for value in metrics["pair"])
        for row_index, row in enumerate(metrics["rows"]):
            originals.append({"unit_dir": str(unit), "row_index": row_index, "row": row})
            scalars.append(
                {
                    "run_id": metrics["run_id"],
                    "pair": pair,
                    "fault_a": int(metrics["pair"][0]),
                    "fault_b": int(metrics["pair"][1]),
                    "seed": int(metrics["seed"]),
                    "method": row["method"],
                    "shots": int(row["shots"]),
                    "draw": int(row["draw"]),
                    **{
                        name: float(row[name])
                        for name in (
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                            "base_accuracy",
                            "enrolled_accuracy",
                            "harmonic_mean",
                            "worst_enrolled_recall",
                            "normal_far_after_enrollment",
                            "nll",
                            "brier",
                            "ece_15",
                            "elapsed_seconds",
                        )
                    },
                }
            )
    return pd.DataFrame(scalars), originals


def _audit_manifests(units: list[Path]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    matched_frozen = 0
    cuda_units = 0
    pre_enrollment_reconstructed = 0
    config = load_config()
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = tuple(int(value) for value in metrics["pair"])
        query = json.loads((unit / "query_manifest.json").read_text(encoding="utf-8"))
        support = json.loads((unit / "support_manifest.json").read_text(encoding="utf-8"))
        contexts = json.loads((unit / "context_manifest.json").read_text(encoding="utf-8"))
        query_groups = set(query["groups"])
        query_labels = np.asarray(query["labels"], dtype=int)
        if len(query_groups) != len(query["groups"]):
            failures.append({"unit": str(unit), "failure": "duplicate_query_group"})
        if any(
            int((query_labels == class_id).sum())
            != int(config["query_examples_per_class"])
            for class_id in range(N_CLASSES)
        ):
            failures.append({"unit": str(unit), "failure": "query_class_count_mismatch"})
        if (
            metrics.get("device") == "cuda:0"
            and metrics.get("cuda_diagnostics", {}).get("actual_device") == "cuda:0"
            and int(metrics.get("cuda_diagnostics", {}).get("peak_allocated_bytes", 0)) > 0
        ):
            cuda_units += 1
        else:
            failures.append({"unit": str(unit), "failure": "cuda_evidence_missing"})
        all_context_groups: set[str] = set()
        if len(contexts) != len(config["context_seeds"]):
            failures.append({"unit": str(unit), "failure": "context_seed_count_mismatch"})
        for context in contexts:
            groups = [
                group
                for values in context["base_groups_by_class"].values()
                for group in values
            ]
            if len(groups) != len(set(groups)):
                failures.append({"unit": str(unit), "failure": "duplicate_base_context_group"})
            if set(groups) & query_groups:
                failures.append({"unit": str(unit), "failure": "base_context_query_overlap"})
            if not context.get("fixed_across_shots_and_draws"):
                failures.append({"unit": str(unit), "failure": "base_context_not_marked_fixed"})
            if any(
                len(values) != int(config["base_context_per_class"])
                for values in context["base_groups_by_class"].values()
            ):
                failures.append({"unit": str(unit), "failure": "base_context_class_count_mismatch"})
            all_context_groups.update(groups)
        if len(support) != len(config["shots"]) * int(config["draws"]):
            failures.append({"unit": str(unit), "failure": "support_draw_count_mismatch"})
        for row in support:
            groups = [group for values in row["groups_by_class"].values() for group in values]
            if len(groups) != len(set(groups)):
                failures.append({"unit": str(unit), "failure": "duplicate_support_group"})
            if set(groups) & (query_groups | all_context_groups):
                failures.append({"unit": str(unit), "failure": "support_context_or_query_overlap"})

        frozen = (
            STUDY_ROOT
            / "summary_only"
            / f"pair_{pair[0]:02d}_{pair[1]:02d}"
            / f"seed_{metrics['seed']}"
        )
        frozen_query = json.loads((frozen / "query_manifest.json").read_text(encoding="utf-8"))
        frozen_support = json.loads((frozen / "support_manifest.json").read_text(encoding="utf-8"))
        if query != frozen_query:
            failures.append({"unit": str(unit), "failure": "frozen_query_mismatch"})
        if support != frozen_support:
            failures.append({"unit": str(unit), "failure": "frozen_support_mismatch"})
        if query == frozen_query and support == frozen_support:
            matched_frozen += 1
        with np.load(unit / "prediction_evidence.npz") as evidence:
            pre_labels = evidence["pre_enrollment_labels"].astype(int)
            pre_probability = evidence["pre_enrollment_probabilities"].astype(float)
            pre_prediction = evidence["pre_enrollment_predictions"].astype(int)
        if not np.array_equal(pre_prediction, pre_probability.argmax(1)):
            failures.append({"unit": str(unit), "failure": "pre_enrollment_argmax_mismatch"})
        else:
            accuracy = float((pre_prediction == pre_labels).mean())
            recalls = {
                str(class_id): float((pre_prediction[pre_labels == class_id] == class_id).mean())
                for class_id in metrics["base_class_ids"]
            }
            saved = metrics["pre_enrollment"]
            if (
                abs(accuracy - float(saved["accuracy"])) > 1e-12
                or abs(float(np.mean(list(recalls.values()))) - float(saved["balanced_accuracy"])) > 1e-12
                or any(
                    abs(value - float(saved["per_class_recall"][class_id])) > 1e-12
                    for class_id, value in recalls.items()
                )
            ):
                failures.append({"unit": str(unit), "failure": "pre_enrollment_metric_mismatch"})
            else:
                pre_enrollment_reconstructed += 1
    return {
        "units": len(units),
        "matched_frozen_query_and_support_units": matched_frozen,
        "cuda_units": cuda_units,
        "pre_enrollment_reconstructed_units": pre_enrollment_reconstructed,
        "failures": failures,
        "passed": not failures,
    }


def _frozen_comparator() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = {int(value) for value in load_config()["seeds"]}
    for unit in sorted((STUDY_ROOT / "summary_only").glob("pair_*/seed_*")):
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        if int(metrics["seed"]) not in seeds:
            continue
        pair = "-".join(str(value) for value in metrics["pair"])
        for row in metrics["rows"]:
            if row["method"] != "tabpfn_v2":
                continue
            rows.append(
                {
                    "pair": pair,
                    "seed": int(metrics["seed"]),
                    "shots": int(row["shots"]),
                    "draw": int(row["draw"]),
                    **{
                        name: float(row[name])
                        for name in (
                            "base_accuracy",
                            "enrolled_accuracy",
                            "harmonic_mean",
                            "balanced_accuracy",
                            "macro_f1",
                            "worst_enrolled_recall",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def _per_fault(rows: pd.DataFrame, originals: list[dict[str, Any]]) -> pd.DataFrame:
    values: list[dict[str, Any]] = []
    for scalar, original in zip(rows.to_dict("records"), originals, strict=True):
        row = original["row"]
        cm = np.asarray(row["confusion_matrix"], dtype=float)
        recall = np.divide(
            np.diag(cm),
            cm.sum(1),
            out=np.zeros(N_CLASSES, dtype=float),
            where=cm.sum(1) > 0,
        )
        for class_id in row["enrolled_class_ids"]:
            values.append(
                {
                    "pair": scalar["pair"],
                    "seed": scalar["seed"],
                    "method": scalar["method"],
                    "shots": scalar["shots"],
                    "draw": scalar["draw"],
                    "class_id": int(class_id),
                    "recall": float(recall[int(class_id)]),
                }
            )
    return pd.DataFrame(values)


def _plot_results(
    pair_units: pd.DataFrame,
    frozen_pair_units: pd.DataFrame,
    fault: pd.DataFrame,
    pre: pd.DataFrame,
) -> list[str]:
    root = PILOT_ROOT / "plots"
    root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    plt.figure(figsize=(7.2, 5.0))
    for method, label in (
        ("cached_base_tabpfn_v2", "Cached-base TabPFN"),
        ("cached_base_linear_svm", "Cached-base linear SVM"),
    ):
        part = pair_units[pair_units["method"] == method]
        summary = part.groupby("shots")["harmonic_mean"].agg(["mean", "std"])
        plt.errorbar(summary.index, summary["mean"], yerr=summary["std"], marker="o", capsize=4, label=label)
    frozen = frozen_pair_units.groupby("shots")["harmonic_mean"].agg(["mean", "std"])
    plt.errorbar(frozen.index, frozen["mean"], yerr=frozen["std"], marker="o", capsize=4, label="Frozen balanced-context TabPFN")
    plt.ylim(0, 1)
    plt.xticks((1, 3, 5))
    plt.xlabel("Novel-fault examples per enrolled class")
    plt.ylabel("Base/enrolled harmonic mean")
    plt.title("Fixed base memory versus balanced fresh context")
    plt.legend()
    plt.tight_layout()
    path = root / "incremental_memory_shot_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    five = pair_units[(pair_units["shots"] == 5) & (pair_units["method"] == "cached_base_tabpfn_v2")]
    long = five.melt(
        id_vars="pair",
        value_vars=("base_accuracy", "enrolled_accuracy", "harmonic_mean"),
        var_name="metric",
        value_name="value",
    )
    pivot = long.groupby("metric")["value"].agg(["mean", "std"])
    order = ["base_accuracy", "enrolled_accuracy", "harmonic_mean"]
    plt.figure(figsize=(6.7, 4.8))
    plt.bar(range(3), pivot.loc[order, "mean"], yerr=pivot.loc[order, "std"], capsize=5)
    plt.xticks(range(3), ["Base", "Enrolled", "H-mean"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy / score")
    plt.title("Cached-base TabPFN at five shots")
    plt.tight_layout()
    path = root / "incremental_memory_five_shot_components.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    selected_fault = fault[(fault["method"] == "cached_base_tabpfn_v2") & (fault["shots"] == 5)]
    fault_summary = selected_fault.groupby("class_id")["recall"].agg(["mean", "std"])
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(fault_summary.index.astype(str), fault_summary["mean"], yerr=fault_summary["std"], capsize=4)
    plt.ylim(0, 1)
    plt.xlabel("Enrolled fault class")
    plt.ylabel("Recall")
    plt.title("Five-shot recall by newly enrolled fault")
    plt.tight_layout()
    path = root / "incremental_memory_per_fault_recall.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    pre_plot = pre.groupby("pair", as_index=False)[
        "pre_enrollment_base_accuracy"
    ].mean()
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(pre_plot["pair"], pre_plot["pre_enrollment_base_accuracy"])
    plt.axhline(pre["pre_enrollment_base_accuracy"].mean(), color="black", linestyle="--", label="Mean")
    plt.xticks(rotation=60, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Base-class accuracy")
    plt.title("Accuracy before appending novel-fault examples")
    plt.legend()
    plt.tight_layout()
    path = root / "incremental_memory_pre_enrollment.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)
    return created


def _analysis_cohorts(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separate genuinely new replication evidence from combined sensitivity."""
    if ACTIVE_STUDY == "confirmatory":
        replication_seeds = {
            int(value) for value in load_config()["execution_seeds"]
        }
        return {
            "primary_replication": frame[frame["seed"].isin(replication_seeds)].copy(),
            "combined_five_seed": frame.copy(),
        }
    return {"exploratory_seed42": frame.copy()}


def analyze(*, expected_units: int | None = None, bootstrap_iterations: int = 5000, sign_flip_iterations: int = 20000) -> dict[str, Any]:
    units = discover_units()
    if expected_units is None:
        expected_units = 105 if ACTIVE_STUDY == "confirmatory" else 21
    if len(units) != expected_units:
        raise RuntimeError(f"Study has {len(units)}/{expected_units} validated units.")
    rows, originals = _load_rows(units)
    expected_rows = expected_units * len(load_config()["methods"]) * len(load_config()["shots"]) * int(load_config()["draws"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"Pilot has {len(rows)}/{expected_rows} result rows.")
    reconstruction = reconstruct_metrics(originals)
    if not reconstruction["passed"]:
        raise AssertionError("Pilot metric reconstruction failed.")
    audit = _audit_manifests(units)
    if not audit["passed"]:
        raise AssertionError("Pilot group or comparator audit failed.")

    table_root = PILOT_ROOT / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    rows.to_csv(table_root / "per_draw.csv", index=False)
    pair_units = rows.groupby(
        ["pair", "fault_a", "fault_b", "seed", "method", "shots"], as_index=False
    ).mean(numeric_only=True)
    pair_units.to_csv(table_root / "pair_units.csv", index=False)

    cohort_pair_units = _analysis_cohorts(pair_units)
    summary_rows: list[dict[str, Any]] = []
    for cohort, cohort_frame in cohort_pair_units.items():
        for (method, shots), part in cohort_frame.groupby(["method", "shots"], sort=True):
            for metric in (
                "harmonic_mean",
                "base_accuracy",
                "enrolled_accuracy",
                "balanced_accuracy",
                "macro_f1",
                "worst_enrolled_recall",
                "normal_far_after_enrollment",
                "nll",
                "brier",
                "ece_15",
                "elapsed_seconds",
            ):
                estimate = hierarchical_bootstrap(
                    part,
                    metric,
                    iterations=bootstrap_iterations,
                )
                summary_rows.append(
                    {
                        "cohort": cohort,
                        "method": method,
                        "shots": int(shots),
                        "metric": metric,
                        **estimate,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_root / "summary.csv", index=False)
    stability_rows = []
    for cohort, cohort_frame in cohort_pair_units.items():
        for (method, shots), part in cohort_frame.groupby(["method", "shots"], sort=True):
            values = part["harmonic_mean"].to_numpy(dtype=float)
            stability_rows.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "shots": int(shots),
                    "pair_seed_units": len(values),
                    "fraction_h_ge_0_90": float(np.mean(values >= 0.90)),
                    "fraction_h_ge_0_95": float(np.mean(values >= 0.95)),
                    "minimum_h": float(np.min(values)),
                    "q05_h": float(np.quantile(values, 0.05)),
                    "median_h": float(np.median(values)),
                }
            )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(table_root / "stability.csv", index=False)

    frozen = _frozen_comparator()
    frozen_pair = frozen.groupby(["pair", "seed", "shots"], as_index=False).mean(numeric_only=True)
    frozen_pair.to_csv(table_root / "frozen_balanced_context_pair_units.csv", index=False)
    comparisons: list[dict[str, Any]] = []
    row_cohorts = _analysis_cohorts(rows)
    frozen_cohorts = _analysis_cohorts(frozen)
    for cohort, cohort_rows in row_cohorts.items():
        cohort_comparisons: list[dict[str, Any]] = []
        cached = cohort_rows[
            cohort_rows["method"] == "cached_base_tabpfn_v2"
        ]
        frozen_cohort = frozen_cohorts[cohort]
        for shots in sorted(cohort_rows["shots"].unique()):
            comparison = paired_sign_flip(
                cached[cached["shots"] == shots],
                frozen_cohort[frozen_cohort["shots"] == shots],
                value="harmonic_mean",
                iterations=sign_flip_iterations,
            )
            cohort_comparisons.append(
                {"cohort": cohort, "shots": int(shots), **comparison}
            )
        adjusted = holm_adjust(
            [row["two_sided_permutation_p"] for row in cohort_comparisons]
        )
        for row, value in zip(cohort_comparisons, adjusted, strict=True):
            row["holm_adjusted_p"] = value
        comparisons.extend(cohort_comparisons)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(table_root / "paired_vs_frozen_tabpfn.csv", index=False)
    cached_method_comparisons: list[dict[str, Any]] = []
    for cohort, cohort_rows in row_cohorts.items():
        cohort_method_comparisons: list[dict[str, Any]] = []
        cached = cohort_rows[
            cohort_rows["method"] == "cached_base_tabpfn_v2"
        ]
        cached_svm = cohort_rows[
            cohort_rows["method"] == "cached_base_linear_svm"
        ]
        for shots in sorted(cohort_rows["shots"].unique()):
            comparison = paired_sign_flip(
                cached[cached["shots"] == shots],
                cached_svm[cached_svm["shots"] == shots],
                value="harmonic_mean",
                iterations=sign_flip_iterations,
                seed=20261201 + int(shots),
            )
            cohort_method_comparisons.append(
                {"cohort": cohort, "shots": int(shots), **comparison}
            )
        cached_adjusted = holm_adjust(
            [row["two_sided_permutation_p"] for row in cohort_method_comparisons]
        )
        for row, value in zip(
            cohort_method_comparisons, cached_adjusted, strict=True
        ):
            row["holm_adjusted_p"] = value
        cached_method_comparisons.extend(cohort_method_comparisons)
    cached_comparison_frame = pd.DataFrame(cached_method_comparisons)
    cached_comparison_frame.to_csv(
        table_root / "paired_cached_tabpfn_vs_svm.csv", index=False
    )

    fault = _per_fault(rows, originals)
    fault.to_csv(table_root / "per_fault_draw.csv", index=False)
    fault_units = fault.groupby(["pair", "seed", "method", "shots", "class_id"], as_index=False)["recall"].mean()
    fault_units.to_csv(table_root / "per_fault_pair_units.csv", index=False)
    fault_summary = fault_units.groupby(["method", "shots", "class_id"], as_index=False)["recall"].agg(["mean", "std", "min"])
    fault_summary.to_csv(table_root / "per_fault_summary.csv", index=False)

    pre_rows = []
    context_rows = []
    efficiency_rows = []
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = "-".join(str(value) for value in metrics["pair"])
        pre_rows.append(
            {
                "pair": pair,
                "seed": int(metrics["seed"]),
                "pre_enrollment_base_accuracy": float(metrics["pre_enrollment"]["accuracy"]),
                "pre_enrollment_balanced_accuracy": float(metrics["pre_enrollment"]["balanced_accuracy"]),
            }
        )
        context_rows.extend(
            {"pair": pair, "seed": int(metrics["seed"]), **row}
            for row in metrics["context_sensitivity_rows"]
        )
        efficiency_rows.append(
            {
                "pair": pair,
                "seed": int(metrics["seed"]),
                "duration_seconds": float(metrics["duration_seconds"]),
                "peak_cuda_allocated_bytes": int(metrics["cuda_diagnostics"]["peak_allocated_bytes"]),
                "peak_cuda_reserved_bytes": int(metrics["cuda_diagnostics"]["peak_reserved_bytes"]),
                "checkpoint_size_bytes": int(metrics["tabpfn_checkpoint_size_bytes"]),
            }
        )
    pre = pd.DataFrame(pre_rows)
    pre.to_csv(table_root / "pre_enrollment.csv", index=False)
    context = pd.DataFrame(context_rows)
    context.to_csv(table_root / "context_sensitivity.csv", index=False)
    context_within = (
        context.groupby(["pair", "seed", "shots", "draw"], as_index=False)[
            "harmonic_mean"
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    context_within["range"] = context_within["max"] - context_within["min"]
    context_summary = (
        context_within.groupby("shots", as_index=False)
        .agg(
            pair_draw_units=("pair", "size"),
            mean_context_std=("std", "mean"),
            median_context_std=("std", "median"),
            maximum_context_std=("std", "max"),
            mean_context_range=("range", "mean"),
        )
    )
    context_within.to_csv(table_root / "context_sensitivity_pair_draw.csv", index=False)
    context_summary.to_csv(table_root / "context_sensitivity_summary.csv", index=False)
    efficiency = pd.DataFrame(efficiency_rows)
    efficiency.to_csv(table_root / "efficiency.csv", index=False)

    sensitivity_rows = []
    sensitivity_note = None
    if pair_units["pair"].nunique() == 21:
        for (method, shots), part in pair_units.groupby(["method", "shots"]):
            sensitivity_rows.append(
                {
                    "method": method,
                    "shots": int(shots),
                    **fault_aware_leave_one_out(part, "harmonic_mean"),
                }
            )
    else:
        sensitivity_note = (
            "Fault-aware leave-one-out is intentionally deferred until all "
            "21 held-out pairs are present."
        )
    atomic_json(
        table_root / "fault_dependence_sensitivity.json",
        {"note": sensitivity_note, "rows": sensitivity_rows},
    )
    plots = _plot_results(pair_units, frozen_pair, fault, pre)

    h = summary[summary["metric"] == "harmonic_mean"].copy()
    h_lookup = {
        (row.cohort, row.method, int(row.shots)): row
        for row in h.itertuples(index=False)
    }
    comparison_lookup = {
        (row.cohort, int(row.shots)): row
        for row in comparison_frame.itertuples(index=False)
    }
    cached_comparison_lookup = {
        (row.cohort, int(row.shots)): row
        for row in cached_comparison_frame.itertuples(index=False)
    }
    stability_lookup = {
        (row.cohort, row.method, int(row.shots)): row
        for row in stability.itertuples(index=False)
    }
    five_fault = fault_summary[
        (fault_summary["method"] == "cached_base_tabpfn_v2")
        & (fault_summary["shots"] == 5)
    ]
    config = load_config()
    report_cohort = (
        "primary_replication"
        if ACTIVE_STUDY == "confirmatory"
        else "exploratory_seed42"
    )
    primary_seeds = {
        int(value)
        for value in config.get("execution_seeds", config["seeds"])
    }
    primary_pre = pre[pre["seed"].isin(primary_seeds)]
    success = None
    if ACTIVE_STUDY == "confirmatory":
        criterion = config["success_criteria"]
        five_h = h_lookup[(report_cohort, "cached_base_tabpfn_v2", 5)]
        five_stability = stability_lookup[
            (report_cohort, "cached_base_tabpfn_v2", 5)
        ]
        success_checks = {
            "five_shot_mean_h": {
                "observed": float(five_h.mean),
                "threshold": float(criterion["five_shot_mean_h_at_least"]),
                "passed": float(five_h.mean)
                >= float(criterion["five_shot_mean_h_at_least"]),
            },
            "five_shot_lower_95_ci": {
                "observed": float(five_h.ci_low),
                "threshold": float(criterion["five_shot_lower_95_ci_at_least"]),
                "passed": float(five_h.ci_low)
                >= float(criterion["five_shot_lower_95_ci_at_least"]),
            },
            "five_shot_fraction_h_ge_0_95": {
                "observed": float(five_stability.fraction_h_ge_0_95),
                "threshold": float(
                    criterion["five_shot_pair_seed_fraction_h_at_least_0_95"]
                ),
                "passed": float(five_stability.fraction_h_ge_0_95)
                >= float(
                    criterion["five_shot_pair_seed_fraction_h_at_least_0_95"]
                ),
            },
            "pre_enrollment_base_accuracy": {
                "observed": float(
                    primary_pre["pre_enrollment_base_accuracy"].mean()
                ),
                "threshold": float(
                    criterion["pre_enrollment_base_accuracy_mean_at_least"]
                ),
                "passed": float(
                    primary_pre["pre_enrollment_base_accuracy"].mean()
                )
                >= float(criterion["pre_enrollment_base_accuracy_mean_at_least"]),
            },
            "cuda_evidence": {
                "observed": int(audit["cuda_units"]),
                "threshold": int(len(units)),
                "passed": int(audit["cuda_units"]) == int(len(units)),
            },
            "metric_reconstruction": {
                "observed": bool(reconstruction["passed"]),
                "threshold": True,
                "passed": bool(reconstruction["passed"]),
            },
        }
        success = {
            "checks": success_checks,
            "passed": all(item["passed"] for item in success_checks.values()),
        }
    report_lines = [
        (
            "# Confirmatory fixed-base-memory enrollment replication"
            if ACTIVE_STUDY == "confirmatory"
            else "# Exploratory fixed-base-memory enrollment study"
        ),
        "",
        (
            "The primary results below use only the four split seeds that had not been evaluated with fixed base memory when the protocol was frozen. Seed 42 appears only in the combined tables."
            if ACTIVE_STUDY == "confirmatory"
            else "This study was designed after the frozen confirmatory analysis. It is a lifecycle-framing sensitivity, not a replacement for the primary result."
        ),
        "",
        "## Protocol",
        "",
        "For every held-out fault pair, six base classes receive a fixed memory of 20 group-distinct examples each (120 total). That memory is selected once and reused for every shot count and support draw. Enrollment appends only 1, 3, or 5 examples from each of the two new faults. Three deterministic context memories are ensembled. The query and novel-fault supports exactly match the frozen summary-only study.",
        "",
        "## Results",
        "",
    ]
    for shots in (1, 3, 5):
        row = h_lookup[(report_cohort, "cached_base_tabpfn_v2", shots)]
        comp = comparison_lookup[(report_cohort, shots)]
        svm_comp = cached_comparison_lookup[(report_cohort, shots)]
        stable = stability_lookup[(report_cohort, "cached_base_tabpfn_v2", shots)]
        report_lines.append(
            f"- Cached-base TabPFN, {shots}-shot: H={row.mean:.4f} (95% CI {row.ci_low:.4f}-{row.ci_high:.4f}); {stable.fraction_h_ge_0_90:.1%} of pair/seed units reached H>=0.90 and {stable.fraction_h_ge_0_95:.1%} reached H>=0.95; minimum={stable.minimum_h:.4f}; difference versus frozen balanced context={comp.mean_difference:+.4f}, Holm p={comp.holm_adjusted_p:.5f}; difference versus cached-base SVM={svm_comp.mean_difference:+.4f}, Holm p={svm_comp.holm_adjusted_p:.5f}."
        )
    report_lines.extend(
        [
            f"- Primary-cohort pre-enrollment base accuracy: mean {primary_pre['pre_enrollment_base_accuracy'].mean():.4f}, minimum {primary_pre['pre_enrollment_base_accuracy'].min():.4f} across {len(primary_pre)} pair/seed units.",
            f"- Five-shot enrolled-fault recall: mean range {five_fault['mean'].min():.4f}–{five_fault['mean'].max():.4f}; minimum observed pair-level recall {five_fault['min'].min():.4f}.",
            f"- Mean within-draw H standard deviation across the three fixed base memories at five shots: {float(context_summary.loc[context_summary['shots'] == 5, 'mean_context_std'].iloc[0]):.4f}.",
            "",
            "## Interpretation",
            "",
            config["interpretation"],
            "",
            "## Frozen success criteria",
            "",
            (
                f"Overall pass={success['passed']}. "
                + "; ".join(
                    f"{name}: observed={item['observed']}, threshold={item['threshold']}, pass={item['passed']}"
                    for name, item in success["checks"].items()
                )
                if success is not None
                else "No confirmatory success rule applies to the exploratory pilot."
            ),
            "",
            "## Validity checks",
            "",
            f"- {len(units)}/{expected_units} units validated by artifact hashes.",
            f"- {reconstruction['rows_reconstructed']} metric rows independently reconstructed; pass={reconstruction['passed']}.",
            f"- Query/support identity with frozen comparator: {audit['matched_frozen_query_and_support_units']}/{len(units)} units.",
            "- Every neural prediction was produced on CUDA and records positive CUDA allocation.",
            "",
        ]
    )
    (PILOT_ROOT / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    result = {
        "schema_version": 1,
        "pilot_config_sha256": CONFIG_SHA256,
        "units": len(units),
        "rows": len(rows),
        "reconstruction": reconstruction,
        "group_and_comparator_audit": audit,
        "bootstrap_iterations": bootstrap_iterations,
        "sign_flip_iterations": sign_flip_iterations,
        "plots": plots,
        "pre_enrollment_base_accuracy_mean": float(pre["pre_enrollment_base_accuracy"].mean()),
        "pre_enrollment_base_accuracy_minimum": float(pre["pre_enrollment_base_accuracy"].min()),
        "cohorts": {
            name: {
                "pair_seed_units": int(
                    frame[["pair", "seed"]].drop_duplicates().shape[0]
                ),
                "seeds": sorted(int(value) for value in frame["seed"].unique()),
            }
            for name, frame in cohort_pair_units.items()
        },
        "frozen_success_criteria": success,
        "completed_at": utc_now(),
    }
    atomic_json(PILOT_ROOT / "analysis_manifest.json", result)
    if ACTIVE_STUDY == "confirmatory":
        state_path = PILOT_ROOT / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "complete" if success and success["passed"] else "analyzed"
        state["analysis_passed"] = bool(success and success["passed"])
        state["combined_validated_units"] = len(units)
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
    return result


def interim_review() -> dict[str, Any]:
    """Audit completed replication units without opening the final analysis gate."""
    if ACTIVE_STUDY != "confirmatory":
        raise RuntimeError("Interim review is defined only for the confirmatory study.")
    config = load_config()
    all_units = discover_units()
    fresh_units = [unit for unit in all_units if CONFIRMATORY_ROOT in unit.parents]
    observed_units = [unit for unit in all_units if EXPLORATORY_ROOT in unit.parents]
    if not fresh_units:
        raise RuntimeError("No completed fresh replication units are available.")

    all_rows, all_originals = _load_rows(all_units)
    fresh_rows, fresh_originals = _load_rows(fresh_units)
    reconstruction = reconstruct_metrics(all_originals)
    audit = _audit_manifests(all_units)
    if not reconstruction["passed"] or not audit["passed"]:
        raise AssertionError("Interim evidence reconstruction or manifest audit failed.")

    pair_seed = (
        fresh_rows.groupby(
            ["pair", "fault_a", "fault_b", "seed", "method", "shots"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    summary = (
        pair_seed.groupby(["method", "shots"], as_index=False)
        .agg(
            pair_seed_units=("pair", "size"),
            harmonic_mean=("harmonic_mean", "mean"),
            base_accuracy=("base_accuracy", "mean"),
            enrolled_accuracy=("enrolled_accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            worst_enrolled_recall=("worst_enrolled_recall", "mean"),
            minimum_h=("harmonic_mean", "min"),
            median_h=("harmonic_mean", "median"),
            fraction_h_ge_0_90=("harmonic_mean", lambda value: float(np.mean(value >= 0.90))),
            fraction_h_ge_0_95=("harmonic_mean", lambda value: float(np.mean(value >= 0.95))),
            nll=("nll", "mean"),
            brier=("brier", "mean"),
            ece_15=("ece_15", "mean"),
        )
    )
    per_seed = (
        pair_seed.groupby(["method", "shots", "seed"], as_index=False)
        .agg(
            pair_seed_units=("pair", "size"),
            harmonic_mean=("harmonic_mean", "mean"),
            base_accuracy=("base_accuracy", "mean"),
            enrolled_accuracy=("enrolled_accuracy", "mean"),
            minimum_h=("harmonic_mean", "min"),
        )
    )

    frozen = _frozen_comparator()
    fresh_keys = fresh_rows[["pair", "seed"]].drop_duplicates()
    frozen_fresh = frozen.merge(fresh_keys, on=["pair", "seed"], how="inner")
    cached = fresh_rows[
        fresh_rows["method"] == "cached_base_tabpfn_v2"
    ]
    cached_svm = fresh_rows[
        fresh_rows["method"] == "cached_base_linear_svm"
    ]
    comparisons: list[dict[str, Any]] = []
    for shots in sorted(fresh_rows["shots"].unique()):
        cached_unit = (
            cached[cached["shots"] == shots]
            .groupby(["pair", "seed"])["harmonic_mean"]
            .mean()
        )
        frozen_unit = (
            frozen_fresh[frozen_fresh["shots"] == shots]
            .groupby(["pair", "seed"])["harmonic_mean"]
            .mean()
        )
        svm_unit = (
            cached_svm[cached_svm["shots"] == shots]
            .groupby(["pair", "seed"])["harmonic_mean"]
            .mean()
        )
        comparisons.append(
            {
                "shots": int(shots),
                "matched_pair_seed_units": int(len(cached_unit)),
                "cached_tabpfn_minus_balanced_tabpfn": float(
                    (cached_unit - frozen_unit).mean()
                ),
                "cached_tabpfn_minus_cached_svm": float(
                    (cached_unit - svm_unit).mean()
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)

    fault_draw = _per_fault(fresh_rows, fresh_originals)
    fault_units = (
        fault_draw.groupby(
            ["pair", "seed", "method", "shots", "class_id"], as_index=False
        )["recall"]
        .mean()
    )
    fault_summary = (
        fault_units.groupby(["method", "shots", "class_id"], as_index=False)
        .agg(
            pair_seed_units=("recall", "size"),
            recall_mean=("recall", "mean"),
            recall_minimum=("recall", "min"),
        )
    )

    pre_rows = []
    context_rows = []
    for unit in fresh_units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pre_rows.append(
            {
                "pair": "-".join(str(value) for value in metrics["pair"]),
                "seed": int(metrics["seed"]),
                "accuracy": float(metrics["pre_enrollment"]["accuracy"]),
                "balanced_accuracy": float(
                    metrics["pre_enrollment"]["balanced_accuracy"]
                ),
                "duration_seconds": float(metrics["duration_seconds"]),
                "peak_cuda_allocated_bytes": int(
                    metrics["cuda_diagnostics"]["peak_allocated_bytes"]
                ),
            }
        )
        context_rows.extend(
            {
                "pair": "-".join(str(value) for value in metrics["pair"]),
                "seed": int(metrics["seed"]),
                **row,
            }
            for row in metrics["context_sensitivity_rows"]
        )
    pre = pd.DataFrame(pre_rows)
    context = pd.DataFrame(context_rows)
    context_within = (
        context.groupby(["pair", "seed", "shots", "draw"])["harmonic_mean"]
        .std()
        .rename("context_h_std")
        .reset_index()
    )
    context_summary = (
        context_within.groupby("shots", as_index=False)
        .agg(
            pair_seed_draw_units=("pair", "size"),
            mean_context_h_std=("context_h_std", "mean"),
            maximum_context_h_std=("context_h_std", "max"),
        )
    )
    expected_fresh_seeds = {
        int(value) for value in config["execution_seeds"]
    }
    coverage_frame = pair_seed[
        (pair_seed["method"] == "cached_base_tabpfn_v2")
        & (pair_seed["shots"] == int(config["shots"][0]))
    ]
    pair_seed_sets = {
        pair: {int(value) for value in part["seed"]}
        for pair, part in coverage_frame.groupby("pair")
    }
    completed_pairs = sorted(
        pair
        for pair, seeds in pair_seed_sets.items()
        if seeds == expected_fresh_seeds
    )
    partially_completed_pairs = sorted(
        pair
        for pair, seeds in pair_seed_sets.items()
        if seeds and seeds != expected_fresh_seeds
    )
    expected_pairs = [
        f"{int(pair[0])}-{int(pair[1])}" for pair in config["pairs"]
    ]
    missing_pairs = [pair for pair in expected_pairs if pair not in pair_seed_sets]
    not_fully_completed_pairs = [
        pair for pair in expected_pairs if pair not in completed_pairs
    ]
    fault_pair_counts = {
        str(fault): sum(
            fault in {int(value) for value in pair.split("-")}
            for pair in completed_pairs
        )
        for fault in range(1, 8)
    }
    incomplete_dirs = [
        str(path)
        for path in (CONFIRMATORY_ROOT / "units").glob("pair_*/seed_*")
        if not (path / "manifest.json").is_file()
    ]
    registry_path = CONFIRMATORY_ROOT / "experiment_registry.jsonl"
    registry_events = (
        [
            json.loads(line)
            for line in registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if registry_path.is_file()
        else []
    )
    terminal_events: dict[str, list[dict[str, Any]]] = {}
    for event in registry_events:
        terminal_events.setdefault(str(event.get("run_id")), []).append(event)
    unresolved_starts = []
    for run_id, events in terminal_events.items():
        starts = [event for event in events if event.get("event") == "started"]
        completions = [event for event in events if event.get("event") == "completed"]
        failures = [event for event in events if event.get("event") == "failed"]
        # A prior forced pause may leave an old unmatched start even after the
        # unit is later rerun successfully. Only the latest terminal state is
        # operationally unresolved.
        if events[-1].get("event") == "started":
            unresolved_starts.append(
                {
                    "run_id": run_id,
                    "start_events": len(starts),
                    "completion_events": len(completions),
                    "failure_events": len(failures),
                    "last_start_timestamp": starts[-1].get("timestamp"),
                }
            )

    table_root = CONFIRMATORY_ROOT / "interim_tables"
    table_root.mkdir(parents=True, exist_ok=True)
    pair_seed.to_csv(table_root / "pair_seed_units.csv", index=False)
    summary.to_csv(table_root / "summary.csv", index=False)
    per_seed.to_csv(table_root / "per_seed.csv", index=False)
    comparison_frame.to_csv(table_root / "descriptive_comparisons.csv", index=False)
    pre.to_csv(table_root / "pre_enrollment_and_cuda.csv", index=False)
    fault_units.to_csv(table_root / "per_fault_pair_seed.csv", index=False)
    fault_summary.to_csv(table_root / "per_fault_summary.csv", index=False)
    context_summary.to_csv(table_root / "context_sensitivity_summary.csv", index=False)
    context_within.to_csv(
        table_root / "context_sensitivity_pair_seed_draw.csv", index=False
    )

    tabpfn_summary = summary[summary["method"] == "cached_base_tabpfn_v2"]
    tabpfn_fault_five = fault_summary[
        (fault_summary["method"] == "cached_base_tabpfn_v2")
        & (fault_summary["shots"] == 5)
    ]
    partial_clause = (
        f", with partial coverage for {len(partially_completed_pairs)} additional pair(s)"
        if partially_completed_pairs
        else ""
    )
    report = [
        "# Interim integrity and result review",
        "",
        f"This report was generated while the confirmatory matrix was paused. It is descriptive only: {len(completed_pairs)}/21 held-out pairs are complete across all four fresh seeds{partial_clause}, so this is not a representative substitute for the frozen 84-unit replication cohort.",
        "",
        "## Execution integrity",
        "",
        f"- Completed fresh units: {len(fresh_units)}/84.",
        f"- Reused exploratory seed-42 units available for combined sensitivity: {len(observed_units)}/21.",
        f"- Result rows reconstructed across completed fresh and observed units: {reconstruction['rows_reconstructed']}; exact reconstruction pass={reconstruction['passed']}.",
        f"- CUDA units with positive allocation: {audit['cuda_units']}/{len(all_units)}.",
        f"- Query/support cohorts exactly matched to the frozen comparator: {audit['matched_frozen_query_and_support_units']}/{len(all_units)}.",
        f"- Manifest/group/pre-enrollment audit failures: {len(audit['failures'])}.",
        f"- Incomplete directories ignored on resume: {len(incomplete_dirs)}.",
        f"- Runs with an unmatched start event after forced pause: {len(unresolved_starts)}; these are operational interruptions and are not counted as completed evidence.",
        "",
        "## Descriptive fresh-unit results",
        "",
    ]
    for row in tabpfn_summary.itertuples(index=False):
        comparison = comparison_frame[comparison_frame["shots"] == row.shots].iloc[0]
        report.append(
            f"- {int(row.shots)} shot: H={row.harmonic_mean:.4f}, base={row.base_accuracy:.4f}, enrolled={row.enrolled_accuracy:.4f}, minimum pair/seed H={row.minimum_h:.4f}, H>=0.95 fraction={row.fraction_h_ge_0_95:.1%}, delta versus matched balanced-context TabPFN={comparison.cached_tabpfn_minus_balanced_tabpfn:+.4f}, delta versus cached SVM={comparison.cached_tabpfn_minus_cached_svm:+.4f}."
        )
    if any("svm_solver_policy" in json.loads((unit / "metrics.json").read_text(encoding="utf-8")) for unit in fresh_units):
        report.append(
            "- SVM deltas are provisional after the disclosed solver-watchdog amendment and require a uniform bounded-SVM replay before final reporting; TabPFN estimates are unaffected."
        )
    report.extend(
        [
            "",
            "## Fault and context checks",
            "",
            "- Five-shot mean enrolled recall by currently covered fault: "
            + ", ".join(
                f"fault {int(row.class_id)}={row.recall_mean:.4f}"
                for row in tabpfn_fault_five.itertuples(index=False)
            )
            + ".",
            "- Mean H standard deviation across the three fixed base memories: "
            + ", ".join(
                f"{int(row.shots)} shot={row.mean_context_h_std:.4f}"
                for row in context_summary.itertuples(index=False)
            )
            + ".",
            f"- Completed pairs: {', '.join(completed_pairs)}.",
            f"- Partially completed pairs: {', '.join(partially_completed_pairs) if partially_completed_pairs else 'none'}.",
            f"- Pairs with no completed fresh unit: {', '.join(missing_pairs)}.",
            f"- Pairs not yet complete across all four fresh seeds: {', '.join(not_fully_completed_pairs)}.",
            "- Completed-pair representation by fault: "
            + ", ".join(
                f"fault {fault}={count} pairs"
                for fault, count in fault_pair_counts.items()
            )
            + ".",
            "- Largest single draw-level context H standard deviation: "
            f"{context_within['context_h_std'].max():.4f} across all shots and {context_within.loc[context_within['shots'] == 5, 'context_h_std'].max():.4f} at five shots; these are local sensitivity cases, while the five-shot mean across all completed pair/seed/draw units is {float(context_summary.loc[context_summary['shots'] == 5, 'mean_context_h_std'].iloc[0]):.4f}.",
            "",
            "## Interim judgment",
            "",
            "The implementation is working as designed if the machine-readable audit passes. The accuracy values above are useful for detecting catastrophic regressions, but they must not be promoted to final estimates because fault-pair coverage is incomplete and the remaining late-index cross-pairs underrepresent faults 5-7 in the current subset.",
            "",
        ]
    )
    (CONFIRMATORY_ROOT / "INTERIM_REVIEW.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    result = {
        "schema_version": 1,
        "reviewed_at": utc_now(),
        "status": "paused_interim_review_only",
        "fresh_completed_units": len(fresh_units),
        "fresh_expected_units": 84,
        "observed_seed42_units": len(observed_units),
        "all_discovered_units": len(all_units),
        "fresh_rows": len(fresh_rows),
        "all_rows": len(all_rows),
        "reconstruction": reconstruction,
        "manifest_audit": audit,
        "incomplete_unvalidated_directories": incomplete_dirs,
        "unresolved_registry_starts": unresolved_starts,
        "interim_inference_allowed": False,
        "coverage_warning": "Completed fresh pairs are not balanced across enrolled faults; do not use this interim subset for confirmatory claims.",
        "coverage": {
            "completed_pairs": completed_pairs,
            "partially_completed_pairs": partially_completed_pairs,
            "missing_pairs": missing_pairs,
            "not_fully_completed_pairs": not_fully_completed_pairs,
            "completed_pair_count_by_fault": fault_pair_counts,
        },
    }
    atomic_json(CONFIRMATORY_ROOT / "INTERIM_REVIEW.json", result)
    return result


def replay_uniform_bounded_svm(*, sign_flip_iterations: int = 20000) -> dict[str, Any]:
    """Rebuild the secondary SVM control under one uniform bounded policy.

    The max-iteration guard was added after an unlimited libsvm fit stalled.
    TabPFN is the frozen primary method and is not rerun here.  This audit
    deterministically reconstructs every SVM context, checks its query,
    support, and base-memory identities against the immutable unit artifacts,
    and compares the replay probabilities with the originally persisted SVM
    evidence before allowing the secondary comparison to be interpreted.
    """
    if ACTIVE_STUDY != "confirmatory":
        raise RuntimeError("Uniform SVM replay is defined for the confirmatory study.")

    config = load_config()
    expected_units = len(config["pairs"]) * len(config["seeds"])
    final_analysis_path = CONFIRMATORY_ROOT / "analysis_manifest.json"
    if not final_analysis_path.is_file():
        raise RuntimeError("Run the final validated analysis before replaying the SVM.")
    final_analysis = json.loads(final_analysis_path.read_text(encoding="utf-8"))
    if not (
        final_analysis.get("units") == expected_units
        and final_analysis.get("reconstruction", {}).get("passed") is True
        and final_analysis.get("group_and_comparator_audit", {}).get("passed") is True
        and final_analysis.get("frozen_success_criteria", {}).get("passed") is True
    ):
        raise RuntimeError("The prerequisite final artifact audit is incomplete or failed.")
    units = sorted(
        [
            *CONFIRMATORY_ROOT.glob("units/pair_*/seed_*"),
            *EXPLORATORY_ROOT.glob(
                f"units/pair_*/seed_{int(config['previously_observed_seed'])}"
            ),
        ],
        key=str,
    )
    units = [
        unit
        for unit in units
        if (unit / "manifest.json").is_file()
        and (unit / "metrics.json").is_file()
        and (unit / "prediction_evidence.npz").is_file()
    ]
    if len(units) != expected_units:
        raise RuntimeError(
            f"Expected {expected_units} validated units before replay, found {len(units)}."
        )

    replay_root = CONFIRMATORY_ROOT / "uniform_bounded_svm_replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = file_sha256(DATA_PATH)
    scalar_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "base_accuracy",
        "enrolled_accuracy",
        "harmonic_mean",
        "worst_enrolled_recall",
        "normal_far_after_enrollment",
        "nll",
        "brier",
        "ece_15",
    )
    partial_state_path = replay_root / "partial_state.json"
    partial_rows_path = replay_root / "partial_per_draw.csv"
    partial_diagnostics_path = replay_root / "partial_solver_diagnostics.csv"
    if partial_state_path.is_file():
        partial = json.loads(partial_state_path.read_text(encoding="utf-8"))
        if (
            partial.get("config_sha256") != CONFIG_SHA256
            or partial.get("dataset_sha256") != dataset_sha256
        ):
            raise RuntimeError("Existing SVM replay checkpoint has incompatible inputs.")
        completed_run_ids = set(partial.get("completed_run_ids", []))
        partial_rows = pd.read_csv(partial_rows_path)
        partial_diagnostics = pd.read_csv(partial_diagnostics_path)
        rows = partial_rows[
            partial_rows["run_id"].isin(completed_run_ids)
        ].to_dict("records")
        diagnostics = partial_diagnostics[
            partial_diagnostics["run_id"].isin(completed_run_ids)
        ].to_dict("records")
        identity_failures = list(partial.get("identity_failures", []))
        maximum_probability_difference = float(
            partial.get("maximum_probability_difference", 0.0)
        )
        changed_prediction_rows = int(partial.get("changed_prediction_rows", 0))
        changed_predictions = int(partial.get("changed_predictions", 0))
        original_solver_policy_counts = dict(
            partial.get("original_solver_policy_counts", {})
        )
        maximum_metric_difference = {
            name: float(partial.get("maximum_metric_difference", {}).get(name, 0.0))
            for name in scalar_names
        }
    else:
        completed_run_ids: set[str] = set()
        rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        identity_failures: list[dict[str, Any]] = []
        maximum_probability_difference = 0.0
        changed_prediction_rows = 0
        changed_predictions = 0
        original_solver_policy_counts: dict[str, int] = {}
        maximum_metric_difference = {name: 0.0 for name in scalar_names}
    frame = pd.read_csv(DATA_PATH)

    for unit_index, unit in enumerate(units, start=1):
        saved = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = tuple(int(value) for value in saved["pair"])
        seed = int(saved["seed"])
        run_id = str(saved["run_id"])
        if run_id in completed_run_ids:
            print(
                f"[uniform SVM replay {unit_index}/{len(units)} cached] "
                f"pair={pair[0]}-{pair[1]} seed={seed}",
                flush=True,
            )
            continue
        policy = saved.get("svm_solver_policy", {})
        policy_name = (
            f"max_iter={policy.get('max_iter')}"
            if policy
            else "legacy_unlimited"
        )
        original_solver_policy_counts[policy_name] = (
            original_solver_policy_counts.get(policy_name, 0) + 1
        )
        print(
            f"[uniform SVM replay {unit_index}/{len(units)}] "
            f"pair={pair[0]}-{pair[1]} seed={seed}",
            flush=True,
        )

        fold = fit_lifecycle_fold(frame, holdout=pair, seed=seed, regime="summary_only")
        batches = fold.batches
        train_x = _features(batches["train"], "summary_only")
        train_y = batches["train"].labels.numpy()
        train_groups = batches["train"].group_ids
        reference_x = _features(batches["reference_pool"], "summary_only")
        reference_frame = fold.split.reference_pool
        outer_x_all = np.vstack(
            (
                _features(batches["seen_test"], "summary_only"),
                _features(batches["query"], "summary_only"),
            )
        )
        outer_y_all = np.r_[
            batches["seen_test"].labels.numpy(),
            batches["query"].labels.numpy(),
        ]
        outer_groups = (*batches["seen_test"].group_ids, *batches["query"].group_ids)
        query_indices = _balanced_query_indices(
            outer_y_all,
            outer_groups,
            per_class=int(config["query_examples_per_class"]),
        )
        query_x = outer_x_all[query_indices]
        query_y = outer_y_all[query_indices]
        query_groups = tuple(outer_groups[index] for index in query_indices)
        base_ids = tuple(sorted(int(value) for value in np.unique(train_y)))

        query_manifest = json.loads(
            (unit / "query_manifest.json").read_text(encoding="utf-8")
        )
        if (
            list(query_groups) != query_manifest["groups"]
            or query_y.astype(int).tolist() != query_manifest["labels"]
        ):
            identity_failures.append({"run_id": run_id, "artifact": "query"})

        context_seeds = tuple(int(value) for value in config["context_seeds"])
        base_count = int(config["base_context_per_class"])
        base_memories: list[dict[str, Any]] = []
        replay_context_manifest: list[dict[str, Any]] = []
        for context_seed in context_seeds:
            indices = _base_memory_indices(
                train_y,
                train_groups,
                base_ids=base_ids,
                count=base_count,
                context_seed=context_seed,
                namespace=str(config["base_context_namespace"]),
            )
            groups = tuple(train_groups[index] for index in indices)
            base_memories.append(
                {
                    "context_seed": context_seed,
                    "features": train_x[indices],
                    "labels": train_y[indices],
                    "groups": groups,
                }
            )
            replay_context_manifest.append(
                {
                    "context_seed": context_seed,
                    "fixed_across_shots_and_draws": True,
                    "examples_per_base_class": base_count,
                    "base_groups_by_class": {
                        str(class_id): [
                            groups[index]
                            for index in np.flatnonzero(train_y[indices] == class_id)
                        ]
                        for class_id in base_ids
                    },
                }
            )
        saved_context_manifest = json.loads(
            (unit / "context_manifest.json").read_text(encoding="utf-8")
        )
        if replay_context_manifest != saved_context_manifest:
            identity_failures.append({"run_id": run_id, "artifact": "context"})

        saved_support_manifest = json.loads(
            (unit / "support_manifest.json").read_text(encoding="utf-8")
        )
        replay_support_manifest: list[dict[str, Any]] = []
        evidence = np.load(unit / "prediction_evidence.npz", allow_pickle=False)
        evidence_row_lookup = {
            str(row_id): index for index, row_id in enumerate(evidence["row_ids"])
        }
        original_row_lookup = {
            (int(row["shots"]), int(row["draw"])): row
            for row in saved["rows"]
            if row["method"] == "cached_base_linear_svm"
        }

        for shots, draw in itertools.product(
            (int(value) for value in config["shots"]),
            range(int(config["draws"])),
        ):
            selected = deterministic_support_indices(
                reference_frame,
                class_ids=pair,
                shots=shots,
                seed=seed,
                draw=draw,
                namespace=str(config["support_namespace"]),
            )
            positions = reference_frame.index.get_indexer(selected)
            if np.any(positions < 0):
                raise AssertionError("Replay support rows did not map to the reference pool.")
            support_x = reference_x[positions]
            support_y = reference_frame.loc[selected, "Class"].to_numpy(dtype=int)
            support_groups = tuple(
                reference_frame.loc[selected, "_input_group"].astype(str)
            )
            replay_support_manifest.append(
                {
                    "shots": shots,
                    "draw": draw,
                    "groups_by_class": {
                        str(class_id): [
                            support_groups[index]
                            for index in np.flatnonzero(support_y == class_id)
                        ]
                        for class_id in pair
                    },
                    "query_used": False,
                }
            )

            context_probabilities: list[np.ndarray] = []
            elapsed_seconds = 0.0
            for memory in base_memories:
                started = time.perf_counter()
                probability, diagnostic = _bounded_linear_svm_probability(
                    np.vstack((memory["features"], support_x)),
                    np.r_[memory["labels"], support_y],
                    query_x,
                )
                fit_seconds = time.perf_counter() - started
                elapsed_seconds += fit_seconds
                context_probabilities.append(probability)
                diagnostics.append(
                    {
                        "run_id": run_id,
                        "pair": f"{pair[0]}-{pair[1]}",
                        "fault_a": pair[0],
                        "fault_b": pair[1],
                        "seed": seed,
                        "shots": shots,
                        "draw": draw,
                        "context_seed": int(memory["context_seed"]),
                        "elapsed_seconds": fit_seconds,
                        **diagnostic,
                    }
                )

            mean_probability = np.mean(context_probabilities, axis=0)
            mean_probability /= mean_probability.sum(1, keepdims=True)
            metric = metric_row(
                labels=query_y,
                probability=mean_probability,
                base_class_ids=base_ids,
                enrolled_class_ids=pair,
                method="cached_base_linear_svm_uniform_bounded",
                shots=shots,
                draw=draw,
                elapsed_seconds=elapsed_seconds,
                probability_source="fixed_softmax_score_ensemble",
                extra={
                    "base_class_ids": list(base_ids),
                    "enrolled_class_ids": list(pair),
                    "ensemble_contexts": len(context_seeds),
                    "base_context_examples": len(base_ids) * base_count,
                    "appended_novel_examples": len(pair) * shots,
                    "context_examples": len(base_ids) * base_count + len(pair) * shots,
                    "query_examples": len(query_y),
                },
            )

            evidence_id = f"cached_base_linear_svm|shot={shots}|draw={draw}"
            original_probability = evidence["probabilities"][
                evidence_row_lookup[evidence_id]
            ]
            probability_difference = float(
                np.max(np.abs(mean_probability - original_probability))
            )
            maximum_probability_difference = max(
                maximum_probability_difference, probability_difference
            )
            changed = int(
                np.sum(
                    mean_probability.argmax(1)
                    != original_probability.argmax(1)
                )
            )
            changed_predictions += changed
            changed_prediction_rows += int(changed > 0)
            original_metric = original_row_lookup[(shots, draw)]
            for name in scalar_names:
                maximum_metric_difference[name] = max(
                    maximum_metric_difference[name],
                    abs(float(metric[name]) - float(original_metric[name])),
                )
            rows.append(
                {
                    "run_id": run_id,
                    "pair": f"{pair[0]}-{pair[1]}",
                    "fault_a": pair[0],
                    "fault_b": pair[1],
                    "seed": seed,
                    "cohort": (
                        "combined_seed42_sensitivity"
                        if seed == int(config["previously_observed_seed"])
                        else "primary_replication"
                    ),
                    "shots": shots,
                    "draw": draw,
                    **{name: float(metric[name]) for name in scalar_names},
                    "elapsed_seconds": elapsed_seconds,
                    "maximum_probability_difference_vs_original": probability_difference,
                    "changed_predictions_vs_original": changed,
                }
            )

        evidence.close()
        if replay_support_manifest != saved_support_manifest:
            identity_failures.append({"run_id": run_id, "artifact": "support"})
        completed_run_ids.add(run_id)
        rows_temp = partial_rows_path.with_suffix(".tmp")
        diagnostics_temp = partial_diagnostics_path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(rows_temp, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_temp, index=False)
        rows_temp.replace(partial_rows_path)
        diagnostics_temp.replace(partial_diagnostics_path)
        atomic_json(
            partial_state_path,
            {
                "schema_version": 1,
                "updated_at": utc_now(),
                "config_sha256": CONFIG_SHA256,
                "dataset_sha256": dataset_sha256,
                "completed_run_ids": sorted(completed_run_ids),
                "identity_failures": identity_failures,
                "maximum_probability_difference": maximum_probability_difference,
                "changed_prediction_rows": changed_prediction_rows,
                "changed_predictions": changed_predictions,
                "original_solver_policy_counts": original_solver_policy_counts,
                "maximum_metric_difference": maximum_metric_difference,
            },
        )

    per_draw = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    group_keys = ["pair", "fault_a", "fault_b", "seed", "cohort", "shots"]
    pair_units = per_draw.groupby(group_keys, as_index=False)[list(scalar_names)].mean()
    summary_rows: list[dict[str, Any]] = []
    cohort_frames = {
        "primary_replication": pair_units[pair_units["cohort"] == "primary_replication"],
        "combined_five_seed": pair_units,
    }
    for cohort, cohort_frame in cohort_frames.items():
        for shots, part in cohort_frame.groupby("shots", sort=True):
            summary_rows.append(
                {
                    "cohort": cohort,
                    "shots": int(shots),
                    "pair_seed_units": len(part),
                    "mean_harmonic_mean": float(part["harmonic_mean"].mean()),
                    "std_harmonic_mean": float(part["harmonic_mean"].std()),
                    "minimum_harmonic_mean": float(part["harmonic_mean"].min()),
                    "mean_base_accuracy": float(part["base_accuracy"].mean()),
                    "mean_enrolled_accuracy": float(part["enrolled_accuracy"].mean()),
                    "fraction_h_ge_0_95": float((part["harmonic_mean"] >= 0.95).mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    original_rows, _ = _load_rows(units)
    tabpfn_units = (
        original_rows[original_rows["method"] == "cached_base_tabpfn_v2"]
        .groupby(["pair", "fault_a", "fault_b", "seed", "shots"], as_index=False)[
            "harmonic_mean"
        ]
        .mean()
    )
    comparisons: list[dict[str, Any]] = []
    for cohort, cohort_svm in cohort_frames.items():
        cohort_tabpfn = tabpfn_units[
            tabpfn_units["seed"].isin(cohort_svm["seed"].unique())
        ]
        for shots in sorted(cohort_svm["shots"].unique()):
            comparison = paired_sign_flip(
                cohort_tabpfn[cohort_tabpfn["shots"] == shots],
                cohort_svm[cohort_svm["shots"] == shots],
                value="harmonic_mean",
                iterations=sign_flip_iterations,
                seed=20261201 + int(shots),
            )
            comparisons.append({"cohort": cohort, "shots": int(shots), **comparison})
    comparison_frame = pd.DataFrame(comparisons)
    for cohort in comparison_frame["cohort"].unique():
        selected = comparison_frame["cohort"] == cohort
        comparison_frame.loc[selected, "holm_adjusted_p"] = holm_adjust(
            comparison_frame.loc[selected, "two_sided_permutation_p"].tolist()
        )

    per_draw.to_csv(replay_root / "per_draw.csv", index=False)
    diagnostic_frame.to_csv(replay_root / "solver_diagnostics.csv", index=False)
    pair_units.to_csv(replay_root / "pair_units.csv", index=False)
    summary.to_csv(replay_root / "summary.csv", index=False)
    comparison_frame.to_csv(
        replay_root / "paired_tabpfn_vs_uniform_bounded_svm.csv", index=False
    )
    nonconverged = int((~diagnostic_frame["converged"]).sum())
    expected_per_draw_rows = (
        expected_units * len(config["shots"]) * int(config["draws"])
    )
    expected_solver_contexts = expected_per_draw_rows * len(config["context_seeds"])
    matrix_complete = (
        len(completed_run_ids) == expected_units
        and len(per_draw) == expected_per_draw_rows
        and len(diagnostic_frame) == expected_solver_contexts
    )
    result = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "purpose": "Uniform post-freeze replay of the secondary SVM control only; frozen TabPFN results are unchanged.",
        "config_sha256": CONFIG_SHA256,
        "dataset_sha256": dataset_sha256,
        "units": len(units),
        "per_draw_rows": len(per_draw),
        "solver_contexts": len(diagnostic_frame),
        "matrix_complete": matrix_complete,
        "solver_policy": {
            "solver": "sklearn.svm.SVC",
            "kernel": "linear",
            "max_iter": SVM_MAX_ITER,
        },
        "identity_audit": {
            "failures": identity_failures,
            "passed": not identity_failures,
        },
        "comparison_to_persisted_mixed_policy_evidence": {
            "original_solver_policy_unit_counts": original_solver_policy_counts,
            "maximum_probability_difference": maximum_probability_difference,
            "changed_prediction_rows": changed_prediction_rows,
            "changed_predictions": changed_predictions,
            "maximum_metric_difference": maximum_metric_difference,
            "hard_predictions_identical": changed_predictions == 0,
        },
        "convergence": {
            "converged_contexts": int(diagnostic_frame["converged"].sum()),
            "nonconverged_contexts": nonconverged,
            "nonconverged_fraction": float(nonconverged / len(diagnostic_frame)),
            "maximum_binary_iterations": int(
                diagnostic_frame["maximum_binary_iterations"].max()
            ),
            "maximum_fit_seconds": float(diagnostic_frame["elapsed_seconds"].max()),
        },
        "tables": {
            path.name: file_sha256(path)
            for path in sorted(replay_root.glob("*.csv"))
        },
        "passed": matrix_complete and not identity_failures and changed_predictions == 0,
    }
    atomic_json(replay_root / "replay_manifest.json", result)
    return result


def _parse_pair(value: str) -> tuple[int, int]:
    parts = tuple(int(item) for item in value.replace("_", "-").split("-"))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Pair must look like 1-2.")
    return parts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("pilot", "confirmatory"),
        default="pilot",
        help="Select the immutable exploratory or multi-seed study root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("review")
    unit = subparsers.add_parser("unit")
    unit.add_argument("--pair", type=_parse_pair, required=True)
    unit.add_argument("--seed", type=int, default=42)
    unit.add_argument("--device", default="cuda:0")
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--pair-limit", type=int)
    matrix.add_argument("--device", default="cuda:0")
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--expected-units", type=int)
    analysis.add_argument("--bootstrap-iterations", type=int, default=5000)
    analysis.add_argument("--sign-flip-iterations", type=int, default=20000)
    replay_svm = subparsers.add_parser("replay-svm")
    replay_svm.add_argument("--sign-flip-iterations", type=int, default=20000)
    args = parser.parse_args(argv)
    activate_study(args.study)

    if args.command == "freeze":
        result = freeze_source_snapshot()
    elif args.command == "review":
        result = interim_review()
    elif args.command == "unit":
        result = run_unit(
            frame=pd.read_csv(DATA_PATH),
            pair=args.pair,
            seed=args.seed,
            device=args.device,
        )
    elif args.command == "matrix":
        result = run_matrix(device=args.device, pair_limit=args.pair_limit)
    elif args.command == "replay-svm":
        result = replay_uniform_bounded_svm(
            sign_flip_iterations=args.sign_flip_iterations
        )
    else:
        result = analyze(
            expected_units=args.expected_units,
            bootstrap_iterations=args.bootstrap_iterations,
            sign_flip_iterations=args.sign_flip_iterations,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
