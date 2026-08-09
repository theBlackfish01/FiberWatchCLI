"""Interpretable morphology/acquisition factorization under frozen PHI splits."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .evaluation_ladder_v1 import (
    classification_metrics,
    eligible_date_class_cells,
    stratified_test_split,
)
from .morphology_attributes_v3 import _view_indices
from .shift_protocol_v1 import (
    finalize_payload,
    load_locked_config,
    process_memory_snapshot,
    sha256_file,
    write_csv,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _metadata(bundle: Mapping[str, np.ndarray], sessions: np.ndarray) -> dict[str, np.ndarray]:
    window_sessions = bundle["sessions"].astype(str)
    output: dict[str, list[object]] = {"labels": [], "dates": [], "eras": [], "sources": []}
    for session in sessions.astype(str):
        indices = np.flatnonzero(window_sessions == session)
        if not len(indices):
            raise ValueError(f"Session attributes are absent from morphology bundle: {session}")
        for source, target in (
            (bundle["labels"], "labels"),
            (bundle["date_tokens"], "dates"),
            (bundle["eras"], "eras"),
            (bundle["source_tokens"], "sources"),
        ):
            value = np.unique(source[indices].astype(str) if target != "labels" else source[indices])
            if len(value) != 1:
                raise ValueError(f"Inconsistent {target} for {session}")
            output[target].append(int(value[0]) if target == "labels" else str(value[0]))
    return {
        "labels": np.asarray(output["labels"], dtype=np.int64),
        "dates": np.asarray(output["dates"], dtype=str),
        "eras": np.asarray(output["eras"], dtype=str),
        "sources": np.asarray(output["sources"], dtype=str),
    }


def _fit_classifier(x: np.ndarray, y: np.ndarray, config: Mapping[str, object], seed: int):
    return LogisticRegression(
        C=float(config["C"]),
        class_weight="balanced",
        max_iter=int(config["max_iter"]),
        solver=str(config["solver"]),
        random_state=int(seed % (2**32)),
    ).fit(x, y)


def _source_date_residual(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_dates: np.ndarray,
    test_dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grand = np.mean(train_x, axis=0)
    means = {date: np.mean(train_x[train_dates == date], axis=0) for date in np.unique(train_dates)}
    transformed_train = np.stack([row - means[date] + grand for row, date in zip(train_x, train_dates, strict=True)])
    transformed_test = np.stack(
        [row - means[date] + grand if date in means else row for row, date in zip(test_x, test_dates, strict=True)]
    )
    return transformed_train, transformed_test


def _nuisance_projection(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_dates: np.ndarray,
    *,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_x)
    train = scaler.transform(train_x)
    test = scaler.transform(test_x)
    unique_dates = sorted(set(train_dates.tolist()))
    design = np.stack([(train_dates == date).astype(float) for date in unique_dates], axis=1)
    design -= np.mean(design, axis=0, keepdims=True)
    coefficient = Ridge(alpha=ridge, fit_intercept=False).fit(train, design).coef_.T
    directions, singular, _ = np.linalg.svd(coefficient, full_matrices=False)
    nuisance = directions[:, : min(rank, directions.shape[1])]
    projection = np.eye(train.shape[1]) - nuisance @ nuisance.T
    return train @ projection, test @ projection, singular


def _covariance_map(source: np.ndarray, target: np.ndarray, floor: float, shrinkage: float) -> np.ndarray:
    def covariance(values: np.ndarray) -> np.ndarray:
        raw = np.cov(values, rowvar=False) if len(values) > 1 else np.eye(values.shape[1])
        scale = float(np.trace(raw) / raw.shape[0])
        return (1.0 - shrinkage) * raw + shrinkage * scale * np.eye(raw.shape[0])

    def power(matrix: np.ndarray, exponent: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, floor)
        return (vectors * (values**exponent)) @ vectors.T

    return power(covariance(target), -0.5) @ power(covariance(source), 0.5)


def transform_fold(
    method: str,
    features: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    context_mask: np.ndarray,
    dates: np.ndarray,
    method_config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_raw = features[train_mask]
    test_raw = features[test_mask]
    diagnostics: dict[str, object] = {}
    if method == "source_date_residual":
        train_raw, test_raw = _source_date_residual(
            train_raw, test_raw, dates[train_mask], dates[test_mask]
        )
        scaler = StandardScaler().fit(train_raw)
        return scaler.transform(train_raw), scaler.transform(test_raw), diagnostics
    if method == "source_nuisance_projection_rank4":
        train, test, singular = _nuisance_projection(
            train_raw,
            test_raw,
            dates[train_mask],
            rank=int(method_config["rank"]),
            ridge=float(method_config["ridge"]),
        )
        diagnostics["nuisance_singular_values"] = singular[:10].tolist()
        return train, test, diagnostics
    scaler = StandardScaler().fit(train_raw)
    train = scaler.transform(train_raw)
    test = scaler.transform(test_raw)
    if method == "baseline":
        return train, test, diagnostics
    context = scaler.transform(features[context_mask])
    if not len(context):
        raise ValueError(f"Transductive context is empty for {method}")
    if method == "target_unlabelled_mean_alignment":
        shift = np.mean(train, axis=0) - np.mean(context, axis=0)
        diagnostics["alignment_shift_norm"] = float(np.linalg.norm(shift))
        return train, test + shift, diagnostics
    if method == "target_unlabelled_coral":
        context_mean = np.mean(context, axis=0)
        source_mean = np.mean(train, axis=0)
        mapping = _covariance_map(
            train,
            context,
            float(method_config["eigenvalue_floor"]),
            float(method_config["shrinkage"]),
        )
        diagnostics["alignment_shift_norm"] = float(np.linalg.norm(source_mean - context_mean))
        diagnostics["coral_map_norm"] = float(np.linalg.norm(mapping, ord="fro"))
        return train, (test - context_mean) @ mapping + source_mean, diagnostics
    raise ValueError(f"Unknown factorization method: {method}")


def _fold(
    *,
    method: str,
    features: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    dates: np.ndarray,
    eras: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    context_mask: np.ndarray,
    config: Mapping[str, object],
    seed: int,
    level: str,
    name: str,
    target_access: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if np.any(train_mask & test_mask):
        raise AssertionError("Train and test sessions overlap")
    train, test, diagnostics = transform_fold(
        method,
        features,
        train_mask,
        test_mask,
        context_mask,
        dates,
        config["methods"][method],
    )
    model = _fit_classifier(train, labels[train_mask], config["classifier"], seed)
    local = model.predict_proba(test)
    probabilities = np.zeros((len(test), len(CLASS_NAMES)), dtype=float)
    probabilities[:, model.classes_.astype(int)] = local
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    metrics = classification_metrics(labels[test_mask], probabilities)
    fold = {
        "level": level,
        "fold": name,
        "seed": seed,
        "method": method,
        "target_access": target_access,
        "train_sessions": int(np.sum(train_mask)),
        "test_sessions": int(np.sum(test_mask)),
        "train_dates": len(set(dates[train_mask].tolist())),
        "test_dates": len(set(dates[test_mask].tolist())),
        "train_test_session_overlap": 0,
        **{key: value for key, value in metrics.items() if key not in {"per_class_recall", "confusion_matrix"}},
        **{f"recall_{key}": value for key, value in metrics["per_class_recall"].items()},
        **{key: json.dumps(value) if isinstance(value, list) else value for key, value in diagnostics.items()},
    }
    predictions = []
    for local_index, global_index in enumerate(np.flatnonzero(test_mask)):
        row: dict[str, object] = {
            "level": level,
            "fold": name,
            "seed": seed,
            "method": method,
            "target_access": target_access,
            "session_id": sessions[global_index],
            "date_token": dates[global_index],
            "era": eras[global_index],
            "true_label": int(labels[global_index]),
            "true_class": CLASS_NAMES[int(labels[global_index])],
            "predicted_label": int(np.argmax(probabilities[local_index])),
            "predicted_class": CLASS_NAMES[int(np.argmax(probabilities[local_index]))],
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"prob_{class_name}"] = float(probabilities[local_index, class_id])
        predictions.append(row)
    return fold, predictions


def _summary(
    fold_rows: Sequence[Mapping[str, object]], prediction_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    output = []
    groups = sorted({(str(row["level"]), str(row["method"])) for row in fold_rows})
    for level, method in groups:
        local_folds = [row for row in fold_rows if row["level"] == level and row["method"] == method]
        if level == "random_session":
            metrics = {
                key: float(np.mean([float(row[key]) for row in local_folds]))
                for key in ("macro_f1_six_classes", "balanced_accuracy_observed_classes", "worst_observed_class_recall", "negative_log_likelihood", "brier_score", "ece_10")
            }
            output.append({"level": level, "direction": "pooled", "method": method, "folds": len(local_folds), **metrics})
            continue
        direction_groups = ["pooled"]
        if level == "cross_era":
            direction_groups = sorted({str(row["fold"]) for row in local_folds})
        for direction in direction_groups:
            local = [
                row
                for row in prediction_rows
                if row["level"] == level
                and row["method"] == method
                and (direction == "pooled" or row["fold"] == direction)
            ]
            labels = np.asarray([int(row["true_label"]) for row in local])
            probabilities = np.asarray(
                [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in local]
            )
            metrics = classification_metrics(labels, probabilities)
            output.append(
                {
                    "level": level,
                    "direction": direction,
                    "method": method,
                    "folds": len(local_folds) if direction == "pooled" else 1,
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                }
            )
    return output


def _graph_components(dates: np.ndarray, labels: np.ndarray) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for date, label in sorted(set(zip(dates.tolist(), labels.tolist(), strict=True))):
        left, right = f"date:{date}", f"class:{CLASS_NAMES[int(label)]}"
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(adjacency)
    components = []
    while unseen:
        root = min(unseen)
        queue = deque([root])
        component = []
        unseen.remove(root)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _graph_factorization(
    features: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    sessions: np.ndarray,
    eligible: Sequence[tuple[str, int]],
    ridge: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    unique_dates = sorted(set(dates.tolist()))
    observed = sorted(set(zip(dates.tolist(), labels.tolist(), strict=True)))
    cell_means = {
        (date, int(label)): np.mean(features[(dates == date) & (labels == label)], axis=0)
        for date, label in observed
    }

    def design(cells: Sequence[tuple[str, int]]) -> np.ndarray:
        rows = []
        for date, label in cells:
            row = [1.0]
            row.extend(float(date == value) for value in unique_dates[1:])
            row.extend(float(label == value) for value in range(1, len(CLASS_NAMES)))
            rows.append(row)
        return np.asarray(rows)

    edge_rows = []
    for target in eligible:
        training = [cell for cell in observed if cell != target]
        model = Ridge(alpha=ridge, fit_intercept=False).fit(
            design(training), np.stack([cell_means[cell] for cell in training])
        )
        actual = cell_means[target]
        predicted = model.predict(design([target]))[0]
        class_only = np.mean(
            np.stack([cell_means[cell] for cell in training if cell[1] == target[1]]), axis=0
        )
        scale = np.sqrt(np.mean(np.var(np.stack(list(cell_means.values())), axis=0)))
        additive_error = float(np.sqrt(np.mean((actual - predicted) ** 2)) / max(scale, 1e-12))
        class_error = float(np.sqrt(np.mean((actual - class_only) ** 2)) / max(scale, 1e-12))
        candidate_centroids = []
        for class_id in range(len(CLASS_NAMES)):
            candidate = (target[0], class_id)
            candidate_centroids.append(model.predict(design([candidate]))[0])
        predicted_class = int(np.argmin([np.linalg.norm(actual - value) for value in candidate_centroids]))
        edge_rows.append(
            {
                "date_token": target[0],
                "class_id": target[1],
                "class_name": CLASS_NAMES[target[1]],
                "sessions": int(np.sum((dates == target[0]) & (labels == target[1]))),
                "additive_normalized_rmse": additive_error,
                "class_only_normalized_rmse": class_error,
                "rmse_improvement": class_error - additive_error,
                "predicted_class": CLASS_NAMES[predicted_class],
                "correct": predicted_class == target[1],
            }
        )
    cells_design = design(observed)
    singular = np.linalg.svd(cells_design, compute_uv=False)
    date_degree = Counter(date for date, _ in observed)
    class_degree = Counter(CLASS_NAMES[int(label)] for _, label in observed)
    summary = {
        "observed_cells": len(observed),
        "possible_cells": len(unique_dates) * len(CLASS_NAMES),
        "eligible_edges": len(eligible),
        "design_rank": int(np.linalg.matrix_rank(cells_design)),
        "design_columns": cells_design.shape[1],
        "condition_number": float(singular[0] / max(singular[-1], 1e-12)),
        "connected_components": _graph_components(dates, labels),
        "date_degrees": dict(sorted(date_degree.items())),
        "class_degrees": dict(sorted(class_degree.items())),
        "weak_date_anchors": sorted([date for date, degree in date_degree.items() if degree == 1]),
        "mean_additive_rmse": float(np.mean([row["additive_normalized_rmse"] for row in edge_rows])),
        "mean_class_only_rmse": float(np.mean([row["class_only_normalized_rmse"] for row in edge_rows])),
        "mean_rmse_improvement": float(np.mean([row["rmse_improvement"] for row in edge_rows])),
        "edge_classification_accuracy": float(np.mean([row["correct"] for row in edge_rows])),
    }
    return edge_rows, summary


def _era_probes(features: np.ndarray, eras: np.ndarray, dates: np.ndarray, rank: int, ridge: float) -> list[dict[str, object]]:
    targets = (eras == "april_may").astype(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260809)
    rows = []
    for method in ("baseline", "source_date_residual", "source_nuisance_projection_rank4"):
        predictions = np.zeros(len(features), dtype=int)
        for train, test in splitter.split(features, targets):
            if method == "source_date_residual":
                train_x, test_x = _source_date_residual(features[train], features[test], dates[train], dates[test])
                scaler = StandardScaler().fit(train_x)
                train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)
            elif method == "source_nuisance_projection_rank4":
                train_x, test_x, _ = _nuisance_projection(features[train], features[test], dates[train], rank=rank, ridge=ridge)
            else:
                scaler = StandardScaler().fit(features[train])
                train_x, test_x = scaler.transform(features[train]), scaler.transform(features[test])
            predictions[test] = LogisticRegression(class_weight="balanced", max_iter=2000).fit(train_x, targets[train]).predict(test_x)
        recall = [np.mean(predictions[targets == value] == value) for value in (0, 1)]
        rows.append({"method": method, "balanced_accuracy": float(np.mean(recall)), "recall_january": float(recall[0]), "recall_april_may": float(recall[1])})
    return rows


def run(
    *,
    morphology_bundle_path: Path,
    session_attributes_path: Path,
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config, config_hash = load_locked_config(config_path, config_hash_path)
    if sha256_file(session_attributes_path) != config["input_session_attributes_sha256"]:
        raise ValueError("Session-attribute hash mismatch")
    attributes = _load_npz(session_attributes_path)
    bundle = _load_npz(morphology_bundle_path)
    sessions = attributes["sessions"].astype(str)
    names = attributes["attribute_names"].astype(str)
    indices = _view_indices(names, "morphology_only")
    if len(indices) != 114:
        raise ValueError(f"Expected 114 position-free morphology attributes, found {len(indices)}")
    features = attributes["attributes"][:, indices].astype(np.float64)
    metadata = _metadata(bundle, sessions)
    labels, dates, eras = metadata["labels"], metadata["dates"], metadata["eras"]
    if not np.array_equal(labels, attributes["labels"]):
        raise ValueError("Session labels are misaligned")
    session_rows = {
        session: {"label": int(labels[i]), "date": dates[i], "era": eras[i]}
        for i, session in enumerate(sessions)
    }
    eligible = eligible_date_class_cells(session_rows, min_sessions=int(config["minimum_cell_sessions"]))
    methods = list(config["methods"])
    folds = []
    predictions = []

    def evaluate(level: str, name: str, train_mask: np.ndarray, test_mask: np.ndarray, context_mask: np.ndarray, seed: int) -> None:
        for method in methods:
            fold, local = _fold(
                method=method,
                features=features,
                labels=labels,
                sessions=sessions,
                dates=dates,
                eras=eras,
                train_mask=train_mask,
                test_mask=test_mask,
                context_mask=context_mask,
                config=config,
                seed=seed,
                level=level,
                name=name,
                target_access=str(config["methods"][method]["target_access"]),
            )
            folds.append(fold)
            predictions.extend(local)

    for seed in config["seeds"]:
        train_indices, test_indices = stratified_test_split(
            labels, test_fraction=float(config["random_session_test_fraction"]), seed=int(seed)
        )
        train_mask = np.zeros(len(labels), dtype=bool)
        test_mask = np.zeros(len(labels), dtype=bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        evaluate("random_session", f"seed_{seed}", train_mask, test_mask, test_mask, int(seed))
    fixed_seed = int(config["seeds"][0])
    for date, class_id in eligible:
        test_mask = (dates == date) & (labels == class_id)
        train_mask = ~test_mask
        context_mask = dates == date
        evaluate("heldout_date_class_cell", f"{date}__{CLASS_NAMES[class_id]}", train_mask, test_mask, context_mask, fixed_seed)
    for date in sorted(set(dates.tolist())):
        test_mask = dates == date
        train_mask = ~test_mask
        evaluate("leave_one_date_out", date, train_mask, test_mask, test_mask, fixed_seed)
    for source, target in (("january", "april_may"), ("april_may", "january")):
        train_mask, test_mask = eras == source, eras == target
        evaluate("cross_era", f"{source}_to_{target}", train_mask, test_mask, test_mask, fixed_seed)

    summaries = _summary(folds, predictions)
    edge_rows, graph = _graph_factorization(
        features, labels, dates, sessions, eligible, float(config["graph_analysis"]["ridge"])
    )
    nuisance_config = config["methods"]["source_nuisance_projection_rank4"]
    probes = _era_probes(
        features,
        eras,
        dates,
        int(nuisance_config["rank"]),
        float(nuisance_config["ridge"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_results.csv", folds)
    write_csv(output_dir / "session_predictions.csv", predictions)
    write_csv(output_dir / "summary.csv", [
        {**row, "per_class_recall": json.dumps(row.get("per_class_recall", {}))}
        for row in summaries
    ])
    write_csv(output_dir / "graph_edge_predictions.csv", edge_rows)
    write_csv(output_dir / "era_probes.csv", probes)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": config["protocol_name"],
        "evidence_status": config["evidence_status"],
        "config_sha256": config_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "input_hashes": {
            "morphology_bundle_sha256": sha256_file(morphology_bundle_path),
            "session_attributes_sha256": sha256_file(session_attributes_path),
        },
        "feature_view": config["input_view"],
        "feature_count": features.shape[1],
        "session_count": len(sessions),
        "fold_count": len(folds),
        "prediction_count": len(predictions),
        "summaries": summaries,
        "graph_factorization": graph,
        "era_probes": probes,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "factorization_results.json"
        },
        "limitations": [
            "All target-era outcomes are historically exposed; the comparison is retrospective development evidence.",
            "Target-unlabelled alignment is transductive and assumes a batch of target sessions is available.",
            "The observed date-class graph is sparse, so additive effects can be weakly anchored or non-identifiable.",
            "Era-probe reduction is diagnostic and is not sufficient evidence of preserved class information.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "process_memory": process_memory_snapshot(),
    }
    return finalize_payload(payload, output_dir / "factorization_results.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphology-bundle", type=Path, required=True)
    parser.add_argument("--session-attributes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        morphology_bundle_path=args.morphology_bundle,
        session_attributes_path=args.session_attributes,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"], "fold_count": result["fold_count"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
