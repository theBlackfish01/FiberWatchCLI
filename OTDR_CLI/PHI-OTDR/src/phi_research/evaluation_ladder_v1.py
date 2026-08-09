"""Acquisition-safe evaluation ladder for the complete BJTU PHI-OTDR corpus.

The ladder deliberately holds the classifier and signal representation fixed
while progressively strengthening the independence of the test cohort:

1. random windows (recording sessions may cross the split),
2. random complete sessions,
3. held-out date/class cells,
4. held-out acquisition dates, and
5. held-out acquisition eras.

The study is retrospective.  It is designed to quantify protocol sensitivity,
not to create a new confirmatory target result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES, canonical_json_hash
from .morphology_features import transform_view


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_locked_config(config_path: Path, hash_path: Path) -> tuple[dict[str, object], str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = hash_path.read_text(encoding="utf-8").split()[0]
    observed = canonical_json_hash(payload)
    if observed != expected:
        raise ValueError(f"Evaluation-ladder config hash mismatch: {observed} != {expected}")
    return payload, expected


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_bundle(bundle: Mapping[str, np.ndarray], config: Mapping[str, object]) -> None:
    required = {
        "features",
        "feature_names",
        "labels",
        "sessions",
        "window_ids",
        "eras",
        "date_tokens",
        "source_tokens",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Morphology bundle is missing fields: {sorted(missing)}")
    expected = config["data_contract"]
    window_count = len(bundle["labels"])
    sessions = bundle["sessions"].astype(str)
    if window_count != int(expected["windows"]):
        raise ValueError(f"Expected {expected['windows']} windows, found {window_count}")
    if len(set(sessions)) != int(expected["sessions"]):
        raise ValueError(f"Expected {expected['sessions']} sessions")
    if bundle["features"].shape[0] != window_count:
        raise ValueError("Feature and label row counts disagree")
    if set(np.unique(bundle["labels"]).tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("The complete six-class ontology is not present")
    for session in sorted(set(sessions)):
        mask = sessions == session
        for field in ("labels", "eras", "date_tokens", "source_tokens"):
            if len(np.unique(bundle[field][mask].astype(str))) != 1:
                raise ValueError(f"Session {session} is inconsistent for {field}")


def _session_metadata(bundle: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    sessions = bundle["sessions"].astype(str)
    output: dict[str, dict[str, object]] = {}
    for index, session in enumerate(sessions):
        row = {
            "label": int(bundle["labels"][index]),
            "era": str(bundle["eras"][index]),
            "date": str(bundle["date_tokens"][index]),
            "source": str(bundle["source_tokens"][index]),
        }
        if session in output and output[session] != row:
            raise ValueError(f"Session metadata changed within {session}")
        output[session] = row
    return output


def stratified_test_split(
    labels: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split item indices within each class with deterministic coverage."""
    labels = np.asarray(labels, dtype=np.int64)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    test_parts = []
    train_parts = []
    for class_id in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            raise ValueError(f"Class {class_id} has fewer than two split items")
        shuffled = rng.permutation(indices)
        test_size = min(len(indices) - 1, max(1, int(round(len(indices) * test_fraction))))
        test_parts.append(shuffled[:test_size])
        train_parts.append(shuffled[test_size:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))


def random_window_masks(
    labels: np.ndarray, *, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    train_indices, test_indices = stratified_test_split(
        labels, test_fraction=test_fraction, seed=seed
    )
    train = np.zeros(len(labels), dtype=bool)
    test = np.zeros(len(labels), dtype=bool)
    train[train_indices] = True
    test[test_indices] = True
    return train, test


def random_session_masks(
    sessions: np.ndarray,
    labels: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sessions = np.asarray(sessions).astype(str)
    labels = np.asarray(labels, dtype=np.int64)
    unique_sessions = np.asarray(sorted(set(sessions.tolist())))
    session_labels = np.asarray(
        [np.unique(labels[sessions == session]).item() for session in unique_sessions],
        dtype=np.int64,
    )
    train_indices, test_indices = stratified_test_split(
        session_labels, test_fraction=test_fraction, seed=seed
    )
    train_sessions = set(unique_sessions[train_indices].tolist())
    test_sessions = set(unique_sessions[test_indices].tolist())
    return (
        np.asarray([session in train_sessions for session in sessions]),
        np.asarray([session in test_sessions for session in sessions]),
    )


def eligible_date_class_cells(
    session_rows: Mapping[str, Mapping[str, object]], *, min_sessions: int = 2
) -> list[tuple[str, int]]:
    """Return cells whose date and class both have independent graph anchors."""
    dates_by_class: dict[int, set[str]] = defaultdict(set)
    classes_by_date: dict[str, set[int]] = defaultdict(set)
    sessions_by_cell: dict[tuple[str, int], list[str]] = defaultdict(list)
    for session, row in session_rows.items():
        date = str(row["date"])
        label = int(row["label"])
        dates_by_class[label].add(date)
        classes_by_date[date].add(label)
        sessions_by_cell[(date, label)].append(session)
    return sorted(
        cell
        for cell, cell_sessions in sessions_by_cell.items()
        if len(cell_sessions) >= min_sessions
        and len(classes_by_date[cell[0]]) >= 2
        and len(dates_by_class[cell[1]]) >= 2
    )


def overlap_diagnostics(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    sessions: np.ndarray,
    dates: np.ndarray,
    labels: np.ndarray,
    eras: np.ndarray,
) -> dict[str, object]:
    sessions = np.asarray(sessions).astype(str)
    dates = np.asarray(dates).astype(str)
    eras = np.asarray(eras).astype(str)
    labels = np.asarray(labels, dtype=np.int64)

    def values(array: np.ndarray, mask: np.ndarray) -> set[object]:
        return set(array[mask].tolist())

    train_sessions = values(sessions, train_mask)
    test_sessions = values(sessions, test_mask)
    train_dates = values(dates, train_mask)
    test_dates = values(dates, test_mask)
    train_eras = values(eras, train_mask)
    test_eras = values(eras, test_mask)
    train_cells = set(zip(dates[train_mask].tolist(), labels[train_mask].tolist(), strict=True))
    test_cells = set(zip(dates[test_mask].tolist(), labels[test_mask].tolist(), strict=True))
    session_overlap = train_sessions & test_sessions
    return {
        "train_windows": int(np.sum(train_mask)),
        "test_windows": int(np.sum(test_mask)),
        "train_sessions": len(train_sessions),
        "test_sessions": len(test_sessions),
        "train_test_session_overlap": len(session_overlap),
        "test_session_overlap_fraction": (
            float(len(session_overlap) / len(test_sessions)) if test_sessions else 0.0
        ),
        "train_test_date_overlap": len(train_dates & test_dates),
        "train_test_cell_overlap": len(train_cells & test_cells),
        "train_test_era_overlap": len(train_eras & test_eras),
    }


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    correct = prediction == labels
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > left) & (confidence <= right)
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return float(result)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(CLASS_NAMES)):
        raise ValueError("Probability shape does not match the six-class contract")
    prediction = np.argmax(probabilities, axis=1)
    present = np.unique(labels)
    recall = recall_score(
        labels,
        prediction,
        labels=np.arange(len(CLASS_NAMES)),
        average=None,
        zero_division=0,
    )
    present_recall = recall[present]
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    return {
        "count": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1_six_classes": float(
            f1_score(
                labels,
                prediction,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_observed_classes": float(
            f1_score(labels, prediction, labels=present, average="macro", zero_division=0)
        ),
        "balanced_accuracy_observed_classes": float(np.mean(present_recall)),
        "worst_observed_class_recall": float(np.min(present_recall)),
        "per_class_recall": {
            class_name: (float(recall[class_id]) if class_id in present else None)
            for class_id, class_name in enumerate(CLASS_NAMES)
        },
        "negative_log_likelihood": float(
            log_loss(labels, probabilities, labels=np.arange(len(CLASS_NAMES)))
        ),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10": _ece(labels, probabilities),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
    }


def aggregate_session_probabilities(
    labels: np.ndarray,
    sessions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sessions = np.asarray(sessions).astype(str)
    labels = np.asarray(labels, dtype=np.int64)
    unique_sessions = np.asarray(sorted(set(sessions.tolist())))
    session_labels = []
    session_probabilities = []
    window_counts = []
    for session in unique_sessions:
        mask = sessions == session
        local_labels = np.unique(labels[mask])
        if len(local_labels) != 1:
            raise ValueError(f"Session {session} has multiple labels")
        session_labels.append(int(local_labels[0]))
        session_probabilities.append(np.mean(probabilities[mask], axis=0))
        window_counts.append(int(np.sum(mask)))
    output = np.asarray(session_probabilities, dtype=np.float64)
    output /= np.sum(output, axis=1, keepdims=True)
    return (
        unique_sessions,
        np.asarray(session_labels, dtype=np.int64),
        output,
        np.asarray(window_counts, dtype=np.int64),
    )


def _build_representation(
    bundle: Mapping[str, np.ndarray], spec: Mapping[str, object]
) -> np.ndarray:
    feature_names = bundle["feature_names"].astype(str).tolist()
    rows = []
    output_names: tuple[str, ...] | None = None
    for row in bundle["features"]:
        transformed, names = transform_view(
            row,
            feature_names,
            view=str(spec["view"]),
            estimator=str(spec["estimator"]),
            ablation=str(spec["ablation"]),
        )
        if output_names is None:
            output_names = names
        elif output_names != names:
            raise AssertionError("Representation schema changed within the bundle")
        rows.append(transformed)
    return np.asarray(rows, dtype=np.float32)


def _load_or_build_representation(
    bundle: Mapping[str, np.ndarray],
    spec: Mapping[str, object],
    cache_dir: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{spec['name']}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            features = cached["features"]
        if len(features) != len(bundle["labels"]):
            raise ValueError(f"Stale representation cache: {path}")
        return features
    features = _build_representation(bundle, spec)
    np.savez_compressed(path, features=features)
    return features


def _fit_model(features: np.ndarray, labels: np.ndarray, config: Mapping[str, object], seed: int):
    model_config = config["model"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(model_config["C"]),
            class_weight=str(model_config["class_weight"]),
            max_iter=int(model_config["max_iter"]),
            solver=str(model_config["solver"]),
            random_state=seed,
        ),
    )
    model.fit(features, labels)
    classes = model.named_steps["logisticregression"].classes_
    if not np.array_equal(classes, np.arange(len(CLASS_NAMES))):
        raise ValueError(f"Training fold does not contain all classes: {classes}")
    return model


def _evaluate_fold(
    *,
    level: str,
    fold: str,
    seed: int,
    representation: str,
    features: np.ndarray,
    bundle: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if np.any(train_mask & test_mask) or not np.any(train_mask) or not np.any(test_mask):
        raise ValueError(f"Invalid train/test masks for {level}/{fold}")
    labels = bundle["labels"].astype(np.int64)
    started = time.perf_counter()
    model = _fit_model(features[train_mask], labels[train_mask], config, seed)
    fit_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(features[test_mask])
    test_sessions = bundle["sessions"][test_mask].astype(str)
    session_ids, session_labels, session_probs, test_window_counts = aggregate_session_probabilities(
        labels[test_mask], test_sessions, probabilities
    )
    session_rows = _session_metadata(bundle)
    diagnostics = overlap_diagnostics(
        train_mask,
        test_mask,
        sessions=bundle["sessions"],
        dates=bundle["date_tokens"],
        labels=labels,
        eras=bundle["eras"],
    )
    fold_result: dict[str, object] = {
        "level": level,
        "fold": fold,
        "seed": seed,
        "representation": representation,
        **diagnostics,
        "fit_seconds": fit_seconds,
        "window_metrics": classification_metrics(labels[test_mask], probabilities),
        "session_metrics": classification_metrics(session_labels, session_probs),
    }
    predictions = []
    for index, session in enumerate(session_ids):
        prediction = int(np.argmax(session_probs[index]))
        row: dict[str, object] = {
            "level": level,
            "fold": fold,
            "seed": seed,
            "representation": representation,
            "session_id": session,
            "date_token": session_rows[session]["date"],
            "source_token": session_rows[session]["source"],
            "era": session_rows[session]["era"],
            "true_label": int(session_labels[index]),
            "true_class": CLASS_NAMES[int(session_labels[index])],
            "predicted_label": prediction,
            "predicted_class": CLASS_NAMES[prediction],
            "test_window_count": int(test_window_counts[index]),
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"prob_{class_name}"] = float(session_probs[index, class_id])
        predictions.append(row)
    return fold_result, predictions


def _summarize_repeated(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metrics = (
        "accuracy",
        "macro_f1_six_classes",
        "balanced_accuracy_observed_classes",
        "worst_observed_class_recall",
        "negative_log_likelihood",
        "ece_10",
    )
    output: dict[str, object] = {"fold_count": len(folds)}
    for unit in ("window_metrics", "session_metrics"):
        unit_summary = {}
        for metric in metrics:
            values = np.asarray([float(row[unit][metric]) for row in folds])
            unit_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        output[unit] = unit_summary
    output["mean_test_session_overlap_fraction"] = float(
        np.mean([float(row["test_session_overlap_fraction"]) for row in folds])
    )
    return output


def _pooled_metrics(
    predictions: Sequence[Mapping[str, object]], *, require_unique_sessions: bool
) -> dict[str, object]:
    if require_unique_sessions:
        ids = [str(row["session_id"]) for row in predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("Pooled protocol predicts a session more than once")
    labels = np.asarray([int(row["true_label"]) for row in predictions], dtype=np.int64)
    probabilities = np.asarray(
        [
            [float(row[f"prob_{class_name}"]) for class_name in CLASS_NAMES]
            for row in predictions
        ],
        dtype=np.float64,
    )
    return classification_metrics(labels, probabilities)


def run_ladder(
    *,
    bundle_path: Path,
    config_path: Path,
    hash_path: Path,
    output_dir: Path,
    representations: Sequence[str] | None = None,
) -> dict[str, object]:
    config, config_hash = _read_locked_config(config_path, hash_path)
    if _sha256(bundle_path) != str(config["morphology_bundle_sha256"]):
        raise ValueError("Complete morphology bundle hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    _validate_bundle(bundle, config)
    session_rows = _session_metadata(bundle)
    cells = eligible_date_class_cells(
        session_rows, min_sessions=int(config["levels"]["date_class_cell"]["minimum_sessions"])
    )
    expected_cells = int(config["levels"]["date_class_cell"]["expected_cells"])
    if len(cells) != expected_cells:
        raise ValueError(f"Eligible date/class cells changed: {len(cells)} != {expected_cells}")

    requested = None if representations is None else set(representations)
    specs = [
        spec
        for spec in config["representations"]
        if requested is None or str(spec["name"]) in requested
    ]
    if not specs or (requested is not None and {str(spec["name"]) for spec in specs} != requested):
        raise ValueError("Requested representation is not frozen in the config")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "representation_cache"
    all_folds: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    started = time.perf_counter()
    labels = bundle["labels"].astype(np.int64)
    sessions = bundle["sessions"].astype(str)
    dates = bundle["date_tokens"].astype(str)
    eras = bundle["eras"].astype(str)

    for spec in specs:
        representation = str(spec["name"])
        features = _load_or_build_representation(bundle, spec, cache_dir)
        representation_folds: list[dict[str, object]] = []
        representation_predictions: list[dict[str, object]] = []

        for level in ("random_window", "random_session"):
            level_folds = []
            for seed in config["seeds"]:
                if level == "random_window":
                    train_mask, test_mask = random_window_masks(
                        labels,
                        test_fraction=float(config["levels"][level]["test_fraction"]),
                        seed=int(seed),
                    )
                else:
                    train_mask, test_mask = random_session_masks(
                        sessions,
                        labels,
                        test_fraction=float(config["levels"][level]["test_fraction"]),
                        seed=int(seed),
                    )
                fold_result, predictions = _evaluate_fold(
                    level=level,
                    fold=f"seed_{seed}",
                    seed=int(seed),
                    representation=representation,
                    features=features,
                    bundle=bundle,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    config=config,
                )
                level_folds.append(fold_result)
                representation_folds.append(fold_result)
                representation_predictions.extend(predictions)
                print(
                    f"[{representation}] {level} seed={seed} "
                    f"session_f1={fold_result['session_metrics']['macro_f1_six_classes']:.4f}",
                    flush=True,
                )
            summaries[f"{representation}::{level}"] = _summarize_repeated(level_folds)

        cell_folds = []
        cell_predictions = []
        for date, class_id in cells:
            test_mask = (dates == date) & (labels == class_id)
            train_mask = ~test_mask
            fold_result, predictions = _evaluate_fold(
                level="date_class_cell",
                fold=f"{date}__{CLASS_NAMES[class_id]}",
                seed=int(config["fixed_seed"]),
                representation=representation,
                features=features,
                bundle=bundle,
                train_mask=train_mask,
                test_mask=test_mask,
                config=config,
            )
            cell_folds.append(fold_result)
            cell_predictions.extend(predictions)
        representation_folds.extend(cell_folds)
        representation_predictions.extend(cell_predictions)
        summaries[f"{representation}::date_class_cell"] = {
            "fold_count": len(cell_folds),
            "eligible_sessions": len(cell_predictions),
            "pooled_session_metrics": _pooled_metrics(
                cell_predictions, require_unique_sessions=True
            ),
            "mean_fold_target_recall": float(
                np.mean(
                    [
                        row["session_metrics"]["balanced_accuracy_observed_classes"]
                        for row in cell_folds
                    ]
                )
            ),
        }
        print(
            f"[{representation}] date_class_cell folds={len(cell_folds)} "
            f"pooled_f1={summaries[f'{representation}::date_class_cell']['pooled_session_metrics']['macro_f1_six_classes']:.4f}",
            flush=True,
        )

        date_folds = []
        date_predictions = []
        for date in sorted(set(dates.tolist())):
            test_mask = dates == date
            train_mask = ~test_mask
            fold_result, predictions = _evaluate_fold(
                level="leave_one_date_out",
                fold=date,
                seed=int(config["fixed_seed"]),
                representation=representation,
                features=features,
                bundle=bundle,
                train_mask=train_mask,
                test_mask=test_mask,
                config=config,
            )
            date_folds.append(fold_result)
            date_predictions.extend(predictions)
        representation_folds.extend(date_folds)
        representation_predictions.extend(date_predictions)
        summaries[f"{representation}::leave_one_date_out"] = {
            "fold_count": len(date_folds),
            "pooled_session_metrics": _pooled_metrics(date_predictions, require_unique_sessions=True),
            "mean_fold_accuracy": float(
                np.mean([row["session_metrics"]["accuracy"] for row in date_folds])
            ),
        }
        print(
            f"[{representation}] leave_one_date_out folds={len(date_folds)} "
            f"pooled_f1={summaries[f'{representation}::leave_one_date_out']['pooled_session_metrics']['macro_f1_six_classes']:.4f}",
            flush=True,
        )

        era_summaries = {}
        for source_era, target_era in (("january", "april_may"), ("april_may", "january")):
            train_mask = eras == source_era
            test_mask = eras == target_era
            fold_result, predictions = _evaluate_fold(
                level="cross_era",
                fold=f"{source_era}_to_{target_era}",
                seed=int(config["fixed_seed"]),
                representation=representation,
                features=features,
                bundle=bundle,
                train_mask=train_mask,
                test_mask=test_mask,
                config=config,
            )
            representation_folds.append(fold_result)
            representation_predictions.extend(predictions)
            era_summaries[f"{source_era}_to_{target_era}"] = fold_result["session_metrics"]
            print(
                f"[{representation}] {source_era}_to_{target_era} "
                f"session_f1={fold_result['session_metrics']['macro_f1_six_classes']:.4f}",
                flush=True,
            )
        summaries[f"{representation}::cross_era"] = era_summaries
        all_folds.extend(representation_folds)
        all_predictions.extend(representation_predictions)

    fold_csv = output_dir / "fold_results.csv"
    prediction_csv = output_dir / "session_predictions.csv"
    flat_folds = []
    for row in all_folds:
        flat_folds.append(
            {
                "level": row["level"],
                "fold": row["fold"],
                "seed": row["seed"],
                "representation": row["representation"],
                "train_windows": row["train_windows"],
                "test_windows": row["test_windows"],
                "train_sessions": row["train_sessions"],
                "test_sessions": row["test_sessions"],
                "train_test_session_overlap": row["train_test_session_overlap"],
                "test_session_overlap_fraction": row["test_session_overlap_fraction"],
                "train_test_date_overlap": row["train_test_date_overlap"],
                "train_test_cell_overlap": row["train_test_cell_overlap"],
                "train_test_era_overlap": row["train_test_era_overlap"],
                "fit_seconds": row["fit_seconds"],
                "window_accuracy": row["window_metrics"]["accuracy"],
                "window_macro_f1": row["window_metrics"]["macro_f1_six_classes"],
                "window_ece": row["window_metrics"]["ece_10"],
                "session_accuracy": row["session_metrics"]["accuracy"],
                "session_macro_f1": row["session_metrics"]["macro_f1_six_classes"],
                "session_balanced_accuracy": row["session_metrics"][
                    "balanced_accuracy_observed_classes"
                ],
                "session_worst_observed_recall": row["session_metrics"][
                    "worst_observed_class_recall"
                ],
                "session_ece": row["session_metrics"]["ece_10"],
            }
        )
    _write_csv(fold_csv, flat_folds)
    _write_csv(prediction_csv, all_predictions)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "PHI-OTDR acquisition-safe evaluation ladder v1",
        "evidence_status": "retrospective development; all acquisition-era outcomes are historically exposed",
        "config_sha256": config_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "morphology_bundle_sha256": config["morphology_bundle_sha256"],
        "representations_run": [str(spec["name"]) for spec in specs],
        "eligible_date_class_cells": [
            {"date_token": date, "class_id": class_id, "class_name": CLASS_NAMES[class_id]}
            for date, class_id in cells
        ],
        "summaries": summaries,
        "fold_count": len(all_folds),
        "session_prediction_rows": len(all_predictions),
        "elapsed_seconds": time.perf_counter() - started,
        "output_hashes": {
            "fold_results_sha256": _sha256(fold_csv),
            "session_predictions_sha256": _sha256(prediction_csv),
        },
        "limitations": [
            "Random-window evaluation intentionally allows recording-session overlap and is a diagnostic, not a deployment estimate.",
            "Date and source tokens are acquisition proxies, not authoritative site, operator, or subject identities.",
            "Only two acquisition eras exist, and their labels were inspected during earlier development.",
            "Held-out date/class folds test compositional generalization but do not create an unseen acquisition domain.",
            "The fixed logistic model isolates split sensitivity; it is not claimed to be the best possible classifier.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "evaluation_ladder_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representations", nargs="+")
    args = parser.parse_args()
    result = run_ladder(
        bundle_path=args.bundle,
        config_path=args.config,
        hash_path=args.hash,
        output_dir=args.output_dir,
        representations=args.representations,
    )
    print(
        json.dumps(
            {
                "payload_sha256": result["payload_sha256"],
                "fold_count": result["fold_count"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
