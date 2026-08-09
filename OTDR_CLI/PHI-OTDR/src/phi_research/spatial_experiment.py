"""Source-selected retrospective spatial-morphology experiments for Phi-OTDR v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES, canonical_json_hash
from .era_contract import verify_acquisition_manifest
from .morphology_features import aggregate_sessions, transform_view


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-9, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _select_temperature(labels: np.ndarray, probs: np.ndarray) -> float:
    grid = np.geomspace(0.25, 4.0, 65)
    losses = [log_loss(labels, _temperature(probs, value), labels=np.arange(len(CLASS_NAMES))) for value in grid]
    return float(grid[int(np.argmin(losses))])


def _ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    correct = prediction == labels
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > left) & (confidence <= right)
        if np.any(mask):
            result += np.mean(mask) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(result)


def _metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, object]:
    prediction = np.argmax(probs, axis=1)
    recall = recall_score(labels, prediction, labels=np.arange(len(CLASS_NAMES)), average=None, zero_division=0)
    one_hot = np.eye(len(CLASS_NAMES))[labels]
    return {
        "session_count": int(len(labels)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(CLASS_NAMES)},
        "worst_class_recall": float(np.min(recall)),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=np.arange(len(CLASS_NAMES)))),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_10": _ece(labels, probs),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=np.arange(len(CLASS_NAMES))).tolist(),
    }


def _model_grid(model_name: str) -> list[tuple[dict[str, object], object]]:
    if model_name == "logistic":
        return [
            (
                {"C": c},
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=4000,
                        solver="lbfgs",
                        random_state=20260808,
                    ),
                ),
            )
            for c in (0.01, 0.1, 1.0, 10.0)
        ]
    if model_name == "hist_gradient":
        return [
            (
                {"learning_rate": rate, "max_leaf_nodes": leaves},
                HistGradientBoostingClassifier(
                    learning_rate=rate,
                    max_leaf_nodes=leaves,
                    max_iter=250,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=20260808,
                ),
            )
            for rate in (0.03, 0.1)
            for leaves in (7, 15)
        ]
    raise ValueError(f"Unknown model: {model_name}")


def _session_view(
    bundle: dict[str, np.ndarray],
    *,
    view: str,
    estimator: str,
    ablation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_names = bundle["feature_names"].astype(str).tolist()
    transformed = []
    output_names: tuple[str, ...] | None = None
    for row in bundle["features"]:
        values, names = transform_view(
            row,
            feature_names,
            view=view,
            estimator=estimator,
            ablation=ablation,
        )
        if output_names is None:
            output_names = names
        elif output_names != names:
            raise AssertionError("Transformed feature schema changed")
        transformed.append(values)
    session_features, sessions, _ = aggregate_sessions(
        np.asarray(transformed), bundle["sessions"].astype(str), bundle["window_ids"]
    )
    first_index = {session: i for i, session in enumerate(bundle["sessions"].astype(str))}
    labels = np.asarray([bundle["labels"][first_index[session]] for session in sessions], dtype=np.int64)
    return session_features, sessions, labels


def _write_predictions(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    bundle_path: Path,
    manifests: Sequence[Path],
    protocol_path: Path,
    protocol_hash_path: Path,
    output_dir: Path,
    models: Sequence[str] = ("logistic",),
    specs: Sequence[str] | None = None,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = protocol_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(protocol) != expected_hash:
        raise ValueError("V3 protocol hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    if len(bundle["features"]) != 15418 or len(set(bundle["sessions"].astype(str))) != 441:
        raise ValueError("Morphology bundle does not conserve the complete readable cohort")
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    for manifest in manifest_payloads:
        verify_acquisition_manifest(
            manifest,
            expected_dataset_fingerprint=str(protocol["dataset_fingerprint_sha256"]),
        )

    view_specs = [("absolute", "none"), ("invariant", "none")]
    for estimator in protocol["registration_estimators"]:
        for view in ("registered", "registered_position", "dual"):
            view_specs.append((view, str(estimator)))
    ablations = ("amplitude", "dynamics", "fused")
    requested_specs = None if specs is None else set(specs)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "session_views"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    predictions = []
    started = time.perf_counter()
    for spec_index, (view, estimator_label) in enumerate(view_specs, start=1):
        estimator = (
            "multi_estimator_consensus" if estimator_label == "none" else estimator_label
        )
        for ablation in ablations:
            spec_name = f"{view}:{estimator_label}:{ablation}"
            if requested_specs is not None and spec_name not in requested_specs:
                continue
            cache_path = cache_dir / f"{view}__{estimator_label}__{ablation}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as cached:
                    session_features = cached["features"]
                    sessions = cached["sessions"].astype(str)
                    labels = cached["labels"].astype(np.int64)
            else:
                session_features, sessions, labels = _session_view(
                    bundle, view=view, estimator=estimator, ablation=ablation
                )
                np.savez_compressed(cache_path, features=session_features, sessions=sessions, labels=labels)
            print(
                f"[VIEW] {spec_index}/{len(view_specs)} {view}/{estimator_label}/{ablation} "
                f"features={session_features.shape[1]}",
                flush=True,
            )
            for manifest_path, manifest in zip(manifests, manifest_payloads, strict=True):
                session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
                partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
                direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
                train_mask = partitions == "source_train"
                validation_mask = partitions == "source_validation"
                calibration_mask = partitions == "source_calibration"
                query_mask = partitions == "target_query"
                for model_name in models:
                    selected = None
                    selection_rows = []
                    for params, model in _model_grid(model_name):
                        model.fit(session_features[train_mask], labels[train_mask])
                        probs = model.predict_proba(session_features[validation_mask])
                        score = f1_score(
                            labels[validation_mask], np.argmax(probs, axis=1), average="macro", zero_division=0
                        )
                        selection_rows.append({"params": params, "source_validation_macro_f1": float(score)})
                        candidate = (float(score), -len(json.dumps(params)), json.dumps(params, sort_keys=True))
                        if selected is None or candidate > selected[0]:
                            selected = (candidate, params)
                    assert selected is not None
                    selected_params = selected[1]
                    final_model = next(
                        model for params, model in _model_grid(model_name) if params == selected_params
                    )
                    development_mask = train_mask | validation_mask
                    fit_started = time.perf_counter()
                    final_model.fit(session_features[development_mask], labels[development_mask])
                    fit_seconds = time.perf_counter() - fit_started
                    calibration_probs = final_model.predict_proba(session_features[calibration_mask])
                    temperature = _select_temperature(labels[calibration_mask], calibration_probs)
                    source_validation_probs = _temperature(
                        final_model.predict_proba(session_features[validation_mask]), temperature
                    )
                    query_probs = _temperature(
                        final_model.predict_proba(session_features[query_mask]), temperature
                    )
                    validation_metrics = _metrics(labels[validation_mask], source_validation_probs)
                    query_metrics = _metrics(labels[query_mask], query_probs)
                    result = {
                        "direction": direction,
                        "manifest": manifest_path.name,
                        "view": view,
                        "estimator": estimator_label,
                        "ablation": ablation,
                        "model": model_name,
                        "session_feature_count": int(session_features.shape[1]),
                        "selected_params": selected_params,
                        "selection_trace": selection_rows,
                        "temperature": temperature,
                        "source_validation": validation_metrics,
                        "target_query_retrospective": query_metrics,
                        "generalization_gap_macro_f1": float(
                            validation_metrics["macro_f1"] - query_metrics["macro_f1"]
                        ),
                        "fit_seconds": fit_seconds,
                        "selection_used_target_query": False,
                    }
                    results.append(result)
                    query_sessions = sessions[query_mask]
                    for local_index, session in enumerate(query_sessions):
                        row = session_rows[session]
                        prediction = int(np.argmax(query_probs[local_index]))
                        prediction_row: dict[str, object] = {
                            "direction": direction,
                            "view": view,
                            "estimator": estimator_label,
                            "ablation": ablation,
                            "model": model_name,
                            "session_id": session,
                            "true_label": int(labels[query_mask][local_index]),
                            "predicted_label": prediction,
                            "true_class": str(row["class_name"]),
                            "predicted_class": CLASS_NAMES[prediction],
                        }
                        for class_id, class_name in enumerate(CLASS_NAMES):
                            prediction_row[f"prob_{class_name}"] = float(query_probs[local_index, class_id])
                        predictions.append(prediction_row)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "source-selected complete-data Phi-OTDR spatial morphology v3",
        "evidence_status": "retrospective development; not independent confirmation",
        "protocol_sha256": expected_hash,
        "dataset_fingerprint_sha256": protocol["dataset_fingerprint_sha256"],
        "bundle_path": bundle_path.as_posix(),
        "bundle_sha256": _sha256(bundle_path),
        "models": list(models),
        "requested_specs": None if specs is None else list(specs),
        "result_count": len(results),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "spatial_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_predictions(output_dir / "spatial_target_predictions.csv", predictions)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["logistic"])
    parser.add_argument(
        "--specs",
        nargs="+",
        help="Optional exact view:estimator:ablation triples to evaluate",
    )
    args = parser.parse_args()
    result = run_experiment(
        bundle_path=args.bundle,
        manifests=args.manifests,
        protocol_path=args.protocol,
        protocol_hash_path=args.protocol_hash,
        output_dir=args.output_dir,
        models=args.models,
        specs=args.specs,
    )
    summary = {
        "result_count": result["result_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "payload_sha256": result["payload_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
