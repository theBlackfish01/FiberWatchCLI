"""Evaluate development-selected supervised models once on final-query sessions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from .metrics import classification_metrics
from .neural_baseline import _evaluate, build_model
from .neural_data import ManifestWindowDataset


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> dict[str, object]:
    """Multiclass confidence calibration with fixed-width bins."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        include = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        count = int(include.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[include].mean())
        accuracy = float(correct[include].mean())
        ece += count / len(y_true) * abs(mean_confidence - accuracy)
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy,
        })
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return {
        "expected_calibration_error_15bin": float(ece),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1]))),
        "reliability_bins": rows,
    }


def risk_coverage_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Selective-classification error as progressively less-confident samples are retained."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(y_true)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(correct) + 1)
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    requested = {}
    for target in (0.80, 0.90, 0.95, 1.0):
        index = max(0, int(np.ceil(target * len(correct))) - 1)
        requested[f"risk_at_{int(target * 100)}pct_coverage"] = float(cumulative_risk[index])
    curve_indices = np.unique(np.linspace(0, len(correct) - 1, min(101, len(correct)), dtype=int))
    return {
        "area_under_risk_coverage": float(np.trapezoid(cumulative_risk, coverage)),
        **requested,
        "curve": [
            {"coverage": float(coverage[index]), "risk": float(cumulative_risk[index])}
            for index in curve_indices
        ],
    }


def session_probabilities(
    y_true: np.ndarray, sessions: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.unique(sessions.astype(str))
    session_true: list[int] = []
    session_prob: list[np.ndarray] = []
    for session in ordered:
        include = sessions.astype(str) == session
        labels = np.unique(y_true[include])
        if len(labels) != 1:
            raise ValueError(f"Session {session} has multiple labels")
        session_true.append(int(labels[0]))
        session_prob.append(probabilities[include].mean(axis=0))
    return np.asarray(session_true), np.vstack(session_prob), ordered


def evaluate_probabilities(
    y_true: np.ndarray, sessions: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    predictions = probabilities.argmax(axis=1)
    session_true, session_prob, session_ids = session_probabilities(y_true, sessions, probabilities)
    session_pred = session_prob.argmax(axis=1)
    result = {
        "window_metrics": classification_metrics(y_true, predictions),
        "session_metrics": classification_metrics(session_true, session_pred),
        "window_calibration": calibration_metrics(y_true, probabilities),
        "session_calibration": calibration_metrics(session_true, session_prob),
        "window_risk_coverage": risk_coverage_metrics(y_true, probabilities),
        "session_risk_coverage": risk_coverage_metrics(session_true, session_prob),
    }
    arrays = {
        "y_true": y_true, "probabilities": probabilities, "predictions": predictions,
        "sessions": sessions.astype(str), "session_ids": session_ids,
        "session_true": session_true, "session_probabilities": session_prob,
        "session_predictions": session_pred,
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--final-features", type=Path, required=True)
    parser.add_argument("--feature-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--neural-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_bundle = np.load(args.final_features, allow_pickle=False)
    if not np.all(final_bundle["partitions"].astype(str) == "final_query"):
        raise ValueError("Final feature bundle contains a non-final partition")
    saved_feature = joblib.load(args.feature_model)
    if not np.array_equal(saved_feature["feature_names"].astype(str), final_bundle["feature_names"].astype(str)):
        raise ValueError("Feature schema differs from development model")
    mask = saved_feature["feature_mask"]
    started = time.perf_counter()
    feature_prob = saved_feature["model"].predict_proba(final_bundle["features"][:, mask])
    feature_result, feature_arrays = evaluate_probabilities(
        final_bundle["labels"], final_bundle["sessions"].astype(str), feature_prob
    )
    feature_result["inference_seconds"] = time.perf_counter() - started
    np.savez_compressed(args.output_dir / "hist_gradient_boosting_predictions.npz", **feature_arrays)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for neural confirmatory inference")
    device = torch.device("cuda")
    neural_results: dict[str, object] = {}
    for model_name in ("cnn", "tcn", "tft"):
        model_dir = args.neural_root / f"{model_name}_seed20260805"
        checkpoint = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
        model, _ = build_model(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        dataset = ManifestWindowDataset(
            args.data_root, args.manifest, ("final_query",),
            normalization=str(checkpoint["normalization"]), temporal_pool=int(checkpoint["temporal_pool"]),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        started = time.perf_counter()
        raw = _evaluate(model, model_name, loader, device, nn.CrossEntropyLoss())
        result, arrays = evaluate_probabilities(raw["y_true"], raw["sessions"], raw["probabilities"])
        result.update({
            "inference_seconds": time.perf_counter() - started,
            "seed": int(checkpoint["seed"]), "selected_epoch": int(checkpoint["epoch"]),
            "normalization": str(checkpoint["normalization"]),
            "temporal_pool": int(checkpoint["temporal_pool"]),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
        neural_results[model_name] = result
        arrays["logits"] = raw["logits"]
        arrays["rel_paths"] = raw["paths"]
        np.savez_compressed(args.output_dir / f"{model_name}_predictions.npz", **arrays)
        print(f"[{model_name}] final window F1={result['window_metrics']['macro_f1']:.4f} "
              f"session F1={result['session_metrics']['macro_f1']:.4f}", flush=True)

    payload = {
        "protocol": "single untouched final-query evaluation of development-selected supervised models",
        "final_query_windows": int(len(final_bundle["labels"])),
        "final_query_sessions": int(len(np.unique(final_bundle["sessions"].astype(str)))),
        "feature_model": "full__hist_gradient_boosting",
        "feature_model_result": feature_result,
        "neural_results": neural_results,
        "limitations": [
            "Neural models use one training seed and are corrected baselines, not multi-seed estimates.",
            "No model, epoch, feature set, or calibration method was changed after final-query access.",
        ],
    }
    (args.output_dir / "confirmatory_supervised_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "feature": {"window": feature_result["window_metrics"], "session": feature_result["session_metrics"]},
        "neural": {name: {"window": row["window_metrics"]["macro_f1"], "session": row["session_metrics"]["macro_f1"]}
                   for name, row in neural_results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
