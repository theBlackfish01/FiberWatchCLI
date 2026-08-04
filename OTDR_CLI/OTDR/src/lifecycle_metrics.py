from __future__ import annotations

"""Reconstructible metrics shared by lifecycle experiment and analysis code."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .event_openworld_metrics import oscr_auc, raw_partial_auroc


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(1, keepdims=True)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    probability = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    confidence = probability.max(1)
    correct = probability.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            result += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def gate_diagnostics(
    gate: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object] | None:
    """Summarize learned context gates without treating them as causal effects."""
    values = np.asarray(gate, dtype=float)
    y = np.asarray(labels, dtype=int)
    if values.ndim != 2 or len(values) != len(y):
        raise ValueError("Gate values must be a matrix aligned with labels.")
    if not np.isfinite(values).all():
        return None
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "per_channel_mean": values.mean(0).tolist(),
        "per_class_mean": {
            str(class_id): float(values[y == class_id].mean())
            for class_id in sorted(np.unique(y))
        },
        "interpretation": (
            "Descriptive learned context-gate activation; not causal feature "
            "importance and not a normalized morphology/context contribution."
        ),
    }


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    positions: np.ndarray | None = None,
    predicted_positions: np.ndarray | None = None,
) -> dict[str, object]:
    y = np.asarray(labels, dtype=int)
    probability = softmax(logits)
    prediction = probability.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=list(range(8)), zero_division=0
    )
    one_hot = np.eye(8)[y]
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(y, probability, labels=list(range(8)))),
        "brier": float(np.square(probability - one_hot).sum(1).mean()),
        "ece_15": expected_calibration_error(probability, y),
        "confusion_matrix": confusion_matrix(y, prediction, labels=list(range(8))).tolist(),
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(8)
        },
    }
    if positions is not None and predicted_positions is not None:
        true_position = np.asarray(positions, dtype=float)
        predicted_position = np.asarray(predicted_positions, dtype=float)
        valid = np.isfinite(true_position) & np.isfinite(predicted_position)
        if valid.any():
            error = predicted_position[valid] - true_position[valid]
            result["localization_mae"] = float(np.abs(error).mean())
            result["localization_rmse"] = float(np.sqrt(np.square(error).mean()))
            result["localization_count"] = int(valid.sum())
    return result


def open_world_ranking_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    prediction: np.ndarray,
    *,
    holdout: tuple[int, int],
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(score, dtype=float)
    unknown = np.isin(y, holdout)
    binary = unknown.astype(int)
    return {
        "auroc": float(roc_auc_score(binary, values)),
        "aupr": float(average_precision_score(binary, values)),
        "pauroc_0_01": raw_partial_auroc(binary, values, 0.01),
        "pauroc_0_05": raw_partial_auroc(binary, values, 0.05),
        "oscr": oscr_auc(y, np.asarray(prediction, dtype=int), unknown, -values),
    }


def hard_prediction_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    *,
    base_class_ids: tuple[int, ...],
    enrolled_class_ids: tuple[int, ...],
) -> dict[str, object]:
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(prediction, dtype=int)
    base = np.isin(y, base_class_ids)
    novel = np.isin(y, enrolled_class_ids)
    base_accuracy = float((pred[base] == y[base]).mean())
    novel_accuracy = float((pred[novel] == y[novel]).mean())
    harmonic = 0.0 if base_accuracy + novel_accuracy == 0 else (
        2 * base_accuracy * novel_accuracy / (base_accuracy + novel_accuracy)
    )
    ids = tuple(sorted((*base_class_ids, *enrolled_class_ids)))
    per_class = {str(class_id): float((pred[y == class_id] == class_id).mean()) for class_id in ids if np.any(y == class_id)}
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "base_accuracy": base_accuracy,
        "enrolled_accuracy": novel_accuracy,
        "harmonic_mean": harmonic,
        "worst_enrolled_recall": min(per_class[str(class_id)] for class_id in enrolled_class_ids),
        "per_class_recall": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(ids)).tolist(),
    }
