from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def macro_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_ids: Sequence[int]) -> float:
    values = [float((y_pred[y_true == value] == value).mean()) for value in class_ids if np.any(y_true == value)]
    return float(np.mean(values)) if values else 0.0


def harmonic(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else float(2 * left * right / (left + right))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, class_ids: Sequence[int] | None = None) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    active = sorted(np.unique(y_true)) if class_ids is None else list(class_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=active, average="macro", zero_division=0)
    distribution = {str(value): int((y_pred == value).sum()) for value in sorted(np.unique(y_pred))}
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": macro_class_accuracy(y_true, y_pred, active),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
        "per_class_recall": {str(value): float((y_pred[y_true == value] == value).mean()) for value in active if np.any(y_true == value)},
        "prediction_distribution": distribution,
    }


def threshold_at_normal_far(normal_confidence: np.ndarray, far: float) -> float:
    if not 0 <= far < 1 or len(normal_confidence) == 0:
        raise ValueError("A non-empty normal calibration sample and FAR in [0,1) are required.")
    return float(np.quantile(np.asarray(normal_confidence), far, method="lower"))


def balanced_threshold(known_confidence: np.ndarray, pseudo_unknown_confidence: np.ndarray) -> float:
    known = np.asarray(known_confidence, dtype=float)
    unknown = np.asarray(pseudo_unknown_confidence, dtype=float)
    candidates = np.unique(np.quantile(np.concatenate([known, unknown]), np.linspace(0, 1, 401)))
    values = [((known >= threshold).mean() + (unknown < threshold).mean()) / 2 for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def open_set_metrics(*, is_known: np.ndarray, confidence: np.ndarray, predicted: np.ndarray,
                     true_labels: np.ndarray, threshold: float) -> dict[str, float]:
    known = np.asarray(is_known, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    accepted = confidence >= threshold
    normal = np.asarray(true_labels) == 0
    fpr, tpr, thresholds = roc_curve(known.astype(int), confidence)
    idx = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[idx[0]]) if len(idx) else 1.0
    correct_known = known & (np.asarray(predicted) == np.asarray(true_labels))
    order = np.argsort(confidence)[::-1]
    false_unknown_accepts = np.cumsum((~known)[order]) / max((~known).sum(), 1)
    correct_known_accepts = np.cumsum(correct_known[order]) / max(known.sum(), 1)
    oscr = float(np.trapezoid(correct_known_accepts[np.argsort(false_unknown_accepts)], np.sort(false_unknown_accepts)))
    return {
        "auroc": float(roc_auc_score(known, confidence)),
        "aupr": float(average_precision_score(known, confidence)),
        "fpr_at_95_known_tpr": fpr95,
        "known_acceptance": float(accepted[known].mean()),
        "unknown_recall": float((~accepted[~known]).mean()),
        "unknown_false_acceptance": float(accepted[~known].mean()),
        "normal_rejection_rate": float((~accepted[normal]).mean()),
        "oscr_auc": oscr,
    }


def post_enrollment_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, seen_ids: Sequence[int], unseen_ids: Sequence[int]) -> dict[str, object]:
    seen = macro_class_accuracy(y_true, y_pred, seen_ids)
    unseen = macro_class_accuracy(y_true, y_pred, unseen_ids)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": macro_class_accuracy(y_true, y_pred, sorted(np.unique(y_true))),
        "seen_accuracy": seen, "unseen_accuracy": unseen, "harmonic_mean": harmonic(seen, unseen),
        "rejection_rate": float((y_pred == -1).mean()),
        "per_class_accuracy": {str(value): float((y_pred[y_true == value] == value).mean()) for value in sorted(np.unique(y_true))},
    }


def conformal_p_values(calibration_nonconformity: np.ndarray, test_nonconformity: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration_nonconformity, dtype=float)
    test = np.asarray(test_nonconformity, dtype=float)
    if len(calibration) == 0:
        raise ValueError("Conformal calibration cannot be empty.")
    return (1.0 + (calibration[None, :] >= test[:, None]).sum(axis=1)) / (len(calibration) + 1.0)


def summarize_draws(rows: list[dict[str, object]]) -> dict[str, object]:
    keys = ["accuracy", "balanced_accuracy", "seen_accuracy", "unseen_accuracy", "harmonic_mean", "rejection_rate"]
    summary: dict[str, object] = {"draws": rows}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}
    return summary
