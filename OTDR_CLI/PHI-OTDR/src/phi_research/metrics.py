"""Shared classification metrics at window and recording-session levels."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.metrics import average_precision_score, roc_auc_score

from .data_contract import CLASS_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels = np.arange(len(CLASS_NAMES))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "worst_class_recall": float(np.min(recalls)),
        "per_class_recall": {name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "sample_count": int(len(y_true)),
    }


def aggregate_session_predictions(
    y_true: np.ndarray,
    sessions: Sequence[str],
    *,
    probabilities: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if probabilities is None and predictions is None:
        raise ValueError("probabilities or predictions are required")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, session in enumerate(sessions):
        indices[str(session)].append(index)
    session_true: list[int] = []
    session_pred: list[int] = []
    ordered_sessions: list[str] = []
    for session in sorted(indices):
        selected = np.asarray(indices[session], dtype=int)
        labels = np.unique(y_true[selected])
        if len(labels) != 1:
            raise ValueError(f"Session {session} spans labels {labels.tolist()}")
        if probabilities is not None:
            predicted = int(np.argmax(np.mean(probabilities[selected], axis=0)))
        else:
            counts = np.bincount(np.asarray(predictions)[selected], minlength=len(CLASS_NAMES))
            predicted = int(np.argmax(counts))
        ordered_sessions.append(session)
        session_true.append(int(labels[0]))
        session_pred.append(predicted)
    return np.asarray(session_true), np.asarray(session_pred), ordered_sessions


def harmonic_mean(left: float, right: float) -> float:
    return 0.0 if left + right <= 0 else float(2.0 * left * right / (left + right))


def calibrate_rejection_threshold(
    known_confidence: np.ndarray,
    pseudo_unknown_confidence: np.ndarray,
    *,
    target_known_acceptance: float = 0.95,
) -> dict[str, float]:
    known = np.asarray(known_confidence, dtype=float)
    unknown = np.asarray(pseudo_unknown_confidence, dtype=float)
    if known.size == 0 or unknown.size == 0:
        raise ValueError("Threshold calibration requires known and pseudo-unknown scores")
    candidates = np.unique(np.concatenate((known, unknown)))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        known_acceptance = float(np.mean(known >= threshold))
        unknown_recall = float(np.mean(unknown < threshold))
        h = harmonic_mean(known_acceptance, unknown_recall)
        candidate = (h, unknown_recall, known_acceptance, float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    quantile_threshold = float(np.quantile(known, 1.0 - target_known_acceptance, method="higher"))
    return {
        "balanced_threshold": best[3],
        "balanced_h": best[0],
        "balanced_unknown_recall": best[1],
        "balanced_known_acceptance": best[2],
        "known_acceptance_threshold": quantile_threshold,
        "target_known_acceptance": target_known_acceptance,
        "calibration_known_count": int(known.size),
        "calibration_pseudo_unknown_count": int(unknown.size),
    }


def open_set_metrics(
    confidence: np.ndarray,
    is_known: np.ndarray,
    known_correct: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    confidence = np.asarray(confidence, dtype=float)
    is_known = np.asarray(is_known, dtype=bool)
    known_correct = np.asarray(known_correct, dtype=bool)
    if not np.any(is_known) or not np.any(~is_known):
        raise ValueError("Open-set metrics require known and unknown examples")
    accepted = confidence >= threshold
    known_acceptance = float(np.mean(accepted[is_known]))
    unknown_recall = float(np.mean(~accepted[~is_known]))
    labels = (~is_known).astype(int)
    anomaly_score = -confidence
    thresholds = np.unique(confidence)
    fpr: list[float] = [0.0]
    ccr: list[float] = [0.0]
    for candidate in sorted(thresholds, reverse=True):
        candidate_accepted = confidence >= candidate
        fpr.append(float(np.mean(candidate_accepted[~is_known])))
        ccr.append(float(np.mean(candidate_accepted[is_known] & known_correct[is_known])))
    order = np.argsort(fpr)
    return {
        "known_acceptance": known_acceptance,
        "unknown_recall": unknown_recall,
        "detection_h": harmonic_mean(known_acceptance, unknown_recall),
        "unknown_auroc": float(roc_auc_score(labels, anomaly_score)),
        "unknown_aupr": float(average_precision_score(labels, anomaly_score)),
        "oscr": float(np.trapezoid(np.asarray(ccr)[order], np.asarray(fpr)[order])),
        "known_classification_accuracy": float(np.mean(known_correct[is_known])),
        "threshold": float(threshold),
    }
