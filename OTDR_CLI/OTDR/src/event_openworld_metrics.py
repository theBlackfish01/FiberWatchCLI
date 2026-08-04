from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from .study_metrics import harmonic, macro_class_accuracy


def higher_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="higher"))


@dataclass(frozen=True)
class ScoreNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, normal_components: np.ndarray, weights: Iterable[float]) -> "ScoreNormalizer":
        values = np.asarray(normal_components, dtype=float)
        mean = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - mean), axis=0)
        fallback = values.std(axis=0, ddof=1)
        scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
        w = np.asarray(list(weights), dtype=float)
        if len(w) != values.shape[1] or np.allclose(w, 0):
            raise ValueError("Fusion weights must match non-empty novelty components.")
        w = w / np.abs(w).sum()
        return cls(mean, scale, w)

    def transform(self, components: np.ndarray) -> np.ndarray:
        return ((np.asarray(components, dtype=float) - self.mean) / self.scale) @ self.weights


class NormalOnlyCalibrator:
    """Global, Mondrian, or normalized calibration fitted to validation normals only."""

    def __init__(self, mode: str = "global", *, bins: int = 5, minimum_bin_size: int = 128, shrinkage: float = 128.0) -> None:
        if mode not in {"global", "mondrian", "normalized"}:
            raise ValueError("calibration mode must be global, mondrian, or normalized")
        self.mode = mode
        self.bins = bins
        self.minimum_bin_size = minimum_bin_size
        self.shrinkage = shrinkage
        self._fitted = False

    def fit(self, score: np.ndarray, snr: np.ndarray) -> "NormalOnlyCalibrator":
        score = np.asarray(score, dtype=float)
        snr = np.asarray(snr, dtype=float)
        if len(score) != len(snr) or len(score) == 0 or not np.isfinite(score).all():
            raise ValueError("Finite, aligned, non-empty validation-normal score/SNR values are required.")
        self.global_scores = score.copy()
        edges = np.unique(np.quantile(snr, np.linspace(0, 1, self.bins + 1)))
        self.edges = edges if len(edges) >= 2 else np.asarray([-np.inf, np.inf])
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        assignment = np.clip(np.digitize(snr, self.edges[1:-1]), 0, len(self.edges) - 2)
        self.bin_scores: list[np.ndarray | None] = []
        self.bin_counts: list[int] = []
        self.bin_location: list[float] = []
        self.bin_scale: list[float] = []
        global_location = float(np.median(score))
        global_scale = float(1.4826 * np.median(np.abs(score - global_location))) or float(score.std()) or 1.0
        for index in range(len(self.edges) - 1):
            values = score[assignment == index]
            self.bin_counts.append(len(values))
            self.bin_scores.append(values.copy() if len(values) >= self.minimum_bin_size else None)
            weight = len(values) / (len(values) + self.shrinkage)
            local_location = float(np.median(values)) if len(values) else global_location
            local_scale = float(1.4826 * np.median(np.abs(values - local_location))) if len(values) else global_scale
            local_scale = local_scale or global_scale
            self.bin_location.append(weight * local_location + (1 - weight) * global_location)
            self.bin_scale.append(weight * local_scale + (1 - weight) * global_scale)
        if self.mode == "normalized":
            normalized = np.empty_like(score)
            for index in range(len(self.edges) - 1):
                mask = assignment == index
                normalized[mask] = (score[mask] - self.bin_location[index]) / max(self.bin_scale[index], 1e-8)
            self.normalized_scores = normalized
        self._fitted = True
        return self

    def threshold(self, snr: np.ndarray, far: float) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator is not fitted.")
        if not 0 < far < 1:
            raise ValueError("FAR must be in (0,1).")
        snr = np.asarray(snr, dtype=float)
        global_threshold = higher_quantile(self.global_scores, 1 - far)
        if self.mode == "global":
            return np.full(len(snr), global_threshold)
        assignment = np.clip(np.digitize(snr, self.edges[1:-1]), 0, len(self.edges) - 2)
        result = np.empty(len(snr), dtype=float)
        for index in range(len(self.edges) - 1):
            mask = assignment == index
            if self.mode == "mondrian":
                values = self.bin_scores[index]
                result[mask] = higher_quantile(values, 1 - far) if values is not None else global_threshold
            else:
                q = higher_quantile(self.normalized_scores, 1 - far)
                result[mask] = self.bin_location[index] + max(self.bin_scale[index], 1e-8) * q
        return result

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "validation_normal_count": len(self.global_scores),
            "bin_edges": [float(value) if np.isfinite(value) else None for value in self.edges],
            "bin_counts": self.bin_counts,
            "active_mondrian_bins": [values is not None for values in self.bin_scores],
            "minimum_bin_size": self.minimum_bin_size,
            "small_bin_fallback": "global" if self.mode == "mondrian" else "shrink_to_global",
        }


def raw_partial_auroc(y_true: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(y_true, dtype=int), np.asarray(score, dtype=float))
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("max_fpr must be in (0,1].")
    inside = fpr < max_fpr
    x = np.concatenate([fpr[inside], [max_fpr]])
    y = np.concatenate([tpr[inside], [np.interp(max_fpr, fpr, tpr)]])
    return float(np.trapezoid(y, x) / max_fpr)


def oscr_auc(y_true: np.ndarray, predicted: np.ndarray, unknown_mask: np.ndarray, known_confidence: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    predicted = np.asarray(predicted)
    unknown = np.asarray(unknown_mask, dtype=bool)
    confidence = np.asarray(known_confidence, dtype=float)
    order = np.argsort(confidence)[::-1]
    correct = (~unknown) & (predicted == y_true)
    ccr = np.cumsum(correct[order]) / max((~unknown).sum(), 1)
    fpr = np.cumsum(unknown[order]) / max(unknown.sum(), 1)
    return float(np.trapezoid(ccr, fpr))


def evaluate_zero_day(
    *,
    validation_normal_score: np.ndarray,
    validation_normal_snr: np.ndarray,
    test_score: np.ndarray,
    test_snr: np.ndarray,
    true_labels: np.ndarray,
    predicted: np.ndarray,
    holdout: tuple[int, int],
    calibration: str,
    far_points: tuple[float, ...] = (0.01, 0.02, 0.05),
    minimum_bin_size: int = 128,
) -> dict[str, object]:
    true_labels = np.asarray(true_labels, dtype=int)
    unknown = np.isin(true_labels, holdout)
    normal = true_labels == 0
    known = ~unknown
    if not unknown.any() or not normal.any():
        raise ValueError("Zero-day evaluation needs held-out unknowns and test normals.")
    calibrator = NormalOnlyCalibrator(calibration, minimum_bin_size=minimum_bin_size).fit(
        validation_normal_score, validation_normal_snr
    )
    operating: dict[str, object] = {}
    for far in far_points:
        threshold = calibrator.threshold(test_snr, far)
        rejected = test_score > threshold
        per_fault = {str(value): float(rejected[true_labels == value].mean()) for value in holdout}
        operating[f"far_{far:.3f}"] = {
            "target_normal_far": far,
            "observed_normal_far": float(rejected[normal].mean()),
            "unknown_recall": float(rejected[unknown].mean()),
            "known_acceptance": float((~rejected[known]).mean()),
            "per_fault_recall": per_fault,
            "worst_fault_recall": min(per_fault.values()),
            "threshold_mean": float(threshold.mean()),
        }
    binary = unknown.astype(int)
    fpr, tpr, _ = roc_curve(binary, test_score)
    idx = np.flatnonzero(tpr >= 0.95)
    known_acceptance_threshold = higher_quantile(test_score[known], 0.95)
    return {
        "calibration": calibrator.describe(),
        "operating_points": operating,
        "pauroc_0_01": raw_partial_auroc(binary, test_score, 0.01),
        "pauroc_0_05": raw_partial_auroc(binary, test_score, 0.05),
        "auroc": float(roc_auc_score(binary, test_score)),
        "aupr": float(average_precision_score(binary, test_score)),
        "fpr_at_95_unknown_tpr": float(fpr[idx[0]]) if len(idx) else 1.0,
        "unknown_false_acceptance_at_95_known_acceptance": float((test_score[unknown] <= known_acceptance_threshold).mean()),
        "oscr": oscr_auc(true_labels, predicted, unknown, -test_score),
    }


def semantic_metrics(logits: np.ndarray, labels: np.ndarray, holdout: tuple[int, int]) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    unseen_mask = np.isin(labels, holdout)
    strict_local = logits[unseen_mask][:, list(holdout)].argmax(1)
    strict_pred = np.asarray(holdout)[strict_local]
    strict_true = labels[unseen_mask]
    strict_recall = {str(value): float((strict_pred[strict_true == value] == value).mean()) for value in holdout}
    all_pred = logits.argmax(1)
    seen_ids = sorted(set(range(8)) - set(holdout))
    seen_accuracy = macro_class_accuracy(labels, all_pred, seen_ids)
    unseen_accuracy = macro_class_accuracy(labels, all_pred, holdout)
    return {
        "strict_zsl_balanced_accuracy": float(np.mean(list(strict_recall.values()))),
        "strict_per_class_recall": strict_recall,
        "strict_prediction_distribution": {str(value): int((strict_pred == value).sum()) for value in holdout},
        "strict_confusion": confusion_matrix(strict_true, strict_pred, labels=list(holdout)).tolist(),
        "gzsl_seen_accuracy": seen_accuracy,
        "gzsl_unseen_accuracy": unseen_accuracy,
        "gzsl_harmonic_mean": harmonic(seen_accuracy, unseen_accuracy),
        "gzsl_class_collapse_count": int(sum((all_pred == value).sum() == 0 for value in range(8))),
    }
