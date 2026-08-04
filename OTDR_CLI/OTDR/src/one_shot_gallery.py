from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model_functions.multi_similarity_siamese import MultiSimilaritySiamese


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    known_acceptance: float
    unknown_recall: float
    harmonic_mean: float
    normal_far_threshold: float | None
    curve: list[dict[str, float]]


@dataclass(frozen=True)
class ScoreNormalizer:
    center: float
    scale: float

    def transform(self, scores: np.ndarray | torch.Tensor) -> np.ndarray:
        values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        return (values.astype(float) - self.center) / self.scale

    def to_dict(self) -> dict[str, float]:
        return {"center": self.center, "scale": self.scale}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "ScoreNormalizer":
        return cls(center=float(payload["center"]), scale=float(payload["scale"]))


def fit_score_normalizer(known_scores: np.ndarray | torch.Tensor) -> ScoreNormalizer:
    values = known_scores.detach().cpu().numpy() if isinstance(known_scores, torch.Tensor) else np.asarray(known_scores)
    values = values.astype(float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Score normalization requires finite known-class scores.")
    center = float(np.median(values))
    scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    if scale < 1e-8:
        scale = float(values.std())
    if scale < 1e-8:
        scale = 1.0
    return ScoreNormalizer(center=center, scale=scale)


@dataclass(frozen=True)
class ReferenceGallery:
    embeddings: torch.Tensor
    labels: torch.Tensor
    row_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("Gallery embeddings must be a two-dimensional tensor.")
        if len(self.embeddings) != len(self.labels) or len(self.labels) != len(self.row_indices):
            raise ValueError("Gallery embeddings, labels, and row indices must have equal length.")
        if not len(self.embeddings):
            raise ValueError("A reference gallery cannot be empty.")

    def enroll(
        self,
        embeddings: torch.Tensor,
        *,
        class_id: int,
        row_indices: torch.Tensor,
    ) -> "ReferenceGallery":
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Enrollment embedding dimension does not match the gallery.")
        row_indices = row_indices.reshape(-1)
        if len(embeddings) != len(row_indices):
            raise ValueError("Enrollment rows and embeddings must have equal length.")
        labels = torch.full((len(embeddings),), int(class_id), dtype=torch.long, device=embeddings.device)
        target_device = self.embeddings.device
        return ReferenceGallery(
            embeddings=torch.cat((self.embeddings, embeddings.to(target_device))),
            labels=torch.cat((self.labels, labels.to(target_device))),
            row_indices=torch.cat((self.row_indices, row_indices.to(target_device, dtype=torch.long))),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": self.embeddings.detach().cpu(),
                "labels": self.labels.detach().cpu(),
                "row_indices": self.row_indices.detach().cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> "ReferenceGallery":
        payload = torch.load(path, map_location=device, weights_only=True)
        return cls(payload["embeddings"], payload["labels"], payload["row_indices"])


@torch.no_grad()
def pair_scores_against_gallery(
    model: MultiSimilaritySiamese,
    query_embeddings: torch.Tensor,
    gallery: ReferenceGallery,
    *,
    device: torch.device,
    query_batch_size: int = 256,
) -> torch.Tensor:
    """Return sigmoid pair probabilities with shape query x reference."""

    model.eval()
    references = gallery.embeddings.to(device)
    columns: list[torch.Tensor] = []
    for start in range(0, len(query_embeddings), query_batch_size):
        queries = query_embeddings[start : start + query_batch_size].to(device)
        q_count, r_count = len(queries), len(references)
        left = queries[:, None, :].expand(q_count, r_count, -1).reshape(-1, queries.shape[-1])
        right = references[None, :, :].expand(q_count, r_count, -1).reshape(-1, references.shape[-1])
        scores = model.score_embeddings(left, right).sigmoid().reshape(q_count, r_count)
        columns.append(scores.float().cpu())
    return torch.cat(columns) if columns else torch.empty((0, len(gallery.labels)))


def baseline_scores_against_gallery(
    query_embeddings: torch.Tensor,
    gallery: ReferenceGallery,
    *,
    method: str,
) -> torch.Tensor:
    queries = query_embeddings.detach().cpu().float()
    references = gallery.embeddings.detach().cpu().float()
    if method == "cosine_1nn":
        queries = torch.nn.functional.normalize(queries, dim=-1)
        references = torch.nn.functional.normalize(references, dim=-1)
        return queries @ references.T
    if method == "euclidean_1nn":
        return -torch.cdist(queries, references)
    raise ValueError(f"Unknown baseline method: {method}")


def classify_from_pair_scores(
    pair_scores: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    threshold: float,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate top-k references per class and reject low-confidence queries."""

    if pair_scores.ndim != 2 or pair_scores.shape[1] != len(gallery_labels):
        raise ValueError("Pair-score columns must match gallery labels.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    labels = gallery_labels.detach().cpu().long()
    classes = sorted(int(value) for value in labels.unique().tolist())
    class_scores = []
    for class_id in classes:
        candidates = pair_scores[:, labels == class_id]
        k = min(top_k, candidates.shape[1])
        class_scores.append(candidates.topk(k, dim=1).values.mean(dim=1))
    stacked = torch.stack(class_scores, dim=1)
    confidence, indices = stacked.max(dim=1)
    predicted = torch.tensor(classes, dtype=torch.long)[indices]
    accepted = confidence >= float(threshold)
    predicted = predicted.masked_fill(~accepted, -1)
    return predicted, confidence, accepted


def attach_semantic_suggestions(
    gallery_predictions: torch.Tensor,
    semantic_predictions: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    """Suggest a semantic label only for gallery-rejected traces.

    Suggestions are intentionally kept separate from accepted instance labels so
    a human can confirm the zero-day name before enrolling a reference.
    """

    if gallery_predictions.shape != semantic_predictions.shape:
        raise ValueError("Gallery and semantic prediction vectors must have matching shapes.")
    rejected = gallery_predictions == -1
    suggestions = gallery_predictions.clone()
    suggestions[rejected] = semantic_predictions[rejected]
    sources = ["semantic_suggestion" if value else "gallery" for value in rejected.tolist()]
    return suggestions, sources


def calibrate_unknown_threshold(
    confidence_scores: np.ndarray,
    is_known: np.ndarray,
    *,
    normal_scores: np.ndarray | None = None,
    max_normal_false_alarm: float = 0.01,
    max_candidates: int = 1001,
) -> ThresholdCalibration:
    """Choose a threshold by known/unknown H, with a reported normal-FAR point."""

    scores = np.asarray(confidence_scores, dtype=float)
    known = np.asarray(is_known, dtype=bool)
    if scores.ndim != 1 or scores.shape != known.shape or not len(scores):
        raise ValueError("Calibration scores and labels must be non-empty matching vectors.")
    if known.all() or (~known).all():
        raise ValueError("Calibration requires both known and pseudo-unknown examples.")
    unique = np.unique(scores)
    if len(unique) > max_candidates:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, max_candidates)))
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2.0, [np.nextafter(unique[-1], np.inf)])
    )
    curve: list[dict[str, float]] = []
    for threshold in candidates:
        accepted = scores >= threshold
        known_acceptance = float(accepted[known].mean())
        unknown_recall = float((~accepted[~known]).mean())
        harmonic = (
            0.0
            if known_acceptance + unknown_recall == 0
            else 2 * known_acceptance * unknown_recall / (known_acceptance + unknown_recall)
        )
        curve.append(
            {
                "threshold": float(threshold),
                "known_acceptance": known_acceptance,
                "unknown_recall": unknown_recall,
                "harmonic_mean": harmonic,
            }
        )
    best = max(curve, key=lambda row: (row["harmonic_mean"], row["unknown_recall"], -row["threshold"]))
    normal_far_threshold = None
    if normal_scores is not None and len(normal_scores):
        # Rejection is the false alarm here, so retain (1-FAR) of normal scores.
        normal_far_threshold = float(np.quantile(np.asarray(normal_scores, dtype=float), max_normal_false_alarm))
    return ThresholdCalibration(
        threshold=best["threshold"],
        known_acceptance=best["known_acceptance"],
        unknown_recall=best["unknown_recall"],
        harmonic_mean=best["harmonic_mean"],
        normal_far_threshold=normal_far_threshold,
        curve=curve,
    )
