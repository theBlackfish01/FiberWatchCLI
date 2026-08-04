from __future__ import annotations

"""Required classical and dependency-gated CFE baselines."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from .model_functions.zero_shot import require_cuda


def balanced_context_indices(labels: np.ndarray, *, total: int, seed: int) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    ids = sorted(np.unique(y))
    if total < len(ids):
        raise ValueError("Context must contain every class.")
    rng = np.random.default_rng(seed)
    per_class = total // len(ids)
    rows = []
    for class_id in ids:
        candidates = np.flatnonzero(y == class_id)
        rows.append(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    result = np.concatenate(rows)
    remainder = total - len(result)
    if remainder > 0:
        remaining = np.setdiff1d(np.arange(len(y)), result, assume_unique=False)
        result = np.r_[result, rng.choice(remaining, min(remainder, len(remaining)), replace=False)]
    rng.shuffle(result)
    return result.astype(np.int64)


@dataclass(frozen=True)
class NearestNeighborReference:
    features: np.ndarray
    labels: np.ndarray
    metric: str
    diagonal_precision: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        metric: str,
        max_reference: int = 2048,
        seed: int = 42,
    ) -> "NearestNeighborReference":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("Finite aligned reference features and labels are required.")
        if metric not in {"cosine", "euclidean", "diagonal_mahalanobis"}:
            raise ValueError("Unsupported 1NN metric.")
        indices = balanced_context_indices(y, total=min(max_reference, len(y)), seed=seed)
        precision = None
        if metric == "diagonal_mahalanobis":
            variance = np.var(x, axis=0, ddof=1)
            positive = variance[variance > 0]
            target = np.median(positive) if len(positive) else 1.0
            precision = 1 / np.clip(0.9 * variance + 0.1 * target, 1e-8, None)
        return cls(x[indices], y[indices], metric, precision)

    def _distance(self, query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float64)
        g = np.asarray(gallery, dtype=np.float64)
        if self.metric == "cosine":
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
            g = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-12, None)
            return 1 - q @ g.T
        if self.metric == "euclidean":
            squared = np.square(q).sum(1)[:, None] + np.square(g).sum(1)[None, :] - 2 * q @ g.T
            return np.sqrt(np.maximum(squared, 0))
        scale = np.sqrt(self.diagonal_precision)
        q_scaled, g_scaled = q * scale, g * scale
        squared = (
            np.square(q_scaled).sum(1)[:, None]
            + np.square(g_scaled).sum(1)[None, :]
            - 2 * q_scaled @ g_scaled.T
        )
        return np.sqrt(np.maximum(squared, 0))

    def predict(
        self,
        query: np.ndarray,
        *,
        support_features: np.ndarray | None = None,
        support_labels: np.ndarray | None = None,
        chunk_size: int = 512,
    ) -> np.ndarray:
        gallery = self.features
        labels = self.labels
        if support_features is not None:
            support = np.asarray(support_features, dtype=np.float64)
            support_y = np.asarray(support_labels, dtype=int)
            if support.ndim != 2 or len(support) != len(support_y) or support.shape[1] != gallery.shape[1]:
                raise ValueError("Support features and labels must align with the base gallery.")
            gallery = np.vstack((gallery, support))
            labels = np.r_[labels, support_y]
        q = np.asarray(query, dtype=np.float64)
        result = []
        for start in range(0, len(q), chunk_size):
            distance = self._distance(q[start:start + chunk_size], gallery)
            result.append(labels[distance.argmin(1)])
        return np.concatenate(result)

    def nearest(self, query: np.ndarray, *, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float64)
        nearest_distance, nearest_label = [], []
        for start in range(0, len(q), chunk_size):
            distance = self._distance(q[start:start + chunk_size], self.features)
            index = distance.argmin(1)
            nearest_distance.append(distance[np.arange(len(index)), index])
            nearest_label.append(self.labels[index])
        return np.concatenate(nearest_distance), np.concatenate(nearest_label)

    def combine_support(
        self,
        query: np.ndarray,
        base_distance: np.ndarray,
        base_label: np.ndarray,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        *,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        q = np.asarray(query, dtype=np.float64)
        support = np.asarray(support_features, dtype=np.float64)
        labels = np.asarray(support_labels, dtype=int)
        result = np.asarray(base_label, dtype=int).copy()
        for start in range(0, len(q), chunk_size):
            stop = start + chunk_size
            distance = self._distance(q[start:stop], support)
            index = distance.argmin(1)
            support_distance = distance[np.arange(len(index)), index]
            replace_mask = support_distance < np.asarray(base_distance)[start:stop]
            segment = result[start:stop].copy()
            segment[replace_mask] = labels[index[replace_mask]]
            result[start:stop] = segment
        return result


def _cache_hash(model: object) -> str | None:
    candidates = []
    for value in vars(model).values():
        if isinstance(value, (str, Path)) and Path(value).is_file():
            candidates.append(Path(value))
    if not candidates:
        return None
    digest = hashlib.sha256()
    for path in sorted(candidates):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_tabpfn_v2_ensemble(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    device: torch.device | str,
    context_size: int = 4096,
    seeds: tuple[int, ...] = (42, 123, 2026),
) -> tuple[np.ndarray, dict[str, object]]:
    """Run balanced deterministic TabPFN contexts or fail with an explicit dependency gate."""
    device = require_cuda(str(device))
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError(
            "TabPFN-v2 baseline is dependency-gated: install a CUDA-capable `tabpfn` package "
            "and locally available weights before running this arm."
        ) from exc
    started = time.perf_counter()
    probabilities = []
    models = []
    for seed in seeds:
        indices = balanced_context_indices(train_y, total=min(context_size, len(train_y)), seed=seed)
        model = TabPFNClassifier(device=str(device))
        model.fit(np.asarray(train_x)[indices], np.asarray(train_y)[indices])
        probabilities.append(model.predict_proba(query_x))
        models.append(model)
    mean_probability = np.mean(probabilities, axis=0)
    metadata = {
        "package_version": importlib.metadata.version("tabpfn"),
        "model_class": "TabPFNClassifier",
        "device": str(device),
        "context_size": min(context_size, len(train_y)),
        "context_seeds": list(seeds),
        "ensemble_size": len(seeds),
        "duration_seconds": time.perf_counter() - started,
        "weight_hash": _cache_hash(models[0]),
        "cache_source": "local_tabpfn_cache",
    }
    return mean_probability, metadata
