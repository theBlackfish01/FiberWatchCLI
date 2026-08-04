from __future__ import annotations

"""Frozen-encoder few-shot enrollment methods for CFE-OTDR."""

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_functions.zero_shot import require_cuda


PrototypeMethod = Literal["mean", "medoid", "median", "quality_weighted"]
DistanceMetric = Literal["cosine", "euclidean", "diagonal_mahalanobis"]


@dataclass(frozen=True)
class ProjectionAdapterConfig:
    steps: int = 80
    learning_rate: float = 5e-3
    preservation_weight: float = 5.0
    identity_weight: float = 0.1
    temperature: float = 0.1
    base_examples_per_class: int = 256
    seed: int = 42


def _matrix(values: np.ndarray, *, name: str = "embeddings") -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite, non-empty 2D matrix.")
    return result


def normalized(values: np.ndarray) -> np.ndarray:
    x = _matrix(values)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def support_prototype(
    embeddings: np.ndarray,
    *,
    method: PrototypeMethod = "mean",
    quality: np.ndarray | None = None,
) -> np.ndarray:
    x = _matrix(embeddings)
    if method == "mean":
        result = x.mean(0)
    elif method == "median":
        result = np.median(x, axis=0)
    elif method == "medoid":
        distance = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        result = x[distance.sum(1).argmin()]
    elif method == "quality_weighted":
        if quality is None:
            raise ValueError("quality_weighted requires support quality.")
        weights = np.asarray(quality, dtype=float)
        if weights.shape != (len(x),) or not np.isfinite(weights).all():
            raise ValueError("Support quality must be finite and aligned.")
        weights = np.exp(weights - weights.max())
        weights /= np.clip(weights.sum(), 1e-12, None)
        result = (x * weights[:, None]).sum(0)
    else:
        raise ValueError(f"Unknown prototype method: {method}")
    return result / max(np.linalg.norm(result), 1e-12)


def teen_calibrate(
    novel_prototype: np.ndarray,
    base_prototypes: np.ndarray,
    *,
    alpha: float = 0.2,
    temperature: float = 0.5,
) -> np.ndarray:
    """Move a novel prototype toward related base prototypes without query access."""
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError("TEEN alpha must be in [0,1] and temperature positive.")
    novel = np.asarray(novel_prototype, dtype=float).reshape(1, -1)
    base = _matrix(base_prototypes, name="base_prototypes")
    if novel.shape[1] != base.shape[1] or not np.isfinite(novel).all():
        raise ValueError("Novel and base prototypes must be finite and aligned.")
    novel_n = normalized(novel)[0]
    base_n = normalized(base)
    logits = base_n @ novel_n / temperature
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    prior = weights @ base_n
    calibrated = (1 - alpha) * novel_n + alpha * prior
    return calibrated / max(np.linalg.norm(calibrated), 1e-12)


@dataclass(frozen=True)
class BaseMetric:
    diagonal_precision: np.ndarray

    @classmethod
    def fit(cls, base_embeddings: np.ndarray) -> "BaseMetric":
        x = _matrix(base_embeddings, name="base_embeddings")
        variance = np.var(x, axis=0, ddof=1)
        positive = variance[variance > 0]
        target = np.median(positive) if len(positive) else 1.0
        shrunk = 0.9 * variance + 0.1 * target
        return cls(1.0 / np.clip(shrunk, 1e-8, None))


@dataclass(frozen=True)
class EnrollmentSession:
    """Immutable class prototypes; enrollment returns a new session."""

    class_ids: tuple[int, ...]
    prototypes: tuple[np.ndarray, ...]
    metric: DistanceMetric
    base_metric: BaseMetric
    base_class_ids: tuple[int, ...]
    enrollment_history: tuple[dict[str, object], ...] = ()

    @classmethod
    def from_base(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        metric: DistanceMetric = "cosine",
    ) -> "EnrollmentSession":
        x = _matrix(embeddings)
        y = np.asarray(labels, dtype=int)
        if len(x) != len(y):
            raise ValueError("Base embeddings and labels must align.")
        ids = tuple(sorted(int(value) for value in np.unique(y)))
        prototypes = tuple(support_prototype(x[y == class_id], method="mean") for class_id in ids)
        return cls(ids, prototypes, metric, BaseMetric.fit(x), ids)

    def enroll(
        self,
        class_id: int,
        support_embeddings: np.ndarray,
        *,
        method: PrototypeMethod = "mean",
        quality: np.ndarray | None = None,
        teen_alpha: float = 0.0,
        teen_temperature: float = 0.5,
        support_group_ids: tuple[str, ...] | None = None,
    ) -> "EnrollmentSession":
        if class_id in self.class_ids:
            raise ValueError(f"Class {class_id} is already enrolled.")
        support = _matrix(support_embeddings, name="support_embeddings")
        if support.shape[1] != self.prototypes[0].shape[0]:
            raise ValueError("Support embedding dimension differs from base prototypes.")
        if support_group_ids is not None and len(set(support_group_ids)) != len(support):
            raise ValueError("Support groups must be distinct.")
        prototype = support_prototype(support, method=method, quality=quality)
        if teen_alpha:
            base_indices = [self.class_ids.index(class_id_) for class_id_ in self.base_class_ids]
            base = np.stack([self.prototypes[index] for index in base_indices])
            prototype = teen_calibrate(
                prototype, base, alpha=teen_alpha, temperature=teen_temperature
            )
        event = {
            "class_id": int(class_id),
            "shots": len(support),
            "prototype_method": method,
            "teen_alpha": teen_alpha,
            "teen_temperature": teen_temperature,
            "support_group_ids": list(support_group_ids or ()),
            "query_adaptation": False,
        }
        return EnrollmentSession(
            class_ids=(*self.class_ids, int(class_id)),
            prototypes=(*self.prototypes, prototype.copy()),
            metric=self.metric,
            base_metric=self.base_metric,
            base_class_ids=self.base_class_ids,
            enrollment_history=(*self.enrollment_history, event),
        )

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        x = _matrix(embeddings)
        prototypes = np.stack(self.prototypes)
        if x.shape[1] != prototypes.shape[1]:
            raise ValueError("Query dimension differs from prototypes.")
        if self.metric == "cosine":
            return 1.0 - normalized(x) @ normalized(prototypes).T
        difference = x[:, None, :] - prototypes[None, :, :]
        if self.metric == "euclidean":
            return np.linalg.norm(difference, axis=2)
        if self.metric == "diagonal_mahalanobis":
            return np.sqrt(np.maximum((np.square(difference) * self.base_metric.diagonal_precision).sum(2), 0))
        raise ValueError(f"Unknown distance metric: {self.metric}")

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        nearest = self.distances(embeddings).argmin(1)
        return np.asarray(self.class_ids, dtype=int)[nearest]

    def predict_proba(
        self,
        embeddings: np.ndarray,
        *,
        temperature: float,
        n_classes: int = 8,
    ) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("Distance temperature must be positive.")
        logits = -self.distances(embeddings) / temperature
        logits -= logits.max(1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(1, keepdims=True)
        aligned = np.zeros((len(probability), n_classes), dtype=float)
        aligned[:, np.asarray(self.class_ids, dtype=int)] = probability
        return aligned

    @property
    def storage_bytes(self) -> int:
        return int(sum(value.nbytes for value in self.prototypes) + self.base_metric.diagonal_precision.nbytes)


def sequential_orders(holdout: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    left, right = (int(value) for value in holdout)
    if left == right:
        raise ValueError("Sequential enrollment requires two distinct classes.")
    return (left, right), (right, left)


def incremental_metrics(
    labels: np.ndarray,
    before_prediction: np.ndarray,
    after_prediction: np.ndarray,
    *,
    base_class_ids: tuple[int, ...],
    enrolled_class_ids: tuple[int, ...],
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    before = np.asarray(before_prediction, dtype=int)
    after = np.asarray(after_prediction, dtype=int)
    base = np.isin(y, base_class_ids)
    enrolled = np.isin(y, enrolled_class_ids)
    before_base = float((before[base] == y[base]).mean()) if base.any() else float("nan")
    after_base = float((after[base] == y[base]).mean()) if base.any() else float("nan")
    novel = float((after[enrolled] == y[enrolled]).mean()) if enrolled.any() else float("nan")
    harmonic = 0.0 if after_base + novel == 0 else 2 * after_base * novel / (after_base + novel)
    return {
        "base_accuracy_before": before_base,
        "base_accuracy_after": after_base,
        "enrolled_accuracy": novel,
        "harmonic_mean": harmonic,
        "forgetting": max(0.0, before_base - after_base),
        "backward_transfer": after_base - before_base,
        "retention_ratio": after_base / before_base if before_base > 0 else float("nan"),
    }


def fit_distance_temperature(
    session: EnrollmentSession,
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
) -> float:
    """Fit one scalar temperature using known calibration samples only."""
    y = np.asarray(labels, dtype=int)
    if not set(np.unique(y)).issubset(set(session.class_ids)):
        raise ValueError("Temperature labels must belong to the session's base classes.")
    grid = (
        np.logspace(-2.5, 1.0, 72)
        if candidates is None
        else np.asarray(candidates, dtype=float)
    )
    if grid.ndim != 1 or len(grid) == 0 or np.any(grid <= 0):
        raise ValueError("Temperature candidates must be a positive vector.")
    losses = []
    for temperature in grid:
        probability = session.predict_proba(
            embeddings, temperature=float(temperature)
        )
        losses.append(
            float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1)).mean())
        )
    return float(grid[int(np.argmin(losses))])


def projection_adapter_predict(
    base_embeddings: np.ndarray,
    base_labels: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    device: torch.device | str,
    config: ProjectionAdapterConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a small support-conditioned projection while freezing the encoder.

    Only base embeddings and declared support embeddings enter optimization.
    Query embeddings are transformed once after fitting and never contribute to
    gradients, prototypes, early stopping, or model selection.
    """
    device = require_cuda(str(device))
    cfg = config or ProjectionAdapterConfig()
    if cfg.steps < 1 or cfg.base_examples_per_class < 1:
        raise ValueError("Adapter steps and base_examples_per_class must be positive.")
    base = _matrix(base_embeddings, name="base_embeddings")
    support = _matrix(support_embeddings, name="support_embeddings")
    query = _matrix(query_embeddings, name="query_embeddings")
    base_y = np.asarray(base_labels, dtype=int)
    support_y = np.asarray(support_labels, dtype=int)
    if len(base) != len(base_y) or len(support) != len(support_y):
        raise ValueError("Adapter embeddings and labels must align.")
    if base.shape[1] != support.shape[1] or base.shape[1] != query.shape[1]:
        raise ValueError("Adapter embedding dimensions must match.")
    if set(np.unique(base_y)) & set(np.unique(support_y)):
        raise ValueError("Adapter support classes must be novel relative to base classes.")

    rng = np.random.default_rng(cfg.seed)
    sampled = np.concatenate([
        rng.choice(
            np.flatnonzero(base_y == class_id),
            size=min(
                cfg.base_examples_per_class,
                int((base_y == class_id).sum()),
            ),
            replace=False,
        )
        for class_id in sorted(np.unique(base_y))
    ])
    fit_x = np.vstack((base[sampled], support)).astype(np.float32)
    fit_y = np.r_[base_y[sampled], support_y]
    class_ids = tuple(sorted(int(value) for value in np.unique(fit_y)))
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    targets = np.asarray([class_to_index[int(value)] for value in fit_y], dtype=np.int64)
    anchors = np.stack([
        support_prototype(fit_x[fit_y == class_id], method="mean")
        for class_id in class_ids
    ]).astype(np.float32)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    dimension = base.shape[1]
    adapter = nn.Linear(dimension, dimension, bias=False, device=device)
    with torch.no_grad():
        adapter.weight.copy_(
            torch.eye(dimension, device=device, dtype=adapter.weight.dtype)
        )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=cfg.learning_rate, weight_decay=0.0
    )
    x_tensor = torch.from_numpy(fit_x).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    anchor_tensor = F.normalize(torch.from_numpy(anchors).to(device), dim=1)
    base_count = len(sampled)
    identity = torch.eye(dimension, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(cfg.steps):
        transformed = adapter(x_tensor)
        logits = F.normalize(transformed, dim=1) @ anchor_tensor.T
        classification = F.cross_entropy(
            logits / cfg.temperature, target_tensor
        )
        preservation = F.mse_loss(
            transformed[:base_count], x_tensor[:base_count]
        )
        identity_penalty = F.mse_loss(adapter.weight, identity)
        loss = (
            classification
            + cfg.preservation_weight * preservation
            + cfg.identity_weight * identity_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    @torch.no_grad()
    def transform(values: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        result = []
        for start in range(0, len(values), batch_size):
            tensor = torch.from_numpy(
                np.asarray(values[start:start + batch_size], dtype=np.float32)
            ).to(device)
            result.append(
                F.normalize(adapter(tensor), dim=1).cpu().numpy()
            )
        return np.vstack(result)

    transformed_base = transform(base)
    transformed_support = transform(support)
    transformed_query = transform(query)
    prototype_ids = tuple(sorted((*np.unique(base_y), *np.unique(support_y))))
    prototypes = np.stack([
        support_prototype(
            transformed_base[base_y == class_id]
            if class_id in set(np.unique(base_y))
            else transformed_support[support_y == class_id],
            method="mean",
        )
        for class_id in prototype_ids
    ])
    prediction = np.asarray(prototype_ids, dtype=int)[
        (normalized(transformed_query) @ normalized(prototypes).T).argmax(1)
    ]
    metadata = {
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "preservation_weight": cfg.preservation_weight,
        "identity_weight": cfg.identity_weight,
        "temperature": cfg.temperature,
        "base_examples_per_class": cfg.base_examples_per_class,
        "seed": cfg.seed,
        "parameter_count": dimension * dimension,
        "training_seconds": training_seconds,
        "final_training_loss": final_loss,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device": str(device),
        "encoder_frozen": True,
        "query_used_for_training": False,
    }
    return prediction, metadata
