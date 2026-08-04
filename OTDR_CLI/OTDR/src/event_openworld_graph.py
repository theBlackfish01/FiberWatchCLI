from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch.nn import functional as F

from .event_openworld_training import SGMEConfig
from .model_functions.zero_shot import require_cuda


@dataclass(frozen=True)
class GraphEnrollmentResult:
    prototypes: torch.Tensor
    variances: torch.Tensor
    accepted_indices: torch.Tensor
    accepted_labels: torch.Tensor
    propagation: torch.Tensor
    metadata: dict[str, object]


def class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, class_count: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    dim = embeddings.shape[1]
    prototypes = torch.zeros(class_count, dim, dtype=embeddings.dtype, device=embeddings.device)
    variances = torch.ones_like(prototypes)
    for class_id in range(class_count):
        values = embeddings[labels == class_id]
        if len(values):
            prototypes[class_id] = F.normalize(values.mean(0), dim=0)
            if len(values) > 1:
                variances[class_id] = values.var(0, unbiased=False).clamp_min(1e-4)
    return prototypes, variances


def mutual_knn_graph(embeddings: torch.Tensor, *, k: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(embeddings.float(), dim=-1)
    similarity = values @ values.T
    similarity.fill_diagonal_(-torch.inf)
    k = min(k, max(1, len(values) - 1))
    nearest = similarity.topk(k, dim=1).indices
    directed = torch.zeros_like(similarity, dtype=torch.bool)
    directed.scatter_(1, nearest, True)
    mutual = directed & directed.T
    weights = torch.where(mutual, torch.exp(similarity / temperature), torch.zeros_like(similarity))
    row_sum = weights.sum(1, keepdim=True)
    isolated = row_sum.squeeze(1) == 0
    if isolated.any():
        fallback = directed & isolated[:, None]
        weights = torch.where(fallback, torch.exp(similarity / temperature), weights)
        row_sum = weights.sum(1, keepdim=True)
    normalized = weights / row_sum.clamp_min(1e-8)
    indices = torch.nonzero(normalized > 0, as_tuple=False).T
    with torch.sparse.check_sparse_tensor_invariants():
        sparse = torch.sparse_coo_tensor(indices, normalized[indices[0], indices[1]], normalized.shape,
                                         device=normalized.device).coalesce()
    return sparse, nearest


def graph_propagate(
    transition: torch.Tensor,
    seed_labels: torch.Tensor,
    seed_mask: torch.Tensor,
    *,
    alpha: float,
    iterations: int,
    class_count: int = 8,
) -> torch.Tensor:
    initial = torch.zeros(len(seed_labels), class_count, device=transition.device)
    initial[seed_mask] = F.one_hot(seed_labels[seed_mask], class_count).float()
    values = initial.clone()
    for _ in range(iterations):
        propagated = torch.sparse.mm(transition, values) if transition.is_sparse else transition @ values
        values = alpha * propagated + (1 - alpha) * initial
        values[seed_mask] = initial[seed_mask]
    return values / values.sum(1, keepdim=True).clamp_min(1e-8)


def seeded_graph_enrollment(
    *,
    seen_anchor_embeddings: torch.Tensor,
    seen_anchor_labels: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    adaptation_embeddings: torch.Tensor,
    semantic_probabilities: torch.Tensor,
    augmentation_probabilities: torch.Tensor,
    holdout: tuple[int, int],
    device: torch.device,
    config: SGMEConfig,
) -> GraphEnrollmentResult:
    device = require_cuda(str(device))
    started = time.perf_counter()
    anchor = seen_anchor_embeddings.to(device)
    anchor_y = seen_anchor_labels.to(device)
    reference = reference_embeddings.to(device)
    reference_y = reference_labels.to(device)
    adaptation = adaptation_embeddings.to(device)
    semantic = semantic_probabilities.to(device)
    augmentation = augmentation_probabilities.to(device)
    embeddings = torch.cat([anchor, reference, adaptation], dim=0)
    seed_labels = torch.cat([anchor_y, reference_y, torch.zeros(len(adaptation), dtype=torch.long, device=device)])
    seed_mask = torch.zeros(len(embeddings), dtype=torch.bool, device=device)
    seed_mask[:len(anchor) + len(reference)] = True
    transition, nearest = mutual_knn_graph(embeddings, k=config.k_neighbors, temperature=config.graph_temperature)
    propagation = graph_propagate(transition, seed_labels, seed_mask, alpha=config.propagation_alpha,
                                  iterations=config.graph_iterations)
    start = len(anchor) + len(reference)
    adaptation_distribution = propagation[start:]
    confidence, proposed = adaptation_distribution.max(1)
    neighbor_labels = propagation[nearest[start:]].argmax(-1)
    agreement = (neighbor_labels == proposed[:, None]).float().mean(1)
    semantic_confidence = semantic.gather(1, proposed[:, None]).squeeze(1)
    augmentation_confidence = augmentation.gather(1, proposed[:, None]).squeeze(1)
    augmentation_agrees = augmentation.argmax(1) == proposed
    unseen_proposal = torch.zeros_like(proposed, dtype=torch.bool)
    for class_id in holdout:
        unseen_proposal |= proposed == class_id
    seen_ids = sorted(set(range(8)) - set(holdout))
    seen_mass = semantic[:, seen_ids].amax(1)
    unseen_mass = semantic[:, list(holdout)].amax(1)
    seen_rejected = unseen_mass - seen_mass >= config.seen_rejection_threshold
    accepted = unseen_proposal & (confidence >= config.confidence_threshold) & (
        agreement >= config.agreement_threshold
    ) & (semantic_confidence >= config.semantic_threshold) & (
        augmentation_confidence >= config.augmentation_threshold
    ) & augmentation_agrees & seen_rejected
    accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten()
    accepted_labels = proposed[accepted]
    enrolled_embeddings = torch.cat([anchor, reference, adaptation[accepted]], dim=0)
    enrolled_labels = torch.cat([anchor_y, reference_y, accepted_labels], dim=0)
    prototypes, variances = class_prototypes(enrolled_embeddings, enrolled_labels)
    if config.covariance:
        global_variance = enrolled_embeddings.var(0, unbiased=False).clamp_min(1e-4)
        for class_id in range(8):
            values = enrolled_embeddings[enrolled_labels == class_id]
            if len(values) > 1:
                local = values.var(0, unbiased=False).clamp_min(1e-4)
                variances[class_id] = (1 - config.covariance_shrinkage) * local + config.covariance_shrinkage * global_variance
    metadata = {
        "duration_seconds": time.perf_counter() - started,
        "node_count": len(embeddings),
        "seen_anchor_count": len(anchor),
        "reference_count": len(reference),
        "adaptation_count": len(adaptation),
        "accepted_count": int(accepted.sum()),
        "accepted_by_class": {str(value): int((accepted_labels == value).sum()) for value in holdout},
        "guard_rates": {
            "confidence": float((confidence >= config.confidence_threshold).float().mean()) if len(confidence) else 0.0,
            "agreement": float((agreement >= config.agreement_threshold).float().mean()) if len(agreement) else 0.0,
            "semantic": float((semantic_confidence >= config.semantic_threshold).float().mean()) if len(semantic_confidence) else 0.0,
            "augmentation": float(((augmentation_confidence >= config.augmentation_threshold) & augmentation_agrees).float().mean()) if len(augmentation_confidence) else 0.0,
            "seen_rejection": float(seen_rejected.float().mean()) if len(seen_rejected) else 0.0,
        },
    }
    return GraphEnrollmentResult(prototypes, variances, accepted_indices.cpu(), accepted_labels.cpu(), propagation.cpu(), metadata)


def prototype_predict(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    variances: torch.Tensor | None = None,
    *,
    covariance: bool = False,
    abstention_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = query_embeddings.to(prototypes.device).float()
    if covariance and variances is not None:
        distance = ((query[:, None, :] - prototypes[None, :, :]).square() / variances[None, :, :].clamp_min(1e-4)).mean(-1)
        score = -distance
    else:
        score = F.normalize(query, dim=-1) @ F.normalize(prototypes.float(), dim=-1).T
    confidence, predicted = score.max(1)
    if abstention_threshold is not None:
        predicted = predicted.masked_fill(confidence < abstention_threshold, -1)
    return predicted.cpu(), confidence.cpu()
