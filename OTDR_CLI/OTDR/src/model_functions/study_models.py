from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .tcn import OTDRTCNBackbone


def _init(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class TraceEncoder(nn.Module):
    def __init__(self, *, embedding_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = OTDRTCNBackbone(in_ch=2, mid_ch=64, n_blocks=4, k=3, dropout=dropout)
        self.projection = nn.Sequential(
            nn.Linear(64, max(128, embedding_dim)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max(128, embedding_dim), embedding_dim),
        )
        self.embedding_dim = embedding_dim
        _init(self.projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone.encode(x)), dim=-1)


class EpisodicMetricModel(nn.Module):
    """Supervised metric encoder with class proxies used only during fitting."""

    def __init__(self, *, class_count: int, embedding_dim: int = 128, dropout: float = 0.1, temperature: float = 0.1) -> None:
        super().__init__()
        self.encoder = TraceEncoder(embedding_dim=embedding_dim, dropout=dropout)
        self.proxies = nn.Parameter(torch.randn(class_count, embedding_dim))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.02, 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x)
        logits = embeddings @ F.normalize(self.proxies, dim=-1).T / self.temperature
        return logits, embeddings


class PhysicsSemanticModel(nn.Module):
    """Map a trace into a reviewable OTDR physics-attribute space."""

    def __init__(self, *, attribute_dim: int, latent_dim: int = 128, dropout: float = 0.1, temperature: float = 0.08) -> None:
        super().__init__()
        self.encoder = TraceEncoder(embedding_dim=latent_dim, dropout=dropout)
        self.attribute_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim, attribute_dim))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        _init(self.attribute_head)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.02, 1.0)

    def forward(self, x: torch.Tensor, prototypes: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        attributes = torch.sigmoid(self.attribute_head(latent))
        logits = None
        if prototypes is not None:
            logits = F.normalize(attributes, dim=-1) @ F.normalize(prototypes, dim=-1).T / self.temperature
        return logits, attributes, latent


class SelfSupervisedTraceModel(nn.Module):
    """TCN encoder with a masked-trace reconstruction head."""

    def __init__(self, *, embedding_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = TraceEncoder(embedding_dim=embedding_dim, dropout=dropout)
        hidden = max(128, embedding_dim)
        self.reconstruction = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, 31))
        _init(self.reconstruction)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return embedding, self.reconstruction(embedding)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    if len(embeddings) < 2:
        return embeddings.sum() * 0.0
    similarities = embeddings @ embeddings.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    logits = similarities.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positives.sum(1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return -(log_prob.masked_fill(~positives, 0.0).sum(1)[valid] / counts[valid]).mean()


def nt_xent_loss(left: torch.Tensor, right: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    batch = len(left)
    embeddings = torch.cat([left, right], dim=0)
    logits = embeddings @ embeddings.T / temperature
    diagonal = torch.eye(2 * batch, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    targets = torch.arange(2 * batch, device=logits.device)
    targets = (targets + batch) % (2 * batch)
    return F.cross_entropy(logits, targets)
