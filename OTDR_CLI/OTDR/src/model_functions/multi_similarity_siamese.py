from __future__ import annotations

"""Multi-similarity Siamese network for OTDR traces."""

import torch
from torch import nn
from torch.nn import functional as F

from .tcn import OTDRTCNBackbone


class OTDRInstanceEncoder(OTDRTCNBackbone):
    """Encode an OTDR trace without collapsing its metric-space magnitude."""

    def __init__(
        self,
        *,
        in_ch: int = 2,
        mid_ch: int = 64,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(in_ch=in_ch, mid_ch=mid_ch, n_blocks=4, k=3, dropout=dropout)
        self.projection = nn.Sequential(
            nn.Linear(mid_ch, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encode(inputs))


def multi_similarity_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return symmetric L1, RMS-L2, cosine, and Hadamard comparison features."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Pair embeddings must have the same two-dimensional shape.")
    difference = left - right
    l1 = difference.abs().mean(dim=-1, keepdim=True)
    l2 = difference.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-8).unsqueeze(-1)
    product = left * right
    return torch.cat((l1, l2, cosine, product), dim=-1)


def comparison_features(left: torch.Tensor, right: torch.Tensor, mode: str) -> torch.Tensor:
    """Select one comparison ablation or the complete feature set."""

    complete = multi_similarity_features(left, right)
    embedding_dim = left.shape[-1]
    slices = {
        "l1": complete[:, 0:1],
        "l2": complete[:, 1:2],
        "cosine": complete[:, 2:3],
        "product": complete[:, 3 : 3 + embedding_dim],
        "multi": complete,
    }
    if mode not in slices:
        raise ValueError(f"Unknown comparison mode: {mode}")
    return slices[mode]


class MultiSimilaritySiamese(nn.Module):
    """Shared OTDR encoder and learned symmetric pair-similarity head."""

    def __init__(
        self,
        *,
        in_ch: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        similarity_mode: str = "multi",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.similarity_mode = similarity_mode
        self.encoder = OTDRInstanceEncoder(
            in_ch=in_ch,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        feature_dim = {"l1": 1, "l2": 1, "cosine": 1, "product": embedding_dim, "multi": embedding_dim + 3}.get(similarity_mode)
        if feature_dim is None:
            raise ValueError(f"Unknown similarity_mode: {similarity_mode}")
        self.similarity_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        for module in self.similarity_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def score_embeddings(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.similarity_head(comparison_features(left, right, self.similarity_mode)).squeeze(-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.score_embeddings(self.encode(left), self.encode(right))
