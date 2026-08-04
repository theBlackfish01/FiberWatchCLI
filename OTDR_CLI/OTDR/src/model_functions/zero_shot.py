from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .tcn import OTDRTCNBackbone


def require_cuda(requested: str = "cuda:0") -> torch.device:
    """Return a selected CUDA device or fail instead of silently falling back."""
    if not requested.lower().startswith("cuda"):
        raise ValueError("Zero-shot OTDR requires a CUDA device such as cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("Zero-shot OTDR requires CUDA, but torch.cuda.is_available() is false.")
    try:
        device = torch.device(requested)
        index = 0 if device.index is None else device.index
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid CUDA device: {requested}") from exc
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable.")
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


class OTDRZeroShotEncoder(OTDRTCNBackbone):
    def __init__(
        self,
        *,
        in_ch: int = 2,
        mid_ch: int = 64,
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(in_ch=in_ch, mid_ch=mid_ch, n_blocks=4, k=3, dropout=dropout)
        self.projection = nn.Sequential(
            nn.Linear(mid_ch, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.encode(x)), dim=-1)


def aggregate_prototype_scores(
    sample_embeddings: torch.Tensor,
    prototype_embeddings: torch.Tensor,
    *,
    temperature: torch.Tensor | float,
) -> torch.Tensor:
    """Aggregate K description similarities into one score per class."""
    temp = torch.as_tensor(temperature, device=sample_embeddings.device, dtype=sample_embeddings.dtype)
    temp = temp.clamp(1e-3, 1.0)
    similarities = torch.einsum("bd,ckd->bck", sample_embeddings, prototype_embeddings) / temp
    return torch.logsumexp(similarities, dim=-1) - math.log(prototype_embeddings.shape[1])


class ZeroShotClassifier(nn.Module):
    def __init__(self, *, in_ch: int = 2, embedding_dim: int = 768) -> None:
        super().__init__()
        self.encoder = OTDRZeroShotEncoder(in_ch=in_ch, embedding_dim=embedding_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-3, 1.0)

    def forward(self, inputs: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(inputs)
        scores = aggregate_prototype_scores(embeddings, prototypes, temperature=self.temperature)
        return scores, embeddings
