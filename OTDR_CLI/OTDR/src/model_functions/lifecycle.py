from __future__ import annotations

"""Shared compact feature-assisted backbone for KPSC, MCF, and CFE."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..event_openworld_data import EventOpenWorldFold
from .event_openworld import derivative_channels, gaussian_smooth, robust_linear_detrend
from .tcn import OTDRTCNBackbone


@dataclass(frozen=True)
class LifecycleModelConfig:
    width: int = 64
    embedding_dim: int = 64
    context_width: int = 32
    blocks: int = 3
    kernel_size: int = 3
    dropout: float = 0.1
    fusion: str = "gated"
    mode: str = "late_fusion"
    smooth_sigma: float = 1.0
    canonicalize: bool = False
    pooling: str = "attention"
    n_classes: int = 8


class MorphologyEncoder(nn.Module):
    """Robust, short-trace TCN over shape, derivatives, and a smoothed trace."""

    def __init__(self, config: LifecycleModelConfig) -> None:
        super().__init__()
        self.smooth_sigma = config.smooth_sigma
        self.canonicalize = config.canonicalize
        self.backbone = OTDRTCNBackbone(
            in_ch=4,
            mid_ch=config.width,
            n_blocks=config.blocks,
            k=config.kernel_size,
            dropout=config.dropout,
            pooling=config.pooling,
        )
        self.output_dim = self.backbone.output_dim
        self.proposal_projection = nn.Linear(2, self.output_dim)

    def canonicalize_trace(self, trace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft, label-independent event proposal and shift to fixed support."""
        detrended, _ = robust_linear_detrend(trace.float())
        first, second = derivative_channels(detrended)
        saliency = first.abs() + 0.5 * second.abs()
        distribution = torch.softmax(saliency / 0.25, dim=1)
        axis = torch.arange(30, dtype=trace.dtype, device=trace.device)
        center = (distribution * axis).sum(1)
        width = torch.sqrt(
            (distribution * (axis[None, :] - center[:, None]).square()).sum(1).clamp_min(1e-6)
        )
        source = axis[None, :] + center[:, None] - 14.5
        x_coordinate = source / 29.0 * 2 - 1
        grid = torch.stack((x_coordinate, torch.zeros_like(x_coordinate)), dim=-1)[:, None, :, :]
        shifted = F.grid_sample(
            trace[:, None, None, :], grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).squeeze(1).squeeze(1)
        proposal = torch.stack((center / 29.0, width / 30.0), dim=1)
        return shifted, proposal

    def channels(self, trace: torch.Tensor) -> torch.Tensor:
        if trace.ndim != 2 or trace.shape[1] != 30:
            raise ValueError("Trace must have shape [batch, 30].")
        detrended, _ = robust_linear_detrend(trace.float())
        first, _ = derivative_channels(detrended)
        smoothed = gaussian_smooth(detrended, self.smooth_sigma)
        local_scale = 1.4826 * (detrended - detrended.median(1, keepdim=True).values).abs().median(
            1, keepdim=True
        ).values
        local_scale = local_scale.clamp_min(1e-3)
        normalized = detrended / local_scale
        return torch.stack((trace, normalized, first / local_scale, smoothed / local_scale), dim=1)

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        if self.canonicalize:
            trace, proposal = self.canonicalize_trace(trace)
            return self.backbone.encode(self.channels(trace)) + self.proposal_projection(proposal)
        return self.backbone.encode(self.channels(trace))


class ContextEncoder(nn.Module):
    """Encode SNR/loss/Reflectance and explicit availability indicators."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.output_dim = width

    def forward(self, context: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != 3 or missing.shape != context.shape:
            raise ValueError("Context and missingness must both have shape [batch, 3].")
        return self.network(torch.cat((context, missing), dim=1))


class FeatureAssistedOTDR(nn.Module):
    """Feature-assisted classifier/localizer with a normalized metric embedding."""

    VALID_MODES = {"late_fusion", "morphology_only", "context_only"}
    VALID_FUSIONS = {"gated", "concat"}

    def __init__(self, config: LifecycleModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or LifecycleModelConfig()
        if self.config.mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        if self.config.fusion not in self.VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(self.VALID_FUSIONS)}")
        self.morphology = MorphologyEncoder(self.config)
        self.context = ContextEncoder(self.config.context_width, self.config.dropout)
        width = self.config.width
        self.context_projection = nn.Linear(self.config.context_width, width)
        if self.config.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(width + self.config.context_width, width),
                nn.Sigmoid(),
            )
            fused_dim = width
        else:
            self.gate = None
            fused_dim = width + self.config.context_width
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(fused_dim, self.config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.embedding_dim),
        )
        self.class_head = nn.Linear(fused_dim, self.config.n_classes)
        self.location_head = nn.Linear(fused_dim, 1)
        self.competence_head = nn.Linear(fused_dim, 1)
        self.attribute_head = nn.Linear(fused_dim, 4)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        trace: torch.Tensor,
        context: torch.Tensor,
        context_missing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        morphology = self.morphology(trace)
        context_embedding = self.context(context, context_missing)
        if self.config.mode == "morphology_only":
            context_embedding = torch.zeros_like(context_embedding)
        elif self.config.mode == "context_only":
            morphology = torch.zeros_like(morphology)
        if self.config.fusion == "gated":
            gate = self.gate(torch.cat((morphology, context_embedding), dim=1))
            context_value = self.context_projection(context_embedding)
            if self.config.mode == "context_only":
                fused = context_value
            elif self.config.mode == "morphology_only":
                fused = morphology
            else:
                fused = morphology + gate * context_value
        else:
            gate = torch.full_like(morphology, float("nan"))
            fused = torch.cat((morphology, context_embedding), dim=1)
        fused = self.fusion_norm(fused)
        embedding = F.normalize(self.embedding_head(fused), dim=1)
        return {
            "morphology": morphology,
            "context_embedding": context_embedding,
            "gate": gate,
            "fused": fused,
            "embedding": embedding,
        }

    def forward(
        self,
        trace: torch.Tensor,
        context: torch.Tensor,
        context_missing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.encode(trace, context, context_missing)
        fused = output["fused"]
        output.update(
            logits=self.class_head(fused),
            position=self.location_head(fused).squeeze(1),
            competence=self.competence_head(fused).squeeze(1),
            attributes=self.attribute_head(fused),
        )
        return output


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL covariance alignment for declared synthetic/target adaptation arms."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("CORAL inputs must be [samples, shared_features].")
    if len(source) < 2 or len(target) < 2:
        return source.new_zeros(())
    source_centered = source - source.mean(0, keepdim=True)
    target_centered = target - target.mean(0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (len(source) - 1)
    target_cov = target_centered.T @ target_centered / (len(target) - 1)
    return (source_cov - target_cov).square().mean()


def mmd_rbf(source: torch.Tensor, target: torch.Tensor, bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Biased multi-kernel MMD; stable for the small pilot batches."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("MMD inputs must be [samples, shared_features].")

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(left.float(), right.float()).square()
        return sum(torch.exp(-distance / (2 * bandwidth * bandwidth)) for bandwidth in bandwidths)

    return kernel(source, source).mean() + kernel(target, target).mean() - 2 * kernel(source, target).mean()
