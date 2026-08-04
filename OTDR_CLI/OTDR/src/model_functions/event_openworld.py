from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def load_event_recipes(path: str | Path, *, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("recipe_id") != "otdr-event-grammar-v1":
        raise ValueError("Unexpected or unfrozen event recipe grammar.")
    classes = payload.get("classes", [])
    if [item.get("id") for item in classes] != list(range(8)):
        raise ValueError("Event recipes must contain ordered class IDs 0..7.")
    means = torch.tensor([item["mean"] for item in classes], dtype=torch.float32, device=device)
    stds = torch.tensor([item["std"] for item in classes], dtype=torch.float32, device=device)
    if means.shape != (8, len(payload["factor_names"])) or stds.shape != means.shape:
        raise ValueError("Event recipe factor dimensions are inconsistent.")
    if not torch.all((means >= 0) & (means <= 1)) or not torch.all(stds > 0):
        raise ValueError("Recipe means must be in [0,1] and standard deviations positive.")
    return {"payload": payload, "means": means, "stds": stds}


def _fixed_gaussian_kernel(sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    return (kernel / kernel.sum()).view(1, 1, -1)


def gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _fixed_gaussian_kernel(sigma, dtype=x.dtype, device=x.device)
    radius = (kernel.shape[-1] - 1) // 2
    padded = F.pad(x[:, None, :], (radius, radius), mode="reflect")
    return F.conv1d(padded, kernel).squeeze(1)


def robust_linear_detrend(trace: torch.Tensor, iterations: int = 3, delta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Huber-IRLS background fit for short traces."""
    if trace.ndim != 2:
        raise ValueError("trace must have shape [batch, length]")
    batch, length = trace.shape
    t = torch.linspace(-1.0, 1.0, length, dtype=trace.dtype, device=trace.device).expand(batch, -1)
    weights = torch.ones_like(trace)
    slope = torch.zeros(batch, dtype=trace.dtype, device=trace.device)
    intercept = trace.mean(1)
    eps = torch.finfo(trace.dtype).eps * 32
    for _ in range(iterations):
        sw = weights.sum(1).clamp_min(eps)
        mt = (weights * t).sum(1) / sw
        my = (weights * trace).sum(1) / sw
        centered_t = t - mt[:, None]
        slope = (weights * centered_t * (trace - my[:, None])).sum(1) / (
            (weights * centered_t.square()).sum(1).clamp_min(eps)
        )
        intercept = my - slope * mt
        residual = trace - (intercept[:, None] + slope[:, None] * t)
        scale = residual.abs().median(1).values.clamp_min(1e-3)
        ratio = residual.abs() / (delta * scale[:, None])
        weights = torch.where(ratio <= 1, torch.ones_like(ratio), ratio.reciprocal())
    baseline = intercept[:, None] + slope[:, None] * t
    return trace - baseline, baseline


def derivative_channels(detrended: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = F.pad(detrended[:, 1:] - detrended[:, :-1], (1, 0))
    second = F.pad(first[:, 1:] - first[:, :-1], (1, 0))
    return first, second


class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels: int, width: int, kernels: tuple[int, ...] = (3, 5, 7)) -> None:
        super().__init__()
        each = max(8, width // len(kernels))
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, each, kernel, padding=kernel // 2) for kernel in kernels
        ])
        self.projection = nn.Conv1d(each * len(kernels), width, 1)
        self.norm = nn.GroupNorm(max(1, min(8, width // 8)), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.cat([F.gelu(conv(x)) for conv in self.convs], dim=1)
        value = F.gelu(self.norm(self.projection(value)))
        return torch.cat([value.mean(-1), value.amax(-1)], dim=1)


class DilatedTCNBranch(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        layers = []
        current = in_channels
        for dilation in (1, 2, 4):
            layers.extend([
                nn.Conv1d(current, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(max(1, min(8, width // 8)), width), nn.GELU(),
            ])
            current = width
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.network(x)
        return torch.cat([value.mean(-1), value.amax(-1)], dim=1)


class EventCanonicalizer(nn.Module):
    def __init__(self, *, channel_count: int = 5, patch_size: int = 15, soft_alignment: bool = True) -> None:
        super().__init__()
        if patch_size < 5 or patch_size % 2 == 0:
            raise ValueError("patch_size must be odd and at least five")
        self.patch_size = patch_size
        self.soft_alignment = soft_alignment
        self.saliency = nn.Sequential(
            nn.Conv1d(channel_count, 24, 3, padding=1), nn.GELU(),
            nn.Conv1d(24, 1, 3, padding=1),
        )

    def forward(self, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, length = channels.shape
        logits = self.saliency(channels).squeeze(1)
        distribution = logits.softmax(-1)
        positions = torch.linspace(0, length - 1, length, dtype=channels.dtype, device=channels.device)
        if self.soft_alignment:
            center = (distribution * positions).sum(-1)
        else:
            center = distribution.argmax(-1).to(channels.dtype).detach()
        half = (self.patch_size - 1) / 2
        offsets = torch.linspace(-half, half, self.patch_size, dtype=channels.dtype, device=channels.device)
        sample_index = center[:, None] + offsets[None, :]
        x_coordinate = sample_index / max(length - 1, 1) * 2 - 1
        grid = torch.stack([x_coordinate, torch.zeros_like(x_coordinate)], dim=-1)[:, None, :, :]
        patch = F.grid_sample(
            channels[:, :, None, :], grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(2)
        return patch, distribution, center


class EventCompositionalModel(nn.Module):
    """Event-canonicalized probabilistic factor model for EC-CZSL and PC2-OE."""

    def __init__(
        self,
        *,
        factor_count: int = 12,
        width: int = 72,
        latent_dim: int = 64,
        patch_size: int = 15,
        dropout: float = 0.1,
        soft_alignment: bool = True,
        canonicalize: bool = True,
        derivative_channels_enabled: bool = True,
        global_branch_enabled: bool = True,
        residual_enabled: bool = True,
        deterministic_factors: bool = False,
        recipe_mode: str = "probabilistic",
        backbone: str = "multiscale",
    ) -> None:
        super().__init__()
        if backbone not in {"multiscale", "shapelet", "tcn"}:
            raise ValueError("backbone must be multiscale, shapelet, or tcn")
        if recipe_mode not in {"probabilistic", "point"}:
            raise ValueError("recipe_mode must be probabilistic or point")
        self.factor_count = factor_count
        self.canonicalize = canonicalize
        self.derivative_channels_enabled = derivative_channels_enabled
        self.global_branch_enabled = global_branch_enabled
        self.residual_enabled = residual_enabled
        self.deterministic_factors = deterministic_factors
        self.recipe_mode = recipe_mode
        self.backbone = backbone
        channel_count = 5 if derivative_channels_enabled else 3
        self.channel_count = channel_count
        # Conditioning sees both measurement SNR and the pre-normalization trace
        # amplitude.  The latter is required because independent channel RMS
        # normalization would otherwise erase the physical energy factor.
        self.snr_gate = nn.Sequential(nn.Linear(2, 24), nn.GELU(), nn.Linear(24, channel_count), nn.Sigmoid())
        self.canonicalizer = EventCanonicalizer(channel_count=channel_count, patch_size=patch_size, soft_alignment=soft_alignment)
        kernels = (3, 5, 7) if backbone == "multiscale" else (3, 9, 15)
        branch = (lambda: DilatedTCNBranch(channel_count, width)) if backbone == "tcn" else (
            lambda: MultiScaleBranch(channel_count, width, kernels=kernels)
        )
        self.local_branch = branch()
        self.global_branch = branch()
        fused_dim = width * 2 * (2 if global_branch_enabled else 1)
        self.fusion = nn.Sequential(nn.Linear(fused_dim, latent_dim), nn.GELU(), nn.Dropout(dropout))
        self.factor_mean = nn.Linear(latent_dim, factor_count)
        self.factor_log_std = nn.Linear(latent_dim, factor_count)
        self.trace_residual = nn.Linear(latent_dim, latent_dim)
        self.recipe_residual = nn.Sequential(nn.Linear(factor_count, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.15)))
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def _channels(self, features: torch.Tensor) -> torch.Tensor:
        snr = features[:, :1]
        trace = features[:, 1:]
        detrended, _ = robust_linear_detrend(trace)
        first, second = derivative_channels(detrended)
        smooth_one = gaussian_smooth(detrended, 0.8)
        smooth_two = gaussian_smooth(detrended, 1.6)
        values = [detrended, smooth_one, smooth_two]
        if self.derivative_channels_enabled:
            values = [detrended, first, second, smooth_one, smooth_two]
        channels = torch.stack(values, dim=1)
        trace_rms = detrended.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
        conditioning = torch.cat([snr, torch.log1p(trace_rms)], dim=1)
        gate = 0.5 + self.snr_gate(conditioning).unsqueeze(-1)
        # This monotone, bounded multiplier makes absolute event energy
        # explicitly identifiable while retaining the numerical stability of
        # normalized derivative/smoothing channels.
        amplitude = (0.5 + trace_rms / (1.0 + trace_rms)).unsqueeze(1)
        scale = channels.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
        return channels / scale * gate * amplitude

    def forward(self, features: torch.Tensor, recipe_means: torch.Tensor, recipe_stds: torch.Tensor) -> dict[str, torch.Tensor]:
        channels = self._channels(features)
        if self.canonicalize:
            patch, location, center = self.canonicalizer(channels)
        else:
            patch = F.interpolate(channels, size=self.canonicalizer.patch_size, mode="linear", align_corners=True)
            location = channels.square().mean(1).softmax(-1)
            positions = torch.arange(channels.shape[-1], dtype=channels.dtype, device=channels.device)
            center = (location * positions).sum(-1)
        local = self.local_branch(patch)
        pieces = [local]
        if self.global_branch_enabled:
            pieces.append(self.global_branch(channels))
        latent = self.fusion(torch.cat(pieces, dim=1))
        factor_mean = self.factor_mean(latent).sigmoid()
        if self.deterministic_factors:
            factor_std = torch.full_like(factor_mean, 0.05)
        else:
            factor_std = (F.softplus(self.factor_log_std(latent)) + 0.03).clamp_max(0.75)
        delta = factor_mean[:, None, :] - recipe_means[None, :, :]
        if self.recipe_mode == "probabilistic":
            variance = factor_std[:, None, :].square() + recipe_stds[None, :, :].square()
            compatibility = -0.5 * (delta.square() / variance + variance.log()).mean(-1)
        else:
            compatibility = -delta.square().mean(-1)
        bounded_residual = torch.tanh(self.trace_residual(latent)) / math.sqrt(latent.shape[-1])
        recipe_residual = F.normalize(self.recipe_residual(recipe_means), dim=-1)
        residual_logits = bounded_residual @ recipe_residual.T
        residual_weight = self.residual_scale.sigmoid() * (0.25 if self.residual_enabled else 0.0)
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        logits = (compatibility + residual_weight * residual_logits) / temperature
        reconstruction_residual_per_class = (
            delta.square() / recipe_stds[None, :, :].square()
        ).mean(-1)
        reconstruction_residual = reconstruction_residual_per_class.amin(-1)
        return {
            "logits": logits,
            "compatibility": compatibility,
            "factor_mean": factor_mean,
            "factor_std": factor_std,
            "embedding": F.normalize(latent, dim=-1),
            "location": location,
            "center": center,
            "channels": channels,
            "patch": patch,
            "reconstruction_residual": reconstruction_residual,
            "reconstruction_residual_per_class": reconstruction_residual_per_class,
            "residual_norm": bounded_residual.norm(dim=-1),
        }


class PhysicsEventRenderer:
    """Frozen analytical 30-point event composer operating in standardized feature space."""

    def __init__(
        self,
        recipe_means: torch.Tensor,
        recipe_stds: torch.Tensor,
        *,
        snr_mean: float,
        snr_scale: float,
        trace_rms_target: float = 0.75,
    ) -> None:
        self.recipe_means = recipe_means
        self.recipe_stds = recipe_stds
        self.device = recipe_means.device
        self.snr_mean = float(snr_mean)
        self.snr_scale = max(float(snr_scale), 1e-6)
        self.trace_rms_target = max(float(trace_rms_target), 1e-3)

    def _rand(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        return torch.rand(shape, generator=generator, device=self.device)

    def render_named(self, labels: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        labels = labels.to(self.device, dtype=torch.long)
        mean, std = self.recipe_means[labels], self.recipe_stds[labels]
        factors = (mean + torch.randn(mean.shape, generator=generator, device=self.device) * std).clamp(0, 1)
        return self._render(factors, generator=generator), factors

    def _different_fault_pairs(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.randint(1, 8, (batch_size,), generator=generator, device=self.device)
        offset = torch.randint(1, 7, (batch_size,), generator=generator, device=self.device)
        right = (left - 1 + offset) % 7 + 1
        return left, right

    def render_boundary(self, batch_size: int, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = self._different_fault_pairs(batch_size, generator)
        mix = 0.35 + 0.30 * self._rand((batch_size, 1), generator)
        factors = (mix * self.recipe_means[left] + (1 - mix) * self.recipe_means[right]).clone()
        mode = torch.arange(batch_size, device=self.device) % 5
        factors[mode == 0, 3] = 0.65
        factors[mode == 0, 4] = 0.75
        factors[mode == 1, 2] = 0.48
        factors[mode == 2, 6] = 0.55
        factors[mode == 2, 0] = 0.55
        factors[mode == 3, 10] = 0.95
        factors[mode == 4, 8] = 1.0
        return self._render(factors.clamp(0, 1), generator=generator, boundary=True), factors

    def _render(self, f: torch.Tensor, *, generator: torch.Generator, boundary: bool = False) -> torch.Tensor:
        batch = len(f)
        t = torch.linspace(0, 1, 30, device=self.device)[None, :]
        position = 0.12 + 0.76 * self._rand((batch, 1), generator)
        width = 0.025 + 0.255 * (0.15 + 0.85 * f[:, 2:3])
        magnitude = 0.08 + 1.32 * (0.35 + 0.65 * f[:, 8:9])
        slope = -0.10 - 0.70 * self._rand((batch, 1), generator)
        offset = torch.randn((batch, 1), generator=generator, device=self.device) * 0.12
        baseline = offset + slope * (t - 0.5) * (0.45 + 1.10 * f[:, 9:10])
        z = (t - position) / width.clamp_min(0.01)
        skew = 0.75 * (1.0 - f[:, 7:8])
        reflection_width = torch.where(t < position, 1.0 - skew, 1.0 + skew).clamp_min(0.20)
        reflection = f[:, 0:1] * magnitude * torch.exp(-0.5 * (z / (0.40 * reflection_width)).square())
        abrupt = -f[:, 1:2] * magnitude * torch.sigmoid(z * (9.0 - 6.0 * f[:, 2:3]))
        broad = -0.45 * f[:, 1:2] * f[:, 2:3] * magnitude * torch.sigmoid(z * 2.2)
        terminal = -1.6 * f[:, 3:4] * magnitude * torch.sigmoid(z * 12.0)
        continuation = 0.35 + 0.65 * f[:, 4:5]
        loss = (abrupt + broad) * continuation + terminal
        dead_zone = -0.65 * f[:, 11:12] * magnitude * torch.exp(-0.5 * ((t - position - width) / (width * 1.5)).square())
        irregular = 0.22 * f[:, 6:7] * magnitude * torch.sin((t - position) * 55.0) * torch.exp(-0.5 * (z / 1.3).square())
        secondary_position = (position + 0.04 + 0.14 * self._rand((batch, 1), generator)).clamp_max(0.95)
        secondary = 0.55 * f[:, 10:11] * magnitude * torch.exp(-0.5 * ((t - secondary_position) / (width * 0.55)).square())
        post_distance = (t - position).clamp_min(0.0)
        slope_contrast = -(f[:, 5:6] - 0.5) * magnitude * post_distance * torch.sigmoid(z * 8.0)
        trace = baseline + slope_contrast + reflection + loss + dead_zone + irregular + secondary
        if boundary:
            trace = trace + 0.12 * torch.sin(17 * t + self._rand((batch, 1), generator) * math.pi)
        snr_raw = 4.0 + 36.0 * self._rand((batch, 1), generator)
        noise_std = 0.015 + 0.20 * torch.exp(-(snr_raw - 4.0) / 9.0)
        trace = trace + torch.randn(trace.shape, generator=generator, device=self.device) * noise_std
        # Match the nuisance amplitude scale estimated strictly from outer-seen
        # standardized training traces.  Recipe factor 8 retains controlled
        # energy variation instead of being erased by a fixed global rescale.
        centered = trace - trace.mean(1, keepdim=True)
        current_rms = centered.square().mean(1, keepdim=True).sqrt().clamp_min(1e-4)
        target_rms = self.trace_rms_target * (0.4 + 1.2 * f[:, 9:10])
        trace = trace.mean(1, keepdim=True) + centered / current_rms * target_rms
        snr_standardized = (snr_raw - self.snr_mean) / self.snr_scale
        return torch.cat([snr_standardized, trace], dim=1).float()


def novelty_components(
    output: dict[str, torch.Tensor],
    known_class_ids: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    all_logits = output["logits"]
    known = list(range(all_logits.shape[1])) if known_class_ids is None else [int(value) for value in known_class_ids]
    if not known:
        raise ValueError("known_class_ids cannot be empty")
    logits = all_logits[:, known]
    energy = -torch.logsumexp(logits, dim=1)
    per_class = output.get("reconstruction_residual_per_class")
    distance = per_class[:, known].amin(-1) if per_class is not None else output["reconstruction_residual"]
    saliency_entropy = -(output["location"] * output["location"].clamp_min(1e-8).log()).sum(1)
    if logits.shape[1] >= 2:
        top = logits.topk(2, dim=1).values
        max_gap = -(top[:, 0] - top[:, 1])
    else:
        max_gap = torch.zeros_like(energy)
    return torch.stack([energy, distance, saliency_entropy, max_gap], dim=1)
