from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_functions.event_openworld import (
    EventCompositionalModel,
    PhysicsEventRenderer,
    novelty_components,
    robust_linear_detrend,
)
from .model_functions.zero_shot import require_cuda
from .zero_shot_training import seed_everything


@dataclass(frozen=True)
class ECConfig:
    epochs: int = 8
    steps_per_epoch: int = 48
    batch_size: int = 384
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    width: int = 72
    latent_dim: int = 64
    patch_size: int = 15
    dropout: float = 0.10
    factor_weight: float = 0.65
    uncertainty_weight: float = 0.05
    residual_penalty: float = 0.02
    class_dropout_count: int = 1
    canonicalize: bool = True
    soft_alignment: bool = True
    derivative_channels: bool = True
    global_branch: bool = True
    residual: bool = True
    deterministic_factors: bool = False
    recipe_mode: str = "probabilistic"
    backbone: str = "multiscale"
    fusion_weights: tuple[float, float, float, float] = (0.10, 0.70, 0.10, 0.10)
    calibration: str = "normalized"
    seed: int = 42


@dataclass(frozen=True)
class PC2Config:
    epochs: int = 9
    steps_per_epoch: int = 56
    batch_size: int = 384
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    width: int = 80
    latent_dim: int = 72
    patch_size: int = 15
    dropout: float = 0.10
    factor_weight: float = 0.55
    named_weight: float = 0.45
    oe_weight: float = 0.45
    cvar_weight: float = 0.35
    energy_margin: float = 1.0
    cvar_fraction: float = 0.20
    synthetic_fraction: float = 0.35
    outlier_mode: str = "physics"
    atom_ablation: str = "none"
    calibration: str = "normalized"
    fusion_weights: tuple[float, float, float, float] = (0.45, 0.40, 0.10, 0.05)
    seed: int = 42


@dataclass(frozen=True)
class SGMEConfig:
    k_neighbors: int = 12
    graph_iterations: int = 12
    graph_temperature: float = 0.12
    propagation_alpha: float = 0.85
    confidence_threshold: float = 0.82
    agreement_threshold: float = 0.70
    augmentation_threshold: float = 0.78
    semantic_threshold: float = 0.55
    seen_rejection_threshold: float = 0.05
    covariance: bool = True
    covariance_shrinkage: float = 0.25
    abstention_quantile: float = 0.05
    seed: int = 42


def cvar_ranking_loss(normal_score: torch.Tensor, outlier_score: torch.Tensor, *, margin: float, fraction: float) -> torch.Tensor:
    if normal_score.numel() == 0 or outlier_score.numel() == 0:
        return normal_score.sum() * 0.0 + outlier_score.sum() * 0.0
    count = min(len(normal_score), len(outlier_score))
    losses = F.relu(margin + normal_score[:count] - outlier_score[:count])
    tail = max(1, int(math_ceil(len(losses) * fraction)))
    return losses.topk(tail).values.mean()


def math_ceil(value: float) -> int:
    return int(np.ceil(value))


def _balanced_indices(labels: torch.Tensor, batch_size: int, rng: np.random.Generator, allowed: list[int]) -> torch.Tensor:
    labels_np = labels.numpy()
    per_class = max(2, batch_size // len(allowed))
    pieces: list[np.ndarray] = []
    for class_id in allowed:
        candidates = np.flatnonzero(labels_np == class_id)
        pieces.append(rng.choice(candidates, size=per_class, replace=len(candidates) < per_class))
    indices = np.concatenate(pieces)
    if len(indices) < batch_size:
        pool = np.concatenate([np.flatnonzero(labels_np == class_id) for class_id in allowed])
        indices = np.concatenate([indices, rng.choice(pool, size=batch_size - len(indices), replace=True)])
    rng.shuffle(indices)
    return torch.from_numpy(indices[:batch_size].astype(np.int64))


def _amp() -> tuple[torch.dtype, torch.amp.GradScaler]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype, torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)


def _model_from_config(config: ECConfig | PC2Config) -> EventCompositionalModel:
    return EventCompositionalModel(
        width=config.width,
        latent_dim=config.latent_dim,
        patch_size=config.patch_size,
        dropout=config.dropout,
        canonicalize=getattr(config, "canonicalize", True),
        soft_alignment=getattr(config, "soft_alignment", True),
        derivative_channels_enabled=getattr(config, "derivative_channels", True),
        global_branch_enabled=getattr(config, "global_branch", True),
        residual_enabled=getattr(config, "residual", True),
        deterministic_factors=getattr(config, "deterministic_factors", False),
        recipe_mode=getattr(config, "recipe_mode", "probabilistic"),
        backbone=getattr(config, "backbone", "multiscale"),
    )


def _metadata(model: nn.Module, config: Any, started: float, history: list[dict[str, float]], device: torch.device) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "duration_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "history": history,
    }


def train_ec_czsl(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    *,
    device: torch.device,
    config: ECConfig,
) -> tuple[EventCompositionalModel, dict[str, Any]]:
    device = require_cuda(str(device))
    seed_everything(config.seed)
    model = _model_from_config(config).to(device)
    means, stds = recipe_means.to(device), recipe_stds.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype, scaler = _amp()
    rng = np.random.default_rng(config.seed)
    seen_ids = sorted(int(value) for value in train_y.unique())
    seen_faults = [value for value in seen_ids if value != 0]
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config.epochs):
        model.train()
        totals = np.zeros(4, dtype=float)
        for step in range(config.steps_per_epoch):
            count = min(config.class_dropout_count, max(0, len(seen_faults) - 2))
            dropped = set(rng.choice(seen_faults, size=count, replace=False).tolist()) if count else set()
            allowed = [value for value in seen_ids if value not in dropped]
            indices = _balanced_indices(train_y, config.batch_size, rng, allowed)
            x = train_x[indices].to(device, non_blocking=True)
            labels = train_y[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(x, means, stds)
                active_logits = output["logits"][:, allowed]
                target_map = torch.full((8,), -1, dtype=torch.long, device=device)
                target_map[torch.tensor(allowed, device=device)] = torch.arange(len(allowed), device=device)
                ce = F.cross_entropy(active_logits, target_map[labels])
                target_mean = means[labels]
                target_var = output["factor_std"].square() + stds[labels].square()
                factor = 0.5 * (((output["factor_mean"] - target_mean).square() / target_var) + target_var.log()).mean()
                uncertainty = output["factor_std"].mean()
                residual = output["residual_norm"].square().mean()
                loss = ce + config.factor_weight * factor + config.uncertainty_weight * uncertainty + config.residual_penalty * residual
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            totals += [float(loss.detach()), float(ce.detach()), float(factor.detach()), float(uncertainty.detach())]
        history.append({"epoch": epoch + 1, "loss": totals[0] / config.steps_per_epoch,
                        "ce": totals[1] / config.steps_per_epoch, "factor_nll": totals[2] / config.steps_per_epoch,
                        "factor_std": totals[3] / config.steps_per_epoch})
    metadata = _metadata(model, config, started, history, device)
    metadata["seen_class_ids"] = seen_ids
    metadata["episode_type"] = f"leave_{config.class_dropout_count}_seen_fault_out"
    return model, metadata


def train_pc2_oe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    *,
    snr_mean: float,
    snr_scale: float,
    device: torch.device,
    config: PC2Config,
) -> tuple[EventCompositionalModel, dict[str, Any]]:
    device = require_cuda(str(device))
    seed_everything(config.seed)
    model = _model_from_config(config).to(device)
    means, stds = recipe_means.to(device), recipe_stds.to(device)
    with torch.no_grad():
        train_detrended, _ = robust_linear_detrend(train_x[:, 1:].float())
        trace_rms_target = float(
            train_detrended.square().mean(1).sqrt().median().clamp_min(1e-3)
        )
    renderer = PhysicsEventRenderer(
        means,
        stds,
        snr_mean=snr_mean,
        snr_scale=snr_scale,
        trace_rms_target=trace_rms_target,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype, scaler = _amp()
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed + 900_001)
    seen_ids = sorted(int(value) for value in train_y.unique())
    history: list[dict[str, float]] = []
    valid_outlier_modes = {"physics", "generic", "virtual_feature", "none"}
    if config.batch_size < 16:
        raise ValueError("PC2 batch_size must be at least 16 to contain real and synthetic samples.")
    if config.outlier_mode not in valid_outlier_modes:
        raise ValueError(f"outlier_mode must be one of {sorted(valid_outlier_modes)}")
    valid_atoms = {"none", "narrow_reflection", "abrupt_step", "smooth_or_broad_attenuation",
                   "terminal_drop", "reflection_dead_zone", "irregular_mixture", "multi_event"}
    if config.atom_ablation not in valid_atoms:
        raise ValueError(f"atom_ablation must be one of {sorted(valid_atoms)}")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config.epochs):
        model.train()
        totals = np.zeros(6, dtype=float)
        for _ in range(config.steps_per_epoch):
            synthetic_count = max(8, min(config.batch_size - 8, int(config.batch_size * config.synthetic_fraction)))
            real_count = config.batch_size - synthetic_count
            indices = _balanced_indices(train_y, real_count, rng, seen_ids)
            real_x = train_x[indices].to(device, non_blocking=True)
            real_y = train_y[indices].to(device, non_blocking=True)
            synthetic_y = torch.arange(synthetic_count, device=device) % 8
            synthetic_y = synthetic_y[torch.randperm(synthetic_count, generator=generator, device=device)]
            named_x, named_factors = renderer.render_named(synthetic_y, generator=generator)
            boundary_x, boundary_factors = renderer.render_boundary(synthetic_count, generator=generator)
            factor_index = {"narrow_reflection": 0, "abrupt_step": 1, "smooth_or_broad_attenuation": 2,
                            "terminal_drop": 3, "irregular_mixture": 6, "multi_event": 10,
                            "reflection_dead_zone": 11}.get(config.atom_ablation)
            if factor_index is not None:
                named_factors[:, factor_index] = 0
                named_x = renderer._render(named_factors, generator=generator)
                boundary_factors[:, factor_index] = 0
                boundary_x = renderer._render(boundary_factors, generator=generator, boundary=True)
            if config.outlier_mode == "generic":
                source = real_x[torch.randint(len(real_x), (synthetic_count,), generator=generator, device=device)].clone()
                source[:, 1:] = torch.flip(source[:, 1:], dims=[1])
                masks = torch.rand((synthetic_count, 30), generator=generator, device=device) < 0.2
                source[:, 1:].masked_fill_(masks, 0.0)
                boundary_x = source + torch.randn(source.shape, generator=generator, device=device) * 0.15
            elif config.outlier_mode == "virtual_feature":
                left = real_x[torch.randint(len(real_x), (synthetic_count,), generator=generator, device=device)]
                right = real_x[torch.randint(len(real_x), (synthetic_count,), generator=generator, device=device)]
                direction = F.normalize(left - right, dim=-1)
                boundary_x = (left + right) / 2 + direction * (1.0 + 0.5 * torch.rand(
                    (synthetic_count, 1), generator=generator, device=device
                ))
            normal_candidates = real_x[real_y == 0]
            if len(normal_candidates) < synthetic_count:
                extra = normal_candidates[torch.randint(len(normal_candidates), (synthetic_count - len(normal_candidates),), generator=generator, device=device)]
                normal_candidates = torch.cat([normal_candidates, extra], dim=0)
            normal_x = normal_candidates[:synthetic_count]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                real_output = model(real_x, means, stds)
                named_output = model(named_x, means, stds)
                boundary_output = model(boundary_x, means, stds)
                normal_output = model(normal_x, means, stds)
                target_map = torch.full((8,), -1, dtype=torch.long, device=device)
                target_map[torch.tensor(seen_ids, device=device)] = torch.arange(len(seen_ids), device=device)
                real_ce = F.cross_entropy(real_output["logits"][:, seen_ids], target_map[real_y])
                named_ce = F.cross_entropy(named_output["logits"], synthetic_y)
                factor = F.smooth_l1_loss(real_output["factor_mean"], means[real_y]) + F.smooth_l1_loss(
                    named_output["factor_mean"], named_factors
                )
                boundary_energy = novelty_components(boundary_output, seen_ids)[:, 0]
                normal_energy = novelty_components(normal_output, seen_ids)[:, 0]
                energy = F.relu(config.energy_margin + normal_energy - boundary_energy).mean()
                cvar = cvar_ranking_loss(normal_energy, boundary_energy, margin=config.energy_margin,
                                         fraction=config.cvar_fraction)
                oe_weight = 0.0 if config.outlier_mode == "none" else config.oe_weight
                cvar_weight = 0.0 if config.outlier_mode == "none" else config.cvar_weight
                loss = real_ce + config.named_weight * named_ce + config.factor_weight * factor + oe_weight * energy + cvar_weight * cvar
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            totals += [float(loss.detach()), float(real_ce.detach()), float(named_ce.detach()), float(factor.detach()),
                       float(energy.detach()), float(cvar.detach())]
        history.append({"epoch": epoch + 1, "loss": totals[0] / config.steps_per_epoch,
                        "real_ce": totals[1] / config.steps_per_epoch, "named_ce": totals[2] / config.steps_per_epoch,
                        "factor": totals[3] / config.steps_per_epoch, "energy": totals[4] / config.steps_per_epoch,
                        "cvar": totals[5] / config.steps_per_epoch})
    metadata = _metadata(model, config, started, history, device)
    metadata["seen_class_ids"] = seen_ids
    metadata["renderer"] = "otdr-event-grammar-v1"
    metadata["renderer_trace_rms_target"] = trace_rms_target
    metadata["renderer_nuisance_source"] = "outer_seen_standardized_training_traces_only"
    metadata["real_outer_heldout_used_by_renderer"] = False
    return model, metadata


@torch.no_grad()
def infer_event_model(
    model: EventCompositionalModel,
    features: torch.Tensor,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2048,
    known_class_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    device = require_cuda(str(device))
    model.eval()
    means, stds = recipe_means.to(device), recipe_stds.to(device)
    rows: dict[str, list[torch.Tensor]] = {}
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for start in range(0, len(features), batch_size):
        x = features[start:start + batch_size].pin_memory().to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            output = model(x, means, stds)
            output["novelty_components"] = novelty_components(output, known_class_ids)
        for key in ("logits", "factor_mean", "factor_std", "embedding", "location", "center",
                    "reconstruction_residual", "novelty_components"):
            rows.setdefault(key, []).append(output[key].float().cpu())
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}
